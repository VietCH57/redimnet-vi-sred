'''
Training data: random crops of the training utterances + on-the-fly augmentation.

Based on the loader of TaoRuijie/ECAPA-TDNN (and its parameterised version in VietCH57/ECAPA-TDNN):
MUSAN noise / babble / music / tv and RIR reverberation with the same probabilities and SNRs.
Additions for fine-tuning ReDimNet:
  * crop length chosen per batch from --crop_sec (e.g. 2,3 s) by CropBatchSampler
  * speed perturbation (0.9 / 1.1) with new speaker classes, as in WeSpeaker / ReDimNet
  * narrow-band channel simulation (8 kHz resampling or a 300-3400 Hz telephone band),
    applied after noise / reverberation with probability --narrowband_prob
The loader returns raw 16 kHz waveforms: ReDimNet computes its features inside the model.
'''

import glob, math, os, random
from fractions import Fraction
import numpy, soundfile, torch
from scipy import signal

SR = 16000
AUG_NAMES = ['clean', 'reverb', 'babble', 'music', 'noise', 'tv']


def read_list(train_list):
    '''(speaker_ids, relative_paths) from a "<spk> <path>" list (space or tab separated).'''
    spks, paths = [], []
    for line in open(train_list).read().splitlines():
        if line.strip():
            parts = line.split()
            spks.append(parts[0]); paths.append(parts[1])
    return spks, paths


def speaker_dict(train_list):
    '''Speaker -> class index (sorted speaker ids, as in the ECAPA repo).'''
    spks, _ = read_list(train_list)
    return {k: i for i, k in enumerate(sorted(set(spks)))}


def find_musan_root(path):
    '''Folder that contains music/ noise/ speech/ (searching up to 3 levels below path).'''
    if not path or not os.path.isdir(path):
        return None
    for root, dirs, _ in os.walk(path):
        if {'music', 'noise', 'speech'}.issubset(set(dirs)):
            return root
        if root[len(path):].count(os.sep) >= 3:
            dirs[:] = []
    return None


def find_rir_root(path):
    '''Prefer a "simulated_rirs" folder below path; otherwise use path itself.'''
    if not path or not os.path.isdir(path):
        return None
    if os.path.basename(os.path.normpath(path)) == 'simulated_rirs':
        return path
    for root, dirs, _ in os.walk(path):
        if 'simulated_rirs' in dirs:
            return os.path.join(root, 'simulated_rirs')
        if root[len(path):].count(os.sep) >= 3:
            dirs[:] = []
    return path


def load_wav(path):
    '''float32 mono waveform at 16 kHz.'''
    audio, sr = soundfile.read(path, dtype='float32', always_2d=True)
    audio = audio.mean(1)
    if sr != SR:
        f = Fraction(SR, sr).limit_denominator(1000)
        audio = signal.resample_poly(audio, f.numerator, f.denominator).astype(numpy.float32)
    return audio


class CropBatchSampler(torch.utils.data.Sampler):
    '''Shuffled batches of (index, crop_samples); the crop length is drawn once per batch so that a
    batch can be stacked. Deterministic for a given (seed, epoch): call set_epoch() every epoch.
    With DDP (world > 1) every rank draws the same global batches and keeps its own slice
    (batch_size = global batch, each rank gets batch_size // world items of the same crop length).'''

    def __init__(self, n_items, batch_size, crop_sec=(2.0,), crop_probs=None, seed=0, drop_last=True, rank=0, world=1):
        assert batch_size % world == 0, '--batch_size (%d) must be divisible by the number of GPUs (%d)' % (batch_size, world)
        self.rank, self.world = rank, world
        self.n, self.bs, self.seed, self.drop_last = n_items, batch_size, seed, drop_last
        self.crops = [int(round(c * SR)) for c in crop_sec]
        p = list(crop_probs) if crop_probs else [1.0] * len(self.crops)
        assert len(p) == len(self.crops), '--crop_probs needs one value per --crop_sec'
        self.p = [x / sum(p) for x in p]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed * 100003 + self.epoch)
        idx = list(range(self.n)); rng.shuffle(idx)
        for b in range(len(self)):
            chunk = idx[b * self.bs:(b + 1) * self.bs]
            L = rng.choices(self.crops, weights=self.p)[0]
            yield [(i, L) for i in chunk[self.rank::self.world]]

    def __len__(self):
        return self.n // self.bs if self.drop_last else math.ceil(self.n / self.bs)


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, train_list, train_path, musan_path='', rir_path='',
                 aug_probs=(1, 1, 1, 1, 1, 1), snr_noise=(0, 15), snr_speech=(13, 20), snr_music=(5, 15),
                 num_noise=(1, 1), num_speech=(3, 8), num_music=(1, 1),
                 speed_perturb=(), speed_probs=None, narrowband_prob=0.0, narrowband_types=('8k', 'tel'),
                 verbose=True, **kwargs):
        # ---- data & labels ----
        self.spk2id = speaker_dict(train_list)
        spks, paths = read_list(train_list)
        self.files = [os.path.join(train_path, p) for p in paths]
        self.labels = [self.spk2id[s] for s in spks]
        self.n_spk = len(self.spk2id)
        # ---- speed perturbation: speed k -> classes [k*n_spk, (k+1)*n_spk) ----
        self.speeds = [1.0] + [float(s) for s in speed_perturb if abs(float(s) - 1.0) > 1e-6]
        sp = list(speed_probs) if speed_probs else [1.0] * len(self.speeds)
        assert len(sp) == len(self.speeds), '--speed_probs needs %d values (1.0 + --speed_perturb)' % len(self.speeds)
        self.speed_p = [x / sum(sp) for x in sp]
        self.n_class = self.n_spk * len(self.speeds)
        # ---- MUSAN / RIR ----
        self.noisesnr = {'noise': list(snr_noise), 'speech': list(snr_speech), 'music': list(snr_music)}
        self.numnoise = {'noise': list(num_noise), 'speech': list(num_speech), 'music': list(num_music)}
        self.noiselist = {}
        musan_root = find_musan_root(musan_path)
        if musan_root is not None:
            for f in glob.glob(os.path.join(musan_root, '**', '*.wav'), recursive=True):
                cat = os.path.relpath(f, musan_root).split(os.sep)[0]
                if cat in self.noisesnr:
                    self.noiselist.setdefault(cat, []).append(f)
        rir_root = find_rir_root(rir_path)
        self.rir_files = glob.glob(os.path.join(rir_root, '**', '*.wav'), recursive=True) if rir_root else []
        self._nframes = {}
        probs = [float(x) for x in aug_probs]
        need = {1: bool(self.rir_files), 2: 'speech' in self.noiselist, 3: 'music' in self.noiselist,
                4: 'noise' in self.noiselist, 5: 'speech' in self.noiselist and 'music' in self.noiselist}
        for k, ok in need.items():
            if not ok and probs[k] > 0:
                if verbose:
                    print('[data] "%s" augmentation disabled: data not found' % AUG_NAMES[k])
                probs[k] = 0.0
        if sum(probs) <= 0:
            probs[0] = 1.0
        self.aug_probs = probs
        self.nb_prob = float(narrowband_prob)
        self.nb_types = list(narrowband_types)
        self._tel = signal.butter(6, [300, 3400], btype='bandpass', fs=SR, output='sos')
        if verbose:
            tot = sum(probs)
            print('[data] %d utterances, %d speakers, %d classes (speeds %s, p=%s)'
                  % (len(self.files), self.n_spk, self.n_class, self.speeds, [round(x, 3) for x in self.speed_p]))
            print('[data] MUSAN root: %s | %s' % (musan_root, {k: len(v) for k, v in self.noiselist.items()}))
            print('[data] RIR root: %s | %d files' % (rir_root, len(self.rir_files)))
            print('[data] augmentation: ' + ', '.join('%s=%.2f' % (n, p / tot) for n, p in zip(AUG_NAMES, probs))
                  + ' | narrow-band p=%.2f %s' % (self.nb_prob, self.nb_types))

    def __len__(self):
        return len(self.files)

    # ---------------------------------------------------------------- crop / speed
    @staticmethod
    def _crop(audio, length, center=False):
        if audio.shape[0] <= length:
            audio = numpy.pad(audio, (0, length - audio.shape[0]), 'wrap')
        start = (audio.shape[0] - length) // 2 if center else int(random.random() * (audio.shape[0] - length))
        return audio[start:start + length]

    @classmethod
    def _speed(cls, audio, speed, length, center=False):
        '''Crop then change the speed (and pitch) by resampling, as sox "speed" / WeSpeaker do.'''
        if abs(speed - 1.0) < 1e-6:
            return cls._crop(audio, length, center)
        f = Fraction(speed).limit_denominator(100)        # speed = down / up
        up, down = f.denominator, f.numerator
        need = int(math.ceil(length * speed)) + 2
        seg = cls._crop(audio, need, center)
        out = signal.resample_poly(seg, up, down)
        return cls._crop(out, length, center) if out.shape[0] != length else out

    def __getitem__(self, item):
        index, length = item if isinstance(item, (tuple, list)) else (item, int(2 * SR))
        audio = load_wav(self.files[index]).astype(numpy.float64)
        k = random.choices(range(len(self.speeds)), weights=self.speed_p)[0] if len(self.speeds) > 1 else 0
        audio = self._crop(audio, length) if k == 0 else self._speed(audio, self.speeds[k], length)
        audio = audio[None]
        aug = random.choices(range(6), weights=self.aug_probs)[0]
        if aug == 1:
            audio = self.add_rev(audio)
        elif aug == 2:
            audio = self.add_noise(audio, 'speech')
        elif aug == 3:
            audio = self.add_noise(audio, 'music')
        elif aug == 4:
            audio = self.add_noise(audio, 'noise')
        elif aug == 5:
            audio = self.add_noise(self.add_noise(audio, 'speech'), 'music')
        if self.nb_prob > 0 and random.random() < self.nb_prob:
            audio = self.narrowband(audio)
        audio = audio[0].astype(numpy.float32)
        if not numpy.isfinite(audio).all():
            audio = numpy.nan_to_num(audio)
        return torch.from_numpy(audio), self.labels[index] + k * self.n_spk

    # ---------------------------------------------------------------- augmentations
    def narrowband(self, audio):
        t = random.choice(self.nb_types)
        L = audio.shape[1]
        if t == '8k':      # resample to 8 kHz and back: removes everything above 4 kHz
            y = signal.resample_poly(signal.resample_poly(audio, 1, 2, axis=1), 2, 1, axis=1)[:, :L]
        elif t == 'tel':   # telephone band 300-3400 Hz
            y = signal.sosfilt(self._tel, audio, axis=1)
        else:
            raise ValueError('unknown narrow-band type %r (8k / tel)' % t)
        if y.shape[1] < L:
            y = numpy.pad(y, ((0, 0), (0, L - y.shape[1])))
        return y

    def add_rev(self, audio):
        rir, _ = soundfile.read(random.choice(self.rir_files), dtype='float64', always_2d=True)
        rir = rir[:, 0][None]
        rir = rir / numpy.sqrt(numpy.sum(rir ** 2) + 1e-12)
        return signal.convolve(audio, rir, mode='full')[:, :audio.shape[1]]

    def _read_segment(self, file, length):
        n = self._nframes.get(file)
        if n is None:
            n = soundfile.info(file).frames
            self._nframes[file] = n
        if n <= length:
            x, _ = soundfile.read(file, dtype='float64', always_2d=True)
            return numpy.pad(x[:, 0], (0, length - x.shape[0]), 'wrap')[:length]
        start = int(random.random() * (n - length))
        x, _ = soundfile.read(file, start=start, stop=start + length, dtype='float64', always_2d=True)
        return x[:, 0]

    def add_noise(self, audio, cat):
        L = audio.shape[1]
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
        lo, hi = self.numnoise[cat]
        pool = self.noiselist[cat]
        noises = []
        for f in random.sample(pool, min(len(pool), random.randint(lo, hi))):
            x = self._read_segment(f, L)[None]
            noise_db = 10 * numpy.log10(numpy.mean(x ** 2) + 1e-4)
            snr = random.uniform(*self.noisesnr[cat])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - snr) / 10)) * x)
        return numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True) + audio


def worker_init(worker_id):
    numpy.random.seed((torch.initial_seed() + worker_id) % 2 ** 32)
