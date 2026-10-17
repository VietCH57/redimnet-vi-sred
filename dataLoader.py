'''
Training data: random crops of the training utterances + on-the-fly augmentation, following the
official ReDimNet2 training pipeline (PalabraAI/redimnet2, applications/speaker_rec/wespeaker_lite,
a fork of WeSpeaker). The loader returns raw 16 kHz waveforms: ReDimNet computes its features inside
the model.

Pipeline per utterance (wespeaker_lite/dataset/dataset.py: Dataset):
  1. files shorter than --min_dur are skipped               (filter_args.min_num_frames 100 = 1.0 s)
  2. speed perturbation: speed drawn uniformly from {1.0, *--speed_perturb}; speed k is a new speaker
     class (label + k * n_speakers)                          (processor.speed_perturb, speeds 1.0/0.9/1.1)
  3. random crop of --crop_sec seconds, short files wrap-padded (processor.random_chunk /
     read_audio_section extend_to_duration). Official crop length = ((num_frms - 1) * 10 ms + 25 ms):
     200 frames -> 2.015 s (pre-training), 600 frames -> 6.015 s (large-margin fine-tuning)
  4. augmentation, one of the official chains (--aug_chain):
     wespeaker : upstream WeSpeaker processor.add_reverb_noise (wenet-e2e/wespeaker
                 wespeaker/dataset/processor.py), the pipeline ReDimNet was trained with (ReDimNet README:
                 "For model training we used wespeaker pipeline"; conf/redimnet.yaml aug_prob 0.6): with prob
                 --aug_prob either reverb or additive noise (1/2 each); the noise file is drawn uniformly over
                 all MUSAN files, SNR noise 0-15 / speech 10-30 / music 5-15 dB; output peak-normalised.
     wespeaker_lite : processor.add_reverb_noise_gain (`aug_setup.type: wespeaker_augs`, used by the
                 VoxCeleb2 recipes, e.g. b4/ptn.yaml and b4/lm.yaml): with prob --aug_prob (0.6) one of
                 {reverb+noise 0.25, reverb 0.10, noise 0.65}; noise category uniform over
                 noise / speech / babble / music with the 'default' SNR preset; reverb before noise;
                 then a random gain (uniform dB in --gain_db, prob --gain_prob); peak protection.
     hard      : the Sequential tree of the VoxBlink2+VoxCeleb2+CN-Celeb2 recipes
                 (examples/vb2_vox2_cnc2/recipes/redimnet2/b6/ptn_ha / lm): with prob --hard_prob
                 [OneOf(p 0.8){music SNR 2-12, noise 0-12, babble 3-6 speakers SNR 7-18}
                  -> Reverb(p 0.2, RIR cut to 1 s) -> LowPass(p 0.25, cutoff 0.6 * Nyquist, width 0.08)]
                 (augs.CustomNoise / Reverb / LowPassAug).
     none      : no augmentation.
Crops are drawn per batch from --crop_sec (one length per batch so that a batch can be stacked).
Augmentation and pipeline logic ported from the files above: Apache License 2.0, (c) 2021-2023 the
WeSpeaker authors (Binbin Zhang, Chengdong Liang, Hongji Wang, Zhengyang Chen) and PalabraAI.
'''

import glob, math, os, random
from fractions import Fraction
import numpy, soundfile, torch
from scipy import signal

SR = 16000

# processor.add_reverb_noise_gain presets (wespeaker_lite)
SNR_PRESETS = {
    'default': {'noise': [0, 15], 'music': [5, 15], 'speech': [13, 20], 'bubble': [13, 20]},
    'medium':  {'noise': [0, 12], 'music': [3, 15], 'speech': [5, 18], 'bubble': [5, 18]},
    'hard':    {'noise': [0, 12], 'music': [1, 12], 'speech': [2, 12], 'bubble': [2, 12]},
}
AUG_TYPE_PROB = [0.25, 0.1, 0.65]                                         # reverb+noise, reverb, noise
NUM_NOISE = {'noise': [1, 1], 'speech': [1, 1], 'bubble': [3, 7], 'music': [1, 1]}


def read_list(train_list):
    '''(speaker_ids, relative_paths) from a "<spk> <path>" list (space or tab separated).'''
    spks, paths = [], []
    for line in open(train_list).read().splitlines():
        if line.strip():
            parts = line.split()
            spks.append(parts[0]); paths.append(parts[1])
    return spks, paths


def speaker_dict(train_list):
    '''Speaker -> class index (sorted speaker ids, deterministic).'''
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
    '''Prefer a "simulated_rirs" folder below path (the official recipes use the 60,000 simulated RIRs).'''
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


def wav_duration(path):
    info = soundfile.info(path)
    return info.frames / info.samplerate


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


def _random_chunk(x, n):
    '''processor.get_random_chunk: random window of n samples, tiled if shorter.'''
    if x.shape[0] >= n:
        s = random.randint(0, x.shape[0] - n)
        return x[s:s + n]
    return numpy.tile(x, n // max(x.shape[0], 1) + 1)[:n]


class TrainSet(torch.utils.data.Dataset):
    def __init__(self, train_list, train_path, musan_path='', rir_path='', min_dur=1.0,
                 speed_perturb=(), speed_probs=None,
                 aug_chain='wespeaker_lite', aug_prob=0.6, aug_preset='default', gain_prob=1.0, gain_db=(-18.0, 6.0),
                 hard_prob=0.9, hard_noise_prob=0.8, hard_reverb_prob=0.2, lowpass_prob=0.25,
                 lowpass_cutoffs=(0.6,), lowpass_width=0.08,
                 verbose=True, **kwargs):
        # ---- data & labels ----
        self.spk2id = speaker_dict(train_list)
        spks, paths = read_list(train_list)
        files = [os.path.join(train_path, p) for p in paths]
        labels = [self.spk2id[s] for s in spks]
        self.n_skipped = 0
        if min_dur and min_dur > 0:
            keep = [i for i, f in enumerate(files) if wav_duration(f) >= min_dur]
            self.n_skipped = len(files) - len(keep)
            files, labels = [files[i] for i in keep], [labels[i] for i in keep]
        self.files, self.labels = files, labels
        self.n_spk = len(self.spk2id)
        # ---- speed perturbation: speed k -> classes [k*n_spk, (k+1)*n_spk) ----
        self.speeds = [1.0] + [float(s) for s in speed_perturb if abs(float(s) - 1.0) > 1e-6]
        sp = list(speed_probs) if speed_probs else [1.0] * len(self.speeds)
        assert len(sp) == len(self.speeds), '--speed_probs needs %d values (1.0 + --speed_perturb)' % len(self.speeds)
        self.speed_p = [x / sum(sp) for x in sp]
        self.n_class = self.n_spk * len(self.speeds)
        # ---- MUSAN / RIR ----
        self.noiselist = {}
        musan_root = find_musan_root(musan_path)
        if musan_root is not None:
            for f in glob.glob(os.path.join(musan_root, '**', '*.wav'), recursive=True):
                cat = os.path.relpath(f, musan_root).split(os.sep)[0]
                if cat in ('noise', 'music', 'speech'):
                    self.noiselist.setdefault(cat, []).append(f)
        rir_root = find_rir_root(rir_path)
        self.rir_files = sorted(glob.glob(os.path.join(rir_root, '**', '*.wav'), recursive=True)) if rir_root else []
        self._nframes = {}
        # ---- augmentation chain ----
        assert aug_chain in ('wespeaker', 'wespeaker_lite', 'hard', 'none'), aug_chain
        assert aug_preset in SNR_PRESETS, aug_preset
        self.chain = aug_chain
        self.aug_prob, self.snr = float(aug_prob), SNR_PRESETS[aug_preset]
        self.gain_prob, self.gain_db = float(gain_prob), tuple(gain_db)
        self.hard_prob, self.hard_noise_prob, self.hard_reverb_prob = float(hard_prob), float(hard_noise_prob), float(hard_reverb_prob)
        self.lp_prob, self.lp_cutoffs, self.lp_width = float(lowpass_prob), list(lowpass_cutoffs), float(lowpass_width)
        missing = [c for c in ('noise', 'music', 'speech') if c not in self.noiselist]
        if self.chain != 'none' and (missing or not self.rir_files):
            raise SystemExit('--aug_chain %s needs MUSAN (music/ noise/ speech/) and RIRs: missing %s%s. '
                             'Use --aug_chain none to train without augmentation.'
                             % (self.chain, missing, '' if self.rir_files else ' + RIR files'))
        if verbose:
            print('[data] %d utterances (%d shorter than %.2f s skipped), %d speakers, %d classes (speeds %s, p=%s)'
                  % (len(self.files), self.n_skipped, min_dur or 0, self.n_spk, self.n_class, self.speeds,
                     [round(x, 3) for x in self.speed_p]))
            print('[data] MUSAN root: %s | %s' % (musan_root, {k: len(v) for k, v in self.noiselist.items()}))
            print('[data] RIR root: %s | %d files' % (rir_root, len(self.rir_files)))
            if self.chain == 'wespeaker':
                print('[data] aug chain wespeaker (upstream): p=%.2f, reverb or noise 1/2 each' % self.aug_prob)
            elif self.chain == 'wespeaker_lite':
                print('[data] aug chain wespeaker_lite: p=%.2f, preset %s %s, gain p=%.2f %s dB'
                      % (self.aug_prob, aug_preset, self.snr, self.gain_prob, self.gain_db))
            elif self.chain == 'hard':
                print('[data] aug chain hard: p=%.2f [noise p=%.2f, reverb p=%.2f, low-pass p=%.2f cutoffs %s]'
                      % (self.hard_prob, self.hard_noise_prob, self.hard_reverb_prob, self.lp_prob, self.lp_cutoffs))
            else:
                print('[data] no augmentation')

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
        '''Crop then change the speed (and pitch) by resampling, as sox "speed" /
        torchaudio.functional.speed in the official pipeline do.'''
        if abs(speed - 1.0) < 1e-6:
            return cls._crop(audio, length, center)
        f = Fraction(speed).limit_denominator(100)        # speed = down / up
        up, down = f.denominator, f.numerator
        need = int(math.ceil(length * speed)) + 2
        seg = cls._crop(audio, need, center)
        out = signal.resample_poly(seg, up, down)
        return cls._crop(out, length, center) if out.shape[0] != length else out

    def __getitem__(self, item):
        index, length = item if isinstance(item, (tuple, list)) else (item, int(2.015 * SR))
        audio = load_wav(self.files[index]).astype(numpy.float64)
        k = random.choices(range(len(self.speeds)), weights=self.speed_p)[0] if len(self.speeds) > 1 else 0
        audio = self._crop(audio, length) if k == 0 else self._speed(audio, self.speeds[k], length)
        if self.chain == 'wespeaker':
            audio = self.wespeaker_v1_chain(audio)
        elif self.chain == 'wespeaker_lite':
            audio = self.wespeaker_chain(audio)
        elif self.chain == 'hard':
            audio = self.hard_chain(audio)
        audio = audio.astype(numpy.float32)
        if not numpy.isfinite(audio).all():
            audio = numpy.nan_to_num(audio)
        return torch.from_numpy(audio), self.labels[index] + k * self.n_spk

    # ---------------------------------------------------------------- I/O helpers
    def _read_segment(self, file, length):
        '''Random window of `length` samples of a noise file (tiled if shorter); same distribution as
        reading the whole file and taking get_random_chunk, without decoding minutes of music.'''
        n = self._nframes.get(file)
        if n is None:
            n = soundfile.info(file).frames
            self._nframes[file] = n
        if n <= length:
            x, _ = soundfile.read(file, dtype='float64', always_2d=True)
            return _random_chunk(x[:, 0], length)
        start = random.randint(0, n - length)
        x, _ = soundfile.read(file, start=start, stop=start + length, dtype='float64', always_2d=True)
        return x[:, 0]

    def _rir(self, max_sec=None):
        rir, sr = soundfile.read(random.choice(self.rir_files), dtype='float64', always_2d=True)
        rir = rir[:, 0]
        if sr != SR:
            rir = signal.resample(rir, int(len(rir) / sr * SR))
        if max_sec:
            rir = rir[:int(max_sec * SR)]
        return rir / numpy.sqrt(numpy.sum(rir ** 2) + 1e-12)

    # ---------------------------------------------------------------- official chains
    WESPEAKER_SNR = {'noise': (0, 15), 'speech': (10, 30), 'music': (5, 15)}

    def wespeaker_v1_chain(self, audio):
        '''Upstream WeSpeaker processor.add_reverb_noise.'''
        if not self.aug_prob > random.random():
            return audio
        L = audio.shape[0]
        if random.randint(1, 2) == 1:
            out = signal.convolve(audio, self._rir(), mode='full')[:L]
        else:
            if not hasattr(self, '_all_noise'):
                self._all_noise = [(c, f) for c in ('noise', 'speech', 'music') for f in self.noiselist[c]]
            cat, f = random.choice(self._all_noise)
            audio_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
            noise = self._read_segment(f, L)
            noise_db = 10 * numpy.log10(numpy.mean(noise ** 2) + 1e-4)
            snr = random.uniform(*self.WESPEAKER_SNR[cat])
            out = audio + numpy.sqrt(10 ** ((audio_db - noise_db - snr) / 10)) * noise
        return out / (numpy.max(numpy.abs(out)) + 1e-4)

    def wespeaker_chain(self, audio):
        '''wespeaker_lite processor.add_reverb_noise_gain (speech_rms_mode full, peak_protect True).'''
        L = audio.shape[0]
        modified = False
        if self.aug_prob > random.random():
            t = random.choices([0, 1, 2], weights=AUG_TYPE_PROB, k=1)[0]
            if t in (0, 1):                                    # reverb first (noise is at the receiver)
                audio = signal.convolve(audio, self._rir(), mode='full')[:L]
                modified = True
            if t in (0, 2):
                cat = random.choice(list(NUM_NOISE))
                src = 'speech' if cat == 'bubble' else cat
                lo, hi = NUM_NOISE[cat]
                k = numpy.random.randint(lo, hi) if lo < hi else hi
                noise = [self._read_segment(random.choice(self.noiselist[src]), L) for _ in range(k)]
                noise = numpy.mean(noise, axis=0) if k > 1 else noise[0]
                snr = random.uniform(*self.snr[cat])
                ps, pn = float(numpy.dot(audio, audio) / L), float(numpy.dot(noise, noise) / L)
                if ps > 0 and pn > 0:
                    audio = audio + numpy.sqrt(10 ** (-snr / 10) * ps / pn) * noise
                    modified = True
        if self.gain_prob > random.random():
            audio = audio * 10.0 ** (random.uniform(*self.gain_db) / 20.0)
            modified = True
        if modified:
            peak = float(numpy.max(numpy.abs(audio)))
            if peak > 1.0:
                audio = audio * (0.99 / peak)
        return audio

    def _custom_noise(self, audio, cat, snr, babble=False, eps=1e-5):
        '''augs.CustomNoise.'''
        L = audio.shape[0]
        k = numpy.random.randint(3, 7) if babble else 1
        noise = numpy.mean([self._read_segment(random.choice(self.noiselist[cat]), L) for _ in range(k)], axis=0)
        s = random.uniform(*snr)
        pa, pn = numpy.mean(audio ** 2) + eps, numpy.mean(noise ** 2) + eps
        return audio + noise * numpy.sqrt(10 ** ((10 * numpy.log10(pa) - 10 * numpy.log10(pn) - s) / 10))

    def _lowpass(self, audio):
        '''augs.LowPassAug: smooth (raised-cosine) low-pass in the frequency domain.'''
        cutoff_ratio = random.choice(self.lp_cutoffs)
        T = audio.shape[0]
        spec = numpy.fft.rfft(audio)
        freqs = numpy.fft.rfftfreq(T)
        cutoff, tw = cutoff_ratio * freqs[-1], self.lp_width * freqs[-1]
        mask = numpy.ones_like(freqs)
        mask[freqs > cutoff + tw / 2] = 0
        tr = (freqs >= cutoff - tw / 2) & (freqs <= cutoff + tw / 2)
        mask[tr] = 0.5 * (1 + numpy.cos(numpy.pi * (freqs[tr] - cutoff) / tw))
        return numpy.fft.irfft(spec * mask, n=T)

    def hard_chain(self, audio):
        '''Sequential tree of the vb2_vox2_cnc2 recipes (augs.Sequential / OneOf; Augmentor applies with
        probability `prob` via np.random.rand() <= prob).'''
        if not numpy.random.rand() <= self.hard_prob:
            return audio
        if numpy.random.rand() <= self.hard_noise_prob:
            c = numpy.random.randint(0, 3)
            if c == 0:
                audio = self._custom_noise(audio, 'music', (2, 12))
            elif c == 1:
                audio = self._custom_noise(audio, 'noise', (0, 12))
            else:
                audio = self._custom_noise(audio, 'speech', (7, 18), babble=True)
        if numpy.random.rand() <= self.hard_reverb_prob:
            audio = signal.convolve(audio, self._rir(max_sec=1.0), mode='full')[:audio.shape[0]]
        if self.lp_prob > 0 and numpy.random.rand() <= self.lp_prob:
            audio = self._lowpass(audio)
        return audio


def worker_init(worker_id):
    s = (torch.initial_seed() + worker_id) % 2 ** 32
    numpy.random.seed(s)
    random.seed(s)
