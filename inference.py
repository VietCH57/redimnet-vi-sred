'''
Shared inference code: trial lists, embedding extraction, scoring.

Embeddings
  full  : the whole utterance (files shorter than --eval_min_sec are repeat-padded, longer than
          --eval_max_sec are cut), L2-normalised. Same as the zero-shot step-0 notebook.
  crops : --num_eval evenly spaced crops of --eval_sec seconds, mean of the L2-normalised crops.
  both  : score = mean of the two cosine scores (as in TaoRuijie/ECAPA-TDNN).

Speed: files are sorted by length and batched. With --eval_len_tol 0 only files of exactly the
same length share a batch (results identical to batch size 1). With e.g. 0.05, files whose
lengths differ by <= 5 % share a batch and are cut to the shortest one (at most 5 % of the audio
is dropped) - much faster, used for monitoring during training.

Scoring
  cos        : cosine of the embeddings
  center     : subtract a mean embedding (estimated on training utterances, no labels used),
               re-normalise, cosine. The mean is stored in the saved models (center_mean).
'''

import os, numpy, soundfile, torch, tqdm
import torch.nn.functional as F
from dataLoader import load_wav, SR
from tools import compute_metrics

LABEL_WORDS = {'target': 1, 'nontarget': 0, 'non-target': 0, 'tar': 1, 'non': 0, 'true': 1, 'false': 0}


def parse_trials(path):
    '''"1 a.wav b.wav" | "a.wav b.wav target|nontarget|1|0" | "a.wav b.wav" -> (enroll, test, labels|None)'''
    enroll, test, labels = [], [], []
    for n, line in enumerate(open(path).read().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 2:
            a, b, lab = parts[0], parts[1], None
        elif len(parts) == 3 and parts[0] in ('0', '1'):
            lab, a, b = int(parts[0]), parts[1], parts[2]
        elif len(parts) == 3:
            a, b, w = parts
            if w.lower() in LABEL_WORDS:
                lab = LABEL_WORDS[w.lower()]
            elif w in ('0', '1'):
                lab = int(w)
            else:
                raise ValueError('%s:%d: cannot parse label %r' % (path, n, w))
        else:
            raise ValueError('%s:%d: expected 2 or 3 columns, got %r' % (path, n, line))
        enroll.append(a); test.append(b); labels.append(lab)
    if any(l is None for l in labels):
        labels = [None] * len(labels)
    return enroll, test, labels


def _fit(audio, min_len, max_len):
    if max_len and audio.shape[0] > max_len:
        audio = audio[:max_len]
    if audio.shape[0] < min_len:
        audio = numpy.tile(audio, min_len // max(audio.shape[0], 1) + 1)[:min_len]
    return audio


def _crops(audio, L, num):
    if audio.shape[0] <= L:
        audio = numpy.pad(audio, (0, L - audio.shape[0]), 'wrap')
    starts = numpy.linspace(0, audio.shape[0] - L, num=num)
    return numpy.stack([audio[int(s):int(s) + L] for s in starts])


def _length(path, min_len, max_len):
    info = soundfile.info(path)
    n = int(round(info.frames * SR / info.samplerate))
    n = min(n, max_len) if max_len else n
    return max(n, min_len)


def make_batches(lengths, len_tol=0.0, max_batch=64, max_batch_sec=600):
    '''Group indices sorted by length (longest first). Returns [(indices, crop_length)].'''
    order = sorted(range(len(lengths)), key=lambda i: -lengths[i])
    batches, cur = [], []
    for i in order:
        if cur:
            L0, Li = lengths[cur[0]], lengths[i]
            ok = (L0 - Li) <= len_tol * L0 and len(cur) < max_batch and (len(cur) + 1) * Li <= max_batch_sec * SR
            if not ok:
                batches.append((cur, lengths[cur[-1]])); cur = []
        cur.append(i)
    if cur:
        batches.append((cur, lengths[cur[-1]]))
    return batches


class _FullSet(torch.utils.data.Dataset):
    def __init__(self, paths, batches, min_len, max_len):
        self.paths, self.batches, self.min_len, self.max_len = paths, batches, min_len, max_len

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, b):
        idx, L = self.batches[b]
        x = [_fit(load_wav(self.paths[i]), self.min_len, self.max_len)[:L] for i in idx]
        return idx, torch.from_numpy(numpy.stack(x))


class _CropSet(torch.utils.data.Dataset):
    def __init__(self, paths, group, L, num):
        self.paths, self.group, self.L, self.num = paths, group, L, num
        self.chunks = [list(range(i, min(i + group, len(paths)))) for i in range(0, len(paths), group)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, b):
        idx = self.chunks[b]
        return idx, torch.from_numpy(numpy.concatenate([_crops(load_wav(self.paths[i]), self.L, self.num) for i in idx]))


def _forward(encoder, x, device, amp):
    with torch.autocast('cuda', dtype=torch.float16, enabled=amp and device.type == 'cuda'):
        return F.normalize(encoder(x.to(device, non_blocking=True)).float(), dim=1)


@torch.no_grad()
def extract_embeddings(encoder, files, root, device='cuda', mode='full', eval_sec=3.0, num_eval=5,
                       min_sec=1.0, max_sec=60.0, len_tol=0.0, max_batch=64, max_batch_sec=600,
                       amp=False, workers=4, progress=True):
    '''{file: (emb_full [D] | None, emb_crops_mean [D] | None)} as float32 numpy arrays.'''
    device = torch.device(device)
    encoder.eval()
    paths = [os.path.join(root, f) for f in files]
    D = None
    full = crop = None
    kw = dict(batch_size=None, num_workers=workers, pin_memory=device.type == 'cuda')
    if mode in ('full', 'both'):
        min_len, max_len = int(min_sec * SR), int(max_sec * SR) if max_sec > 0 else 0
        lengths = [_length(p, min_len, max_len) for p in paths]
        batches = make_batches(lengths, len_tol, max_batch, max_batch_sec)
        dl = torch.utils.data.DataLoader(_FullSet(paths, batches, min_len, max_len), **kw)
        for idx, x in (tqdm.tqdm(dl, desc='embed full', mininterval=30) if progress else dl):
            e = _forward(encoder, x, device, amp).cpu().numpy()
            if full is None:
                D = e.shape[1]; full = numpy.zeros((len(files), D), numpy.float32)
            full[idx] = e
    if mode in ('crops', 'both'):
        L = int(eval_sec * SR)
        group = max(1, max_batch // num_eval)
        dl = torch.utils.data.DataLoader(_CropSet(paths, group, L, num_eval), **kw)
        for idx, x in (tqdm.tqdm(dl, desc='embed crops', mininterval=30) if progress else dl):
            e = _forward(encoder, x, device, amp).cpu().numpy().reshape(len(idx), num_eval, -1).mean(1)
            if crop is None:
                crop = numpy.zeros((len(files), e.shape[1]), numpy.float32)
            crop[idx] = e
    return {f: (full[i] if full is not None else None, crop[i] if crop is not None else None)
            for i, f in enumerate(files)}


def mean_embedding(emb):
    '''(mean of full, mean of crops) over all files of an embedding dict -> center for scoring.'''
    vals = list(emb.values())
    mf = numpy.mean([v[0] for v in vals], 0) if vals[0][0] is not None else None
    mc = numpy.mean([v[1] for v in vals], 0) if vals[0][1] is not None else None
    return mf, mc


def _l2n(X):
    return X / numpy.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)


def score_trials(emb, enroll, test, mode='full', center=None):
    '''Cosine scores in trial order. center = (mean_full, mean_crops) or None.'''
    keys = sorted(set(enroll) | set(test))
    k2i = {k: i for i, k in enumerate(keys)}
    a = numpy.array([k2i[x] for x in enroll]); b = numpy.array([k2i[x] for x in test])
    scores = []
    for j, use in ((0, mode in ('full', 'both')), (1, mode in ('crops', 'both'))):
        if not use:
            continue
        E = numpy.stack([emb[k][j] for k in keys]).astype(numpy.float64)
        if center is not None and center[j] is not None:
            E = _l2n(E - center[j])
        scores.append((E[a] * E[b]).sum(1))
    return numpy.mean(scores, axis=0)


def subset_metrics(scores, labels, enroll, test, meta_csv, p_target=0.05):
    '''EER on the subsets of val_trials_meta.csv (split notebook): O, H, X and O by min duration.'''
    import pandas as pd
    meta = pd.read_csv(meta_csv)
    key = pd.Series(range(len(meta)), index=pd.MultiIndex.from_arrays([meta.enroll, meta.test]))
    idx = key.reindex(pd.MultiIndex.from_arrays([enroll, test])).values
    ok = ~numpy.isnan(idx)
    if (~ok).any():
        print('[subset_metrics] %d trials not found in %s -> skipped' % ((~ok).sum(), meta_csv))
    m = meta.iloc[idx[ok].astype(int)].reset_index(drop=True)
    s = numpy.asarray(scores)[ok]; l = numpy.asarray(labels)[ok]
    t = (m.label == 1).values
    sub = {'O': t | (m.type == 'nontarget_rand').values,
           'H': t | (m.type == 'nontarget_samegender').values,
           'X': (t & m.cross_channel.values) | ((m.type == 'nontarget_samegender').values & ~m.cross_channel.values)}
    for name, lo, hi in [('dur<2s', 0, 2), ('dur2-4s', 2, 4), ('dur4-8s', 4, 8), ('dur>=8s', 8, 1e9)]:
        sub['O_' + name] = sub['O'] & (m.min_dur.values >= lo) & (m.min_dur.values < hi)
    res = {}
    for name, mask in sub.items():
        if mask.sum() and 0 < l[mask].sum() < mask.sum():
            res['EER_' + name] = compute_metrics(s[mask], l[mask], [p_target])['EER']
    return res


def save_emb(path, emb):
    keys = sorted(emb)
    full = numpy.stack([emb[k][0] for k in keys]) if emb[keys[0]][0] is not None else numpy.zeros((0,))
    crop = numpy.stack([emb[k][1] for k in keys]) if emb[keys[0]][1] is not None else numpy.zeros((0,))
    numpy.savez(path, files=numpy.array(keys), full=full, crops=crop)


def load_emb(path):
    with numpy.load(path, allow_pickle=False) as z:          # every array is read exactly once
        files, full, crop = list(z['files']), z['full'], z['crops']
    full = full if full.size else None
    crop = crop if crop.size else None
    return {f: (full[i] if full is not None else None, crop[i] if crop is not None else None) for i, f in enumerate(files)}
