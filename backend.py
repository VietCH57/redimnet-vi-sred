'''
Back-end study for a fixed encoder: weight interpolation, in-domain embedding normalisation,
adaptive score normalisation and quality-aware calibration. Everything is selected on the
validation trials (speakers disjoint from training); the test sets are only scored.

  1. Weight interpolation between the zero-shot and the fine-tuned encoder
       theta(alpha) = (1 - alpha) * theta_zero-shot + alpha * theta_fine-tuned
     (weight-space ensemble / WiSE-FT: Wortsman et al., CVPR 2022; for SV: "Robust speaker
     verification using deep weight space ensemble", ICASSP 2023). Same architecture, so the
     inference model keeps its 4.8M parameters.
  2. Embedding normalisation with statistics of in-domain TRAINING utterances
       cos    : cosine of the L2-normalised embeddings
       center : subtract the training mean, L2-normalise (already used in this repo)
       white  : center + whitening with the training total covariance (Garcia-Romero &
                Espy-Wilson, "Analysis of i-vector length normalization", Interspeech 2011)
       wccn   : center + whitening with the training WITHIN-speaker covariance
                (Hatch et al., "Within-class covariance normalization for SVM-based speaker
                recognition", ICSLP 2006)
     Covariances are shrunk towards a scaled identity: S <- (1 - lam) S + lam tr(S)/d I.
  3. Adaptive symmetric score normalisation (AS-norm, Matejka et al., Interspeech 2017;
     Cumani et al. 2011) with a cohort of training-speaker mean embeddings and top-K selection.
  5. Band-limited view (test-time, no parameters): the EDA of the training set shows 23 % of the
     files have <= 4 kHz effective bandwidth (telephone / codec / low-cost microphones) and on
     validation, target trials across bandwidths are clearly harder. The same encoder is run on the
     waveform band-limited to 4 kHz (16k -> 8k -> 16k resampling), and the two L2-normalised
     embeddings can be averaged ("full+lp4k"). The view
     and the fusion are selected on validation like everything else.
  4. Quality measure function (QMF, Thienpondt et al., "The IDLAB VoxSRC-20 submission",
     arXiv:2010.11255): logistic regression on [score, log min duration, log max duration],
     trained on validation trials.

Selection is honest: the validation speakers are split into two folds; the configuration chosen
on one fold is evaluated on the other (and vice versa). The final configuration is then chosen on
all validation trials.

CLI
  python backend.py embed  --model A [--model_b B --alpha 0.5] --files list.txt --root DIR --out x.npz
  python backend.py select --val_emb v.npz --train_emb t.npz --train_list l.txt --trials val.txt --out sel.json
  python backend.py score  --sel sel.json --test_emb x.npz --train_emb t.npz --train_list l.txt
                           --val_emb v.npz --val_trials val.txt --trials test.txt --out results.txt [--zip]
'''

import argparse, itertools, json, os, zipfile
import numpy as np
import torch


# ----------------------------------------------------------------------------------- encoders
def make_encoder(model_a, model_b='', alpha=0.0, verbose=True):
    import models
    enc, extra = models.load_encoder(model_a, verbose=verbose)
    if model_b and alpha > 0:
        enc_b, _ = models.load_encoder(model_b, verbose=verbose)
        sa, sb = enc.net.state_dict(), enc_b.net.state_dict()
        assert sa.keys() == sb.keys(), 'models have different architectures'
        mixed = {k: ((1 - alpha) * sa[k].float() + alpha * sb[k].float()).to(sa[k].dtype)
                 if sa[k].is_floating_point() else sb[k] for k in sa}
        enc.net.load_state_dict(mixed)
        if verbose:
            print('Interpolated weights: (1 - %.2f) * %s + %.2f * %s' % (alpha, model_a, alpha, model_b))
    return enc


class BandLimited(torch.nn.Module):
    '''encoder applied to the waveform low-passed to 4 kHz (resample 16k -> 8k -> 16k).'''

    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, x):
        import torchaudio.functional as AF
        n = x.shape[-1]
        y = AF.resample(AF.resample(x, 16000, 8000), 8000, 16000)[..., :n]
        return self.enc(y)


def bandwidth(path, frac=0.999):
    '''frequency (Hz) below which `frac` of the energy lies (same measure as the EDA notebook: bw999).'''
    import soundfile
    y, sr = soundfile.read(path, dtype='float32')
    if y.ndim > 1:
        y = y.mean(1)
    n = 512
    if len(y) < n:
        return float('nan')
    fr = np.lib.stride_tricks.sliding_window_view(y, n)[::n // 2] * np.hanning(n)
    P = (np.abs(np.fft.rfft(fr, axis=1)) ** 2).sum(0)
    c = np.cumsum(P) / max(P.sum(), 1e-20)
    return float(np.searchsorted(c, frac) * sr / n)


def read_files(path):
    '''one path per line, or "<spk> <path>" -> (paths, spk or None)'''
    rows = [l.split() for l in open(path).read().splitlines() if l.strip()]
    paths = [r[-1] for r in rows]
    spk = [r[0] for r in rows] if all(len(r) > 1 for r in rows) else None
    return paths, spk


def cmd_embed(a):
    import soundfile, torch
    from inference import extract_embeddings
    files, _ = read_files(a.files)
    enc = make_encoder(a.model, a.model_b, a.alpha)
    if a.view == 'lp4k':
        enc = BandLimited(enc)
        print('View: band-limited to 4 kHz')
    dev = a.device if (torch.cuda.is_available() or not a.device.startswith('cuda')) else 'cpu'
    enc = enc.to(dev)
    emb = extract_embeddings(enc, files, a.root, device=dev, mode='full', min_sec=a.min_sec, max_sec=a.max_sec,
                             len_tol=0.0, max_batch=a.batch, amp=a.amp, workers=a.workers)
    E = np.stack([emb[f][0] for f in files]).astype(np.float32)
    dur = np.array([soundfile.info(os.path.join(a.root, f)).duration for f in files], np.float32)
    bw = np.array([bandwidth(os.path.join(a.root, f)) for f in files], np.float32) if a.bandwidth else np.full(len(files), np.nan, np.float32)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, files=np.array(files), emb=E, dur=dur, bw=bw)
    print('Saved %d embeddings to %s' % (len(files), a.out))


def load_npz(path):
    '''one file, or several joined with "+" -> embeddings of the views averaged after L2 normalisation.'''
    parts = path.split('+')
    zs = [np.load(p, allow_pickle=False) for p in parts]
    files = list(zs[0]['files'])
    for z in zs[1:]:
        assert list(z['files']) == files, 'views cover different files'
    E = sum(l2n(z['emb'].astype(np.float64)) for z in zs) / len(zs)
    return files, E, zs[0]['dur'].astype(np.float64)


def load_bw(path):
    z = np.load(path.split('+')[0], allow_pickle=False)
    return z['bw'].astype(np.float64) if 'bw' in z else None


# ----------------------------------------------------------------------------------- back-end
def l2n(X):
    return X / np.linalg.norm(X, axis=-1, keepdims=True).clip(1e-12)


def fit_norm(T, spk, kind, lam):
    '''returns f(X) -> normalised, L2-normalised embeddings; statistics from training embeddings T.'''
    T = l2n(T)
    if kind == 'cos':
        return lambda X: l2n(X)
    mu = T.mean(0)
    if kind == 'center':
        return lambda X: l2n(l2n(X) - mu)
    Tc = T - mu
    if kind == 'white':
        S = Tc.T @ Tc / len(Tc)
    elif kind == 'wccn':
        spk = np.asarray(spk)
        S = np.zeros((T.shape[1], T.shape[1]))
        for s in np.unique(spk):
            Xs = Tc[spk == s]
            Xs = Xs - Xs.mean(0)
            S += Xs.T @ Xs
        S /= len(Tc)
    else:
        raise ValueError(kind)
    d = S.shape[0]
    S = (1 - lam) * S + lam * np.trace(S) / d * np.eye(d)
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
    return lambda X: l2n((l2n(X) - mu) @ W)


def speaker_means(F, spk):
    spk = np.asarray(spk)
    return l2n(np.stack([F[spk == s].mean(0) for s in np.unique(spk)]))


def asnorm_stats(E, cohort, k):
    C = E @ cohort.T
    k = min(k, C.shape[1])
    top = -np.sort(-C, axis=1)[:, :k]
    return top.mean(1), top.std(1).clip(1e-6)


def raw_scores(E, ia, ib):
    return (E[ia] * E[ib]).sum(1)


def eer(scores, labels):
    from tools import eer as _eer
    return _eer(scores, labels)


def qmf_features(s, dur, ia, ib):
    da, db = dur[ia], dur[ib]
    return np.stack([s, np.log(np.minimum(da, db)), np.log(np.maximum(da, db))], 1)


def fit_qmf(X, y):
    from sklearn.linear_model import LogisticRegression
    m = np.mean(X, 0); sd = np.std(X, 0).clip(1e-9)
    lr = LogisticRegression(C=1e3, max_iter=2000).fit((X - m) / sd, y)
    return lambda Z: lr.decision_function((Z - m) / sd)


class Scorer:
    '''Pre-computes everything that depends only on (variant, norm kind, lam) for a set of files.'''

    def __init__(self, E, T, tspk, kind, lam):
        f = fit_norm(T, tspk, kind, lam)
        self.E, self.cohort = f(E), speaker_means(f(T), tspk)
        self._as = {}

    def scores(self, ia, ib, k):
        s = raw_scores(self.E, ia, ib)
        if not k:
            return s
        if k not in self._as:
            self._as[k] = asnorm_stats(self.E, self.cohort, k)
        mu, sd = self._as[k]
        return 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])


def trial_index(files, enroll, test):
    k2i = {f: i for i, f in enumerate(files)}
    missing = [x for x in set(enroll) | set(test) if x not in k2i]
    assert not missing, 'files without embeddings, e.g. %s' % missing[:3]
    return np.array([k2i[x] for x in enroll]), np.array([k2i[x] for x in test])


# ----------------------------------------------------------------------------------- selection
def grid(kinds, lams, ks):
    out = []
    for kind in kinds:
        for lam in (lams if kind in ('white', 'wccn') else [0.0]):
            for k in ks:
                out.append((kind, float(lam), int(k)))
    return out


def cmd_select(a):
    from inference import parse_trials
    enroll, test, labels = parse_trials(a.trials)
    y = np.array(labels)
    tfiles, _ = read_files(a.train_list)
    variants = [v.split('=', 1) for v in a.variants.split(',')]     # name=val.npz:train.npz
    spk_of = lambda f: f.split('/')[0]
    spks = sorted({spk_of(x) for x in enroll + test})
    rng = np.random.RandomState(a.seed)
    fold_of = dict(zip(spks, rng.permutation(len(spks)) % 2))
    fa = np.array([fold_of[spk_of(x)] for x in enroll]); fb = np.array([fold_of[spk_of(x)] for x in test])
    folds = [(fa == f) & (fb == f) for f in (0, 1)]
    cfgs = grid(a.kinds.split(','), [float(x) for x in a.lams.split(',')], [int(x) for x in a.asnorm_k.split(',')])
    rows = []
    for name, paths in variants:
        vpath, tpath = paths.split(':')
        vf, VE, vdur = load_npz(vpath)
        tf, TE, _ = load_npz(tpath)
        assert tf == tfiles, 'train embeddings do not match --train_list'
        _, tspk = read_files(a.train_list)
        ia, ib = trial_index(vf, enroll, test)
        cache = {}
        for kind, lam, k in cfgs:
            key = (kind, lam)
            if key not in cache:
                cache[key] = Scorer(VE, TE, tspk, kind, lam)
            s = cache[key].scores(ia, ib, k)
            row = dict(variant=name, kind=kind, lam=lam, asnorm_k=k, qmf=False,
                       EER=eer(s, y), EER_f0=eer(s[folds[0]], y[folds[0]]), EER_f1=eer(s[folds[1]], y[folds[1]]))
            rows.append(row)
            if a.qmf:   # cross-fitted: train on one fold, score the other
                X = qmf_features(s, vdur, ia, ib)
                q = np.zeros_like(s)
                for f in (0, 1):
                    q[folds[1 - f]] = fit_qmf(X[folds[f]], y[folds[f]])(X[folds[1 - f]])
                rows.append(dict(variant=name, kind=kind, lam=lam, asnorm_k=k, qmf=True, EER=np.nan,
                                 EER_f0=eer(q[folds[0]], y[folds[0]]), EER_f1=eer(q[folds[1]], y[folds[1]])))
        print('%s: %d configurations done' % (name, len(cfgs)), flush=True)
    # honest estimate: choose on one fold, report on the other
    def best(rs, col):
        return min(rs, key=lambda r: r[col])
    pick0, pick1 = best(rows, 'EER_f0'), best(rows, 'EER_f1')
    honest = 0.5 * (pick0['EER_f1'] + pick1['EER_f0'])
    # final choice on all validation trials (QMF rows: mean of the two cross-fitted folds)
    for r in rows:      # same trials for every configuration: the within-fold trials of both folds
        r['EER_sel'] = 0.5 * (r['EER_f0'] + r['EER_f1'])
    final = best(rows, 'EER_sel')
    base = [r for r in rows if r['variant'] == variants[0][0] and r['kind'] == 'center' and r['asnorm_k'] == 0 and not r['qmf']]
    res = dict(final=final, chosen_on_fold0=pick0, chosen_on_fold1=pick1, honest_cross_fold_EER=honest,
               reference_center=base[0] if base else None, n_configs=len(rows), rows=rows)
    res['conditions'] = conditions(final, variants, a, enroll, test, y, tfiles)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    for c in res['conditions']:
        print('  %-30s targets %6d  non-targets %6d  EER %6.3f%% (chosen, before QMF)  %6.3f%% (alpha 0, center)' % (c['condition'], c['n_tar'], c['n_non'], c['EER'], c['EER_ref']))
    print('Final configuration (all within-fold val trials):', {k: final[k] for k in ('variant', 'kind', 'lam', 'asnorm_k', 'qmf', 'EER_sel')})
    print('Honest cross-fold estimate of the selection procedure: EER %.3f%%' % honest)


def conditions(final, variants, a, enroll, test, y, tfiles):
    '''EER of the chosen and of the reference (first variant, center) configuration per validation condition.'''
    _, tspk = read_files(a.train_list)
    vmap = dict(variants)

    def scores_for(r):
        vpath, tpath = vmap[r['variant']].split(':')
        vf, VE, vdur = load_npz(vpath); _, TE, _ = load_npz(tpath)
        ia, ib = trial_index(vf, enroll, test)
        return Scorer(VE, TE, tspk, r['kind'], r['lam']).scores(ia, ib, r['asnorm_k']), vdur, load_bw(vpath), ia, ib
    s1, dur, bw, ia, ib = scores_for(final)
    ref = dict(variant=variants[0][0], kind='center', lam=0.0, asnorm_k=0)
    s0 = scores_for(ref)[0]
    md = np.minimum(dur[ia], dur[ib])
    conds = [('all', np.ones_like(y, bool))]
    if bw is not None and np.isfinite(bw).all():
        nb = bw <= 4100
        same = nb[ia] == nb[ib]
        conds += [('same bandwidth class', same), ('cross bandwidth (<=4k vs >4k)', ~same),
                  ('both narrow-band', nb[ia] & nb[ib])]
    conds += [('min dur < 2 s', md < 2), ('min dur 2-4 s', (md >= 2) & (md < 4)), ('min dur >= 4 s', md >= 4)]
    out = []
    for name, m in conds:
        # condition defined on target trials; all non-targets kept (as in the earlier diagnosis)
        mm = m | (y == 0) if name.startswith(('same', 'cross', 'both')) else m
        if (y[mm] == 1).sum() < 20 or (y[mm] == 0).sum() < 20:
            continue
        out.append(dict(condition=name, n_tar=int((y[mm] == 1).sum()), n_non=int((y[mm] == 0).sum()),
                        EER=eer(s1[mm], y[mm]), EER_ref=eer(s0[mm], y[mm])))
    return out


def cmd_score(a):
    from inference import parse_trials
    sel = json.load(open(a.sel))['final']
    tfiles, tspk = read_files(a.train_list)
    tf, TE, _ = load_npz(a.train_emb)
    assert tf == tfiles
    files, XE, xdur = load_npz(a.test_emb)
    enroll, test, labels = parse_trials(a.trials)
    sc = Scorer(XE, TE, tspk, sel['kind'], sel['lam'])
    ia, ib = trial_index(files, enroll, test)
    s = sc.scores(ia, ib, sel['asnorm_k'])
    if sel['qmf']:
        ve, vt, vl = parse_trials(a.val_trials)
        vf, VE, vdur = load_npz(a.val_emb)
        vs = Scorer(VE, TE, tspk, sel['kind'], sel['lam'])
        va, vb = trial_index(vf, ve, vt)
        q = fit_qmf(qmf_features(vs.scores(va, vb, sel['asnorm_k']), vdur, va, vb), np.array(vl))
        s = q(qmf_features(s, xdur, ia, ib))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        for x, t, v in zip(enroll, test, s):
            f.write('%s %s %.6f\n' % (x, t, v))
    print('Wrote %d scores to %s with %s' % (len(s), a.out, {k: sel[k] for k in ('variant', 'kind', 'lam', 'asnorm_k', 'qmf')}))
    if a.zip:
        zp = os.path.join(os.path.dirname(os.path.abspath(a.out)), 'results.zip')
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(a.out, 'results.txt')
        print('Wrote', zp)
    if labels[0] is not None and a.report:
        print('EER on these trials (report only, not used for any choice): %.4f%%' % eer(s, np.array(labels)))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('embed')
    e.add_argument('--model', required=True); e.add_argument('--model_b', default=''); e.add_argument('--alpha', type=float, default=0.0)
    e.add_argument('--files', required=True); e.add_argument('--root', required=True); e.add_argument('--out', required=True)
    e.add_argument('--min_sec', type=float, default=1.0); e.add_argument('--max_sec', type=float, default=60.0)
    e.add_argument('--batch', type=int, default=64); e.add_argument('--amp', action='store_true')
    e.add_argument('--workers', type=int, default=2); e.add_argument('--device', default='cuda')
    e.add_argument('--view', default='full', choices=['full', 'lp4k']); e.add_argument('--bandwidth', type=int, default=1)
    s = sub.add_parser('select')
    s.add_argument('--variants', required=True, help='name=val.npz:train.npz,name2=...  (first one = reference)')
    s.add_argument('--train_list', required=True); s.add_argument('--trials', required=True); s.add_argument('--out', required=True)
    s.add_argument('--kinds', default='cos,center,white,wccn'); s.add_argument('--lams', default='0,0.1,0.3,0.5')
    s.add_argument('--asnorm_k', default='0,100,200,300,500,1000'); s.add_argument('--qmf', type=int, default=1)
    s.add_argument('--seed', type=int, default=0)
    c = sub.add_parser('score')
    c.add_argument('--sel', required=True); c.add_argument('--test_emb', required=True); c.add_argument('--trials', required=True)
    c.add_argument('--train_emb', required=True); c.add_argument('--train_list', required=True)
    c.add_argument('--val_emb', default=''); c.add_argument('--val_trials', default='')
    c.add_argument('--out', required=True); c.add_argument('--zip', action='store_true'); c.add_argument('--report', type=int, default=1)
    a = p.parse_args()
    {'embed': cmd_embed, 'select': cmd_select, 'score': cmd_score}[a.cmd](a)


if __name__ == '__main__':
    main()
