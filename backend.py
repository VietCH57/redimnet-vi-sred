'''
Back-end study for a fixed encoder: weight interpolation, in-domain embedding normalisation,
adaptive score normalisation and quality-aware calibration. Everything is selected on the
validation trials (speakers disjoint from training); the test sets are only scored.

  1. Weight interpolation between the zero-shot and the fine-tuned encoder
       theta(alpha) = (1 - alpha) * theta_zero-shot + alpha * theta_fine-tuned
     (weight-space ensemble / WiSE-FT: Wortsman et al., "Robust fine-tuning of zero-shot models",
     CVPR 2022). Same architecture, so the inference model keeps its number of parameters.
  2. Embedding normalisation with statistics of in-domain TRAINING utterances
       cos    : cosine of the L2-normalised embeddings
       center : subtract the training mean, L2-normalise (already used in this repo)
       white  : center + whitening with the training total covariance (Garcia-Romero &
                Espy-Wilson, "Analysis of i-vector length normalization", Interspeech 2011)
       wccn   : center + whitening with the training WITHIN-speaker covariance
                (Hatch et al., "Within-class covariance normalization for SVM-based speaker
                recognition", ICSLP 2006)
     Covariances are shrunk towards a scaled identity: S <- (1 - lam) S + lam tr(S)/d I.
  3. Adaptive symmetric score normalisation (AS-norm, Matejka et al., Interspeech 2017) with top-K
     cohort selection. Cohort = training-speaker mean embeddings (WeSpeaker tools/vector_mean.py +
     wespeaker/bin/score_norm.py, top_n 300 in examples/voxceleb/v2/run.sh) or all training utterances.
  5. Band-limited view (test-time, no parameters): the EDA of the training set shows 23 % of the
     files have <= 4 kHz effective bandwidth (telephone / codec / low-cost microphones) and on
     validation, target trials across bandwidths are clearly harder. The same encoder is run on the
     waveform band-limited to 4 kHz (16k -> 8k -> 16k resampling), and the two L2-normalised
     embeddings can be averaged ("full+lp4k"). The view
     and the fusion are selected on validation like everything else.
  4. Quality measure function (QMF, Thienpondt et al., "The IDLAB VoxSRC-20 submission",
     arXiv:2010.11255): logistic regression on the score and quality measures, trained on validation
     trials: 'dur' = [score, log min duration, log max duration]; 'dur_imp' adds the imposter means
     (mean of the top-K cohort scores) of the two sides (min, max), as in IDLab's QMF.

Selection is honest: the validation speakers are split into two folds; the configuration chosen
on one fold is evaluated on the other (and vice versa). The final configuration is then chosen on
all validation trials.

CLI
  python backend.py embed  --model A [--model_b B --alpha 0.5] --files list.txt --root DIR --out x.npz
  python backend.py select --variants a1=val.npz:train.npz[,a0.5=...] --train_list l.txt --trials val.txt --out sel.json
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
    tmp = a.out + '.tmp.npz'            # written then renamed: an interrupted run never leaves a partial a.out
    np.savez(tmp, files=np.array(files), emb=E, dur=dur, bw=bw)
    os.replace(tmp, a.out)
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
    """returns f(X) -> normalised, L2-normalised embeddings; statistics from training embeddings T."""
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
        _, inv = np.unique(np.asarray(spk), return_inverse=True)
        M = np.zeros((inv.max() + 1, T.shape[1])); np.add.at(M, inv, Tc)
        M /= np.bincount(inv)[:, None]
        W_ = Tc - M[inv]
        S = W_.T @ W_ / len(Tc)
    else:
        raise ValueError(kind)
    d = S.shape[0]
    S = (1 - lam) * S + lam * np.trace(S) / d * np.eye(d)
    w, V = np.linalg.eigh(S)
    W = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-10))) @ V.T
    return lambda X: l2n((l2n(X) - mu) @ W)


def speaker_means(F, spk):
    """L2-normalised mean embedding per speaker (WeSpeaker tools/vector_mean.py)."""
    _, inv = np.unique(np.asarray(spk), return_inverse=True)
    M = np.zeros((inv.max() + 1, F.shape[1])); np.add.at(M, inv, F)
    return l2n(M)


def topk_sorted(E, C, kmax, chunk=1024):
    """per row of E: the kmax largest cosine scores against the cohort C, sorted in decreasing order."""
    k = min(kmax, len(C))
    out = np.empty((len(E), k), np.float32)
    for i in range(0, len(E), chunk):
        S = (E[i:i + chunk] @ C.T).astype(np.float32)
        if k < S.shape[1]:
            S = np.take_along_axis(S, np.argpartition(-S, k - 1, axis=1)[:, :k], 1)
        out[i:i + chunk] = -np.sort(-S, axis=1)
    return out


def raw_scores(E, ia, ib):
    return (E[ia] * E[ib]).sum(1)


def eer(scores, labels):
    from tools import eer as _eer
    return _eer(scores, labels)


IMP_K = 300     # imposter mean for QMF 'dur_imp' when the configuration has no AS-norm (WeSpeaker top_n)


class Scorer:
    """Everything that depends on (embeddings, norm kind, lam, cohort type) for a set of files."""

    def __init__(self, E, T, tspk, kind, lam, cohort='spk', kmax=1000):
        f = fit_norm(T, tspk, kind, lam)
        self.E = f(E)
        FT = f(T)
        self.C = speaker_means(FT, tspk) if cohort == 'spk' else FT
        self.kmax = kmax
        self._top = None

    def top(self):
        if self._top is None:
            self._top = topk_sorted(self.E, self.C, self.kmax)
        return self._top

    def stats(self, k):
        t = self.top()[:, :min(k, self.top().shape[1])].astype(np.float64)
        return t.mean(1), t.std(1).clip(1e-6)

    def scores(self, ia, ib, k):
        s = raw_scores(self.E, ia, ib)
        if not k:
            return s
        mu, sd = self.stats(k)
        return 0.5 * ((s - mu[ia]) / sd[ia] + (s - mu[ib]) / sd[ib])

    def qmf_features(self, s, dur, ia, ib, k, feats):
        da, db = dur[ia], dur[ib]
        cols = [s, np.log(np.minimum(da, db)), np.log(np.maximum(da, db))]
        if feats == 'dur_imp':
            imp = self.stats(k or IMP_K)[0]
            cols += [np.minimum(imp[ia], imp[ib]), np.maximum(imp[ia], imp[ib])]
        return np.stack(cols, 1)


def fit_qmf(X, y):
    from sklearn.linear_model import LogisticRegression
    m = np.mean(X, 0); sd = np.std(X, 0).clip(1e-9)
    lr = LogisticRegression(C=1e3, max_iter=2000).fit((X - m) / sd, y)
    return lambda Z: lr.decision_function((Z - m) / sd)


def trial_index(files, enroll, test):
    k2i = {f: i for i, f in enumerate(files)}
    missing = [x for x in set(enroll) | set(test) if x not in k2i]
    assert not missing, 'files without embeddings, e.g. %s' % missing[:3]
    return np.array([k2i[x] for x in enroll]), np.array([k2i[x] for x in test])


def score_config(cfg, x_npz, train_npz, train_list, enroll, test, val_npz='', val_trials=''):
    """Scores of the trials (enroll, test) with a configuration from `select` (QMF trained on all
    validation trials). Used by `score` and by the notebooks."""
    from inference import parse_trials
    _, tspk = read_files(train_list)
    tf, TE, _ = load_npz(train_npz)
    files, XE, xdur = load_npz(x_npz)
    sc = Scorer(XE, TE, tspk, cfg['kind'], cfg['lam'], cfg.get('cohort', 'spk'))
    ia, ib = trial_index(files, enroll, test)
    k = cfg['asnorm_k']
    s = sc.scores(ia, ib, k)
    if cfg['qmf']:
        ve, vt, vl = parse_trials(val_trials)
        vf, VE, vdur = load_npz(val_npz)
        vs = Scorer(VE, TE, tspk, cfg['kind'], cfg['lam'], cfg.get('cohort', 'spk'))
        va, vb = trial_index(vf, ve, vt)
        q = fit_qmf(vs.qmf_features(vs.scores(va, vb, k), vdur, va, vb, k, cfg['qmf']), np.array(vl))
        s = q(sc.qmf_features(s, xdur, ia, ib, k, cfg['qmf']))
    return s


# ----------------------------------------------------------------------------------- selection
def cmd_select(a):
    from inference import parse_trials
    enroll, test, labels = parse_trials(a.trials)
    y = np.array(labels)
    tfiles, tspk = read_files(a.train_list)
    variants = [v.split('=', 1) for v in a.variants.split(',')]     # name=val.npz:train.npz
    spk_of = lambda f: f.split('/')[0]
    spks = sorted({spk_of(x) for x in enroll + test})
    rng = np.random.RandomState(a.seed)
    fold_of = dict(zip(spks, rng.permutation(len(spks)) % 2))
    fa = np.array([fold_of[spk_of(x)] for x in enroll]); fb = np.array([fold_of[spk_of(x)] for x in test])
    folds = [(fa == f) & (fb == f) for f in (0, 1)]
    kinds = a.kinds.split(','); lams = [float(x) for x in a.lams.split(',')]
    ks = [int(x) for x in a.asnorm_k.split(',')]; cohorts = a.cohorts.split(',')
    qmfs = [''] + ([q for q in a.qmf_feats.split(',') if q] if a.qmf else [])
    kmax = max([k for k in ks] + [IMP_K])
    rows = []
    for name, paths in variants:
        vpath, tpath = paths.split(':')
        vf, VE, vdur = load_npz(vpath)
        tf, TE, _ = load_npz(tpath)
        assert tf == tfiles, 'train embeddings do not match --train_list'
        ia, ib = trial_index(vf, enroll, test)
        n0 = len(rows)
        for kind in kinds:
            for lam in (lams if kind in ('white', 'wccn') else [0.0]):
                for cohort in cohorts:
                    sc = Scorer(VE, TE, tspk, kind, lam, cohort, kmax)
                    for k in ks:
                        s = sc.scores(ia, ib, k)
                        for q in qmfs:
                            if not k and cohort != cohorts[0] and q != 'dur_imp':
                                continue        # no cohort involved: computed once
                            row = dict(variant=name, kind=kind, lam=lam, cohort=cohort, asnorm_k=k, qmf=q)
                            if not q:
                                row.update(EER=eer(s, y), EER_f0=eer(s[folds[0]], y[folds[0]]), EER_f1=eer(s[folds[1]], y[folds[1]]))
                            else:           # cross-fitted: train on one fold, score the other
                                X = sc.qmf_features(s, vdur, ia, ib, k, q)
                                z = np.zeros_like(s)
                                for f in (0, 1):
                                    z[folds[1 - f]] = fit_qmf(X[folds[f]], y[folds[f]])(X[folds[1 - f]])
                                row.update(EER=np.nan, EER_f0=eer(z[folds[0]], y[folds[0]]), EER_f1=eer(z[folds[1]], y[folds[1]]))
                            rows.append(row)
                    del sc
        print('%s: %d configurations' % (name, len(rows) - n0), flush=True)
    for r in rows:      # same trials for every configuration: the within-fold trials of both folds
        r['EER_sel'] = 0.5 * (r['EER_f0'] + r['EER_f1'])
    best = lambda rs, col: min(rs, key=lambda r: r[col])
    pick0, pick1 = best(rows, 'EER_f0'), best(rows, 'EER_f1')
    honest = 0.5 * (pick0['EER_f1'] + pick1['EER_f0'])     # chosen on one fold, measured on the other
    final = best(rows, 'EER_sel')
    base = [r for r in rows if r['variant'] == variants[0][0] and r['kind'] == 'center' and r['asnorm_k'] == 0 and not r['qmf']]
    res = dict(final=final, chosen_on_fold0=pick0, chosen_on_fold1=pick1, honest_cross_fold_EER=honest,
               reference_center=base[0] if base else None, n_configs=len(rows), rows=rows)
    res['conditions'] = conditions(final, variants, a, enroll, test, y)
    json.dump(res, open(a.out, 'w'), indent=1, default=float)
    for c in res['conditions']:
        print('  %-30s targets %6d  non-targets %6d  EER %6.3f%% (chosen, before QMF)  %6.3f%% (reference, center)'
              % (c['condition'], c['n_tar'], c['n_non'], c['EER'], c['EER_ref']))
    print('Final configuration:', {k: final[k] for k in ('variant', 'kind', 'lam', 'cohort', 'asnorm_k', 'qmf', 'EER_sel')})
    print('Honest cross-fold estimate of the selection procedure: EER %.3f%%' % honest)


def conditions(final, variants, a, enroll, test, y):
    """EER of the chosen (before QMF) and of the reference (first variant, center) configuration per
    validation condition, on all validation trials."""
    _, tspk = read_files(a.train_list)
    vmap = dict(variants)

    def scores_for(r):
        vpath, tpath = vmap[r['variant']].split(':')
        vf, VE, vdur = load_npz(vpath); _, TE, _ = load_npz(tpath)
        ia, ib = trial_index(vf, enroll, test)
        return Scorer(VE, TE, tspk, r['kind'], r['lam'], r.get('cohort', 'spk')).scores(ia, ib, r['asnorm_k']), vdur, load_bw(vpath), ia, ib
    s1, dur, bw, ia, ib = scores_for(final)
    s0 = scores_for(dict(variant=variants[0][0], kind='center', lam=0.0, asnorm_k=0))[0]
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
        # bandwidth conditions are defined on target trials; all non-targets kept
        mm = m | (y == 0) if name.startswith(('same', 'cross', 'both')) else m
        if (y[mm] == 1).sum() < 20 or (y[mm] == 0).sum() < 20:
            continue
        out.append(dict(condition=name, n_tar=int((y[mm] == 1).sum()), n_non=int((y[mm] == 0).sum()),
                        EER=eer(s1[mm], y[mm]), EER_ref=eer(s0[mm], y[mm])))
    return out


def cmd_score(a):
    from inference import parse_trials
    sel = json.load(open(a.sel))['final']
    tfiles, _ = read_files(a.train_list)
    tf, _, _ = load_npz(a.train_emb)
    assert tf == tfiles
    enroll, test, labels = parse_trials(a.trials)
    s = score_config(sel, a.test_emb, a.train_emb, a.train_list, enroll, test, a.val_emb, a.val_trials)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        for x, t, v in zip(enroll, test, s):
            f.write('%s %s %.6f\n' % (x, t, v))
    print('Wrote %d scores to %s with %s' % (len(s), a.out, {k: sel.get(k) for k in ('variant', 'kind', 'lam', 'cohort', 'asnorm_k', 'qmf')}))
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
    s.add_argument('--kinds', default='cos,center,white,wccn'); s.add_argument('--lams', default='0,0.05,0.1,0.2,0.3,0.5')
    s.add_argument('--asnorm_k', default='0,50,100,200,300,400,500,700,1000')
    s.add_argument('--cohorts', default='spk,utt', help='spk = speaker means (WeSpeaker), utt = all training utterances')
    s.add_argument('--qmf', type=int, default=1); s.add_argument('--qmf_feats', default='dur,dur_imp')
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
