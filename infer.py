'''
Inference with a (fine-tuned) ReDimNet / ReDimNet2.

Modes
  score : trial list (labelled or not) -> "file1 file2 score" per line, SAME order as the trial list.
          Labelled lists (e.g. public test "a.wav b.wav target") also print EER / minDCF.
          --zip writes results.zip containing results.txt (submission format).
  eval  : labels required, metrics only (+ --eval_meta subsets).
  embed : embeddings of every file of --file_list (or of --trials) -> .npz

--model accepts best.model / model_XXXX.model / last.ckpt of this repo, an authors' checkpoint,
or a release name ("redimnet:M-vb2+vox2+cnc-ft_mix", "redimnet2:b4-vox2-lm") for zero-shot use.

--score_norm center subtracts the mean embedding stored in the model (center_mean, estimated on
training data during train.py) or, if --cohort_list is given, estimated on those files.

Examples
  python infer.py --mode score --model exps/ft/model/best.model \
      --trials /kaggle/input/.../public-test.txt --data_path /kaggle/input/.../data \
      --out /kaggle/working/submission/results.txt --zip
  python infer.py --mode eval --model redimnet:M-vb2+vox2+cnc-ft_mix --trials val_trials.txt --data_path .../data
'''

import argparse, json, os, random, time, zipfile
import numpy, torch
import models
from dataLoader import read_list
from inference import (parse_trials, extract_embeddings, score_trials, mean_embedding, subset_metrics,
                       save_emb, load_emb)
from tools import compute_metrics


def build_parser():
    p = argparse.ArgumentParser(description='ReDimNet inference', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode', choices=['score', 'eval', 'embed'], default='score')
    p.add_argument('--model', required=True, help='best.model / model_XXXX.model / last.ckpt / release name')
    p.add_argument('--ckpt_dir', type=str, default='', help='Download cache for release names')
    p.add_argument('--trials', type=str, default='', help='Trial list ("label a b", "a b target|nontarget" or "a b")')
    p.add_argument('--file_list', type=str, default='', help='[embed] list of files (last column of each line)')
    p.add_argument('--data_path', type=str, required=True, help='Root folder of the paths in the list')
    p.add_argument('--out', type=str, default='results.txt', help='[score] score file / [embed] .npz')
    p.add_argument('--zip', action='store_true', help='[score] also write <out_dir>/results.zip with results.txt inside')
    p.add_argument('--score_fmt', type=str, default='%.6f')
    p.add_argument('--score_norm', type=str, default='cos', choices=['cos', 'center'])
    p.add_argument('--cohort_list', type=str, default='', help='[center] "<spk> <path>" list to estimate the mean (default: mean stored in the model)')
    p.add_argument('--cohort_path', type=str, default='', help='[center] root of --cohort_list')
    p.add_argument('--cohort_size', type=int, default=1000, help='[center] files drawn from --cohort_list')
    p.add_argument('--eval_mode', type=str, default='full', choices=['full', 'crops', 'both'])
    p.add_argument('--eval_sec', type=float, default=3.0)
    p.add_argument('--num_eval', type=int, default=5)
    p.add_argument('--eval_min_sec', type=float, default=1.0)
    p.add_argument('--eval_max_sec', type=float, default=60.0)
    p.add_argument('--eval_len_tol', type=float, default=0.0, help='0 = exact full-utterance embeddings (recommended for submissions)')
    p.add_argument('--eval_batch', type=int, default=64)
    p.add_argument('--amp', action='store_true', help='fp16 inference (faster, tiny numerical differences)')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--dcf_ptarget', type=str, default='0.05,0.01')
    p.add_argument('--eval_meta', type=str, default='')
    p.add_argument('--emb_cache', type=str, default='', help='.npz to load embeddings from if it exists, otherwise to save them to')
    p.add_argument('--device', type=str, default='cuda')
    return p


def main():
    args = build_parser().parse_args()
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA not available -> using CPU'); args.device = 'cpu'
    p_targets = [float(x) for x in args.dcf_ptarget.split(',') if x]
    for path in (args.out, args.emb_cache):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if args.mode in ('score', 'eval'):
        assert args.trials, '--trials is required for mode %s' % args.mode
        enroll, test, labels = parse_trials(args.trials)
        files = sorted(set(enroll) | set(test))
        print('%d trials, %d unique files, labels: %s' % (len(enroll), len(files), labels[0] is not None))
        if args.mode == 'eval' and labels[0] is None:
            raise SystemExit('mode eval needs a labelled trial list')
    else:
        src = args.file_list or args.trials
        assert src, '--file_list (or --trials) is required for mode embed'
        if args.file_list:
            files = sorted({l.split()[-1] for l in open(args.file_list).read().splitlines() if l.strip()})
        else:
            e, t, _ = parse_trials(args.trials); files = sorted(set(e) | set(t))
    missing = [f for f in files if not os.path.isfile(os.path.join(args.data_path, f))]
    if missing:
        raise SystemExit('%d files not found under %s, e.g. %s' % (len(missing), args.data_path, missing[:3]))

    kw = dict(device=args.device, mode=args.eval_mode, eval_sec=args.eval_sec, num_eval=args.num_eval,
              min_sec=args.eval_min_sec, max_sec=args.eval_max_sec, len_tol=args.eval_len_tol,
              max_batch=args.eval_batch, amp=args.amp, workers=args.workers)
    encoder, extra = None, {}

    def get_encoder():
        nonlocal encoder, extra
        if encoder is None:
            encoder, extra = models.load_encoder(args.model, args.ckpt_dir)
            encoder = encoder.to(args.device).eval()
        return encoder

    emb = None
    if args.emb_cache and os.path.exists(args.emb_cache):
        emb = load_emb(args.emb_cache)
        if not set(files) <= set(emb):
            print('Cache %s does not cover all files -> recomputing' % args.emb_cache); emb = None
        else:
            print('Embeddings loaded from', args.emb_cache)
    if emb is None:
        t0 = time.time()
        emb = extract_embeddings(get_encoder(), files, args.data_path, **kw)
        print('Embedded %d files in %.1f min' % (len(files), (time.time() - t0) / 60))
        if args.emb_cache:
            save_emb(args.emb_cache, emb)

    if args.mode == 'embed':
        save_emb(args.out, emb); print('Saved', args.out); return

    center = None
    if args.score_norm == 'center':
        if args.cohort_list:
            _, paths = read_list(args.cohort_list)
            paths = sorted(random.Random(1234).sample(paths, min(args.cohort_size, len(paths))))
            center = mean_embedding(extract_embeddings(get_encoder(), paths, args.cohort_path or args.data_path, **kw))
            print('Center estimated on %d files of %s' % (len(paths), args.cohort_list))
        else:
            if encoder is None:
                _, extra = models.load_encoder(args.model, args.ckpt_dir, verbose=False)
            cm = extra.get('center_mean') or (extra.get('center') and {'full': extra['center'][0], 'crops': extra['center'][1]})
            if not cm:
                raise SystemExit('--score_norm center: the model has no center_mean; pass --cohort_list/--cohort_path')
            center = (cm.get('full'), cm.get('crops'))
            print('Center: mean stored in the model')

    scores = score_trials(emb, enroll, test, args.eval_mode, center)
    if args.mode == 'score':
        with open(args.out, 'w') as f:
            for a, b, sc in zip(enroll, test, scores):
                f.write('%s %s %s\n' % (a, b, args.score_fmt % sc))
        lines = open(args.out).read().splitlines()
        assert len(lines) == len(enroll) and all(l.split()[:2] == [a, b] for l, a, b in zip(lines, enroll, test))
        print('Wrote %d lines to %s' % (len(lines), args.out))
        if args.zip:
            zpath = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'results.zip')
            with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
                z.write(args.out, arcname='results.txt')
            print('Wrote', zpath, '(contains results.txt)')
    if labels[0] is not None:
        res = compute_metrics(scores, labels, p_targets)
        if args.eval_meta:
            res.update(subset_metrics(scores, labels, enroll, test, args.eval_meta, p_targets[0]))
        print(' | '.join('%s %.4f' % (k, v) for k, v in res.items()))
        mpath = os.path.splitext(os.path.abspath(args.out))[0] + '_metrics.json'
        json.dump(res, open(mpath, 'w'), indent=2); print('Metrics saved to', mpath)


if __name__ == '__main__':
    main()
