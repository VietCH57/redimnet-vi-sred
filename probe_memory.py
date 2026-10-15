'''
Find the largest per-GPU training batch that fits in memory (forward + backward, AMP), so that
--batch_size can be chosen before a long run.

    python probe_memory.py --pretrained redimnet:M-vb2+vox2+cnc-ft_mix --crop_sec 3 --n_gpu 2

Prints the largest per-GPU batch that fits with a safety margin and a suggested
--batch_size / --accum_steps pair for a target effective batch.
'''

import argparse, math, torch
import torch.nn.functional as F
import models
from loss import AAMsoftmax


def fits(enc, head, b, n, amp):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        x = torch.randn(b, n, device='cuda') * 0.1
        y = torch.randint(0, head.weight.shape[0], (b,), device='cuda')
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            e = enc(x)
        loss, _ = head(e.float(), y)
        loss.backward()
        enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        return torch.cuda.max_memory_allocated() / 2 ** 30
    except torch.cuda.OutOfMemoryError:
        enc.zero_grad(set_to_none=True); head.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pretrained', default='redimnet:M-vb2+vox2+cnc-ft_mix')
    p.add_argument('--ckpt_dir', default='')
    p.add_argument('--crop_sec', type=float, default=3.0, help='Longest training crop')
    p.add_argument('--amp', type=int, default=1)
    p.add_argument('--n_gpu', type=int, default=0, help='GPUs used by --data_parallel (0 = all visible)')
    p.add_argument('--target_batch', type=int, default=128, help='Desired effective batch')
    p.add_argument('--margin', type=float, default=0.85, help='Use at most this fraction of GPU memory')
    p.add_argument('--candidates', type=str, default='96,80,64,56,48,40,32,24,16,12,8')
    a = p.parse_args()
    assert torch.cuda.is_available(), 'needs a GPU'
    enc, _ = models.load_encoder(a.pretrained, a.ckpt_dir)
    enc = enc.cuda().train()
    head = AAMsoftmax(3000, enc.embed_dim).cuda()
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    n = int(a.crop_sec * 16000)
    best = None
    for b in sorted({int(x) for x in a.candidates.split(',')}):
        gb = fits(enc, head, b, n, bool(a.amp))
        print('per-GPU batch %3d x %.1f s: %s' % (b, a.crop_sec, 'OOM' if gb is None else '%.2f GB peak' % gb), flush=True)
        if gb is None or gb > a.margin * total:
            break
        best = b
    n_gpu = a.n_gpu or torch.cuda.device_count()
    if best is None:
        print('Even the smallest batch does not fit.'); return
    per_step = best * n_gpu
    accum = max(1, math.ceil(a.target_batch / per_step))
    bs = min(per_step, int(math.ceil(a.target_batch / accum / n_gpu)) * n_gpu)
    print('\nGPU %.1f GB | largest per-GPU batch within %.0f%%: %d' % (total, 100 * a.margin, best))
    print('Suggested: --batch_size %d --accum_steps %d  (effective %d on %d GPU)' % (bs, accum, bs * accum, n_gpu))


if __name__ == '__main__':
    main()
