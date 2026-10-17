'''Regression test: the BatchNorm sync before evaluation must only broadcast contiguous tensors
(NCCL refuses non-contiguous ones; ReDimNet2 has non-contiguous STFT / mel buffers).
    python tests/test_ddp_sync.py'''
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29531')
import torch.distributed as dist
dist.init_process_group('gloo', rank=0, world_size=1)
orig, n = dist.broadcast, [0]
def strict(t, src=0, **k):
    assert t.is_contiguous(), 'non-contiguous tensor broadcast (NCCL would fail)'
    n[0] += 1
    return orig(t, src=src, **k)
dist.broadcast = strict
import trainer
from config import parse_args
for spec, loss in [('redimnet2:b4-vox2-lm', 'sf2'), ('redimnet:M-vb2+vox2+cnc-ft_mix', 'aam')]:
    t = trainer.Trainer(parse_args(['--eval', '--pretrained', spec, '--loss', loss, '--device', 'cpu']), n_class=3)
    t.ddp = True; n[0] = 0; t.sync_buffers()
    assert n[0] > 0
    print('%s: %d BatchNorm buffers broadcast, all contiguous' % (spec, n[0]))
dist.destroy_process_group()
print('DDP SYNC TEST OK')
