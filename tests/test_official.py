'''
Numerical equivalence with the official training code (PalabraAI/redimnet2 applications/speaker_rec).

    git clone --depth 1 https://github.com/PalabraAI/redimnet2 /tmp/redimnet2
    python tests/test_official.py /tmp/redimnet2

Checks: SphereFace2 loss (margin types A / C), AAM logits (= WeSpeaker ArcMarginProduct), the
ExponentialDecrease LR schedule with warm-up and scale_ratio, the MarginScheduler, LowPassAug.
'''

import importlib.util, math, os, sys, types
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from loss import AAMsoftmax, SphereFace2
from trainer import Schedule
from dataLoader import TrainSet


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def main(root):
    W = os.path.join(root, 'applications/speaker_rec/wespeaker_lite')
    proj = load(os.path.join(W, 'models/projections.py'), 'off_proj')
    sch = load(os.path.join(W, 'utils/schedulers.py'), 'off_sched')
    torch.manual_seed(0)
    x = torch.randn(32, 192); y = torch.randint(0, 50, (32,))

    # ---- SphereFace2
    for mt in ('C', 'A'):
        for m in (0.0, 0.2, 0.3):
            off = proj.SphereFace2(192, 50, scale=32.0, margin=0.0, lanbuda=0.7, t=3, margin_type=mt, dynamic_margin=True)
            off.update(m)
            ours = SphereFace2(50, 192, m=m, s=32.0, lanbuda=0.7, t=3, margin_type=mt)
            ours.weight.data.copy_(off.weight.data); off.bias.data.fill_(0.37); ours.bias.data.fill_(0.37)
            out_o, loss_o = off(x, y)
            loss_u, acc_u = ours(x, y)
            assert torch.allclose(loss_o, loss_u, atol=1e-5, rtol=1e-5), (mt, m, loss_o, loss_u)
            acc_o = (out_o.argmax(1) == y).float().mean() * 100
            assert torch.allclose(acc_o, acc_u), (acc_o, acc_u)
    print('SphereFace2 == official (A, C; m 0 / 0.2 / 0.3)')

    # ---- AAM
    for m in (0.0, 0.2, 0.5):
        off = proj.ArcMarginProduct(192, 50, scale=32.0, margin=0.0, easy_margin=False); off.update(m)
        ours = AAMsoftmax(50, 192, m=m, s=32.0); ours.weight.data.copy_(off.weight.data)
        assert torch.allclose(off(x, y), ours.logits(x, y), atol=1e-4), m
        loss_o = torch.nn.functional.cross_entropy(off(x, y), y)
        assert torch.allclose(loss_o, ours(x, y)[0], atol=1e-5)
    print('AAM == official ArcMarginProduct + CE (m 0 / 0.2 / 0.5)')

    # ---- LR schedule (ExponentialDecrease, warm_from_zero, scale_ratio)
    for warm_from_zero, scale in ((True, 2.0), (False, 3.0), (True, 1.0)):
        epoch_iter, epochs = 37, 5
        a = types.SimpleNamespace(lr=1e-4, final_lr=2.5e-5, warmup_epochs=1, warm_from_zero=warm_from_zero,
                                  scheduler='exp', freeze_epochs=0, max_epoch=epochs, m=0.3, m_start=0.3,
                                  m_increase_start_epoch=1, m_fix_start_epoch=1, m_increase_type='exp')
        mine = Schedule(a, epoch_iter, scale)
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
        off = sch.ExponentialDecrease(opt, epochs, epoch_iter, 1e-4, 2.5e-5, warm_up_epoch=1, scale_ratio=scale,
                                      warm_from_zero=warm_from_zero)
        for it in range(epochs * epoch_iter):
            off.step(it)
            assert abs(opt.param_groups[0]['lr'] - mine.lr(it)) < 1e-12, (it, opt.param_groups[0]['lr'], mine.lr(it))
    print('LR schedule == official ExponentialDecrease (warm-up from 0 / not, scale 1 / 2 / 3)')

    # ---- margin schedule (pre-training ramp 0 -> 0.2 over epochs 20..40)
    class _P:
        margin = 0.0
        def update(self, margin): self.margin = margin
    model = types.SimpleNamespace(projection=_P())
    epoch_iter = 11
    off = sch.MarginScheduler(model, epoch_iter, increase_start_epoch=20, fix_start_epoch=40, initial_margin=0.0,
                              final_margin=0.2, update_margin=True, increase_type='exp')
    a = types.SimpleNamespace(lr=1e-4, final_lr=1e-5, warmup_epochs=0, warm_from_zero=False, scheduler='exp',
                              freeze_epochs=0, max_epoch=50, m=0.2, m_start=0.0, m_increase_start_epoch=20,
                              m_fix_start_epoch=40, m_increase_type='exp')
    mine = Schedule(a, epoch_iter, 1.0)
    for it in range(50 * epoch_iter):
        off.step(it)
        assert abs(model.projection.margin - mine.margin(it)) < 1e-12, (it, model.projection.margin, mine.margin(it))
    print('Margin schedule == official MarginScheduler (0 -> 0.2, exp, epochs 20-40)')

    # ---- LowPassAug
    augs_src = open(os.path.join(W, 'dataset/augs.py')).read()
    augs = types.ModuleType('off_augs'); exec(compile(augs_src, 'augs.py', 'exec'), augs.__dict__)
    sig = np.random.default_rng(0).standard_normal(32240).astype(np.float64)
    off = augs.LowPassAug(cutoff_ratio=0.6, transition_width=0.08, prob=1.0)
    ref = off({'wav': torch.from_numpy(sig).unsqueeze(0).float(), 'sample_rate': 16000})['wav'].numpy()[0]
    ts = TrainSet.__new__(TrainSet); ts.lp_cutoffs, ts.lp_width = [0.6], 0.08
    ours = ts._lowpass(sig.astype(np.float32).astype(np.float64))
    assert np.allclose(ref, ours, atol=1e-5), np.abs(ref - ours).max()
    print('LowPassAug == official')
    print('ALL OFFICIAL-EQUIVALENCE CHECKS PASSED')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '/tmp/redimnet2')
