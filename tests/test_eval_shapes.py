'''Validation during training must feed the network a single input length (official fixed eval_dur),
whatever the file lengths: python tests/test_eval_shapes.py <fake data root from make_fake_data.py>'''
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import trainer
from config import parse_args
R = sys.argv[1]
K = R + '/kaggle/input/datasets'; L = K + '/hongvitchugo/vi-sred-cleaned-labels/labels'
D = K + '/hongvitchugo/vi-sred26-train/Vi-SRED26-Train/data'
for spec, loss in [('redimnet2:b4-vox2-lm', 'sf2'), ('redimnet:M-vb2+vox2+cnc-ft_mix', 'aam')]:
    a = parse_args(['--eval', '--pretrained', spec, '--loss', loss, '--device', 'cpu', '--eval_dur', '2.0',
                    '--eval_batch', '5', '--n_cpu', '1'])
    t = trainer.Trainer(a, n_class=3)
    shapes = set()
    t.encoder.register_forward_pre_hook(lambda m, inp: shapes.add(tuple(inp[0].shape[1:])))
    res = t.eval_network(L + '/val_trials.txt', D, cohort=(sorted(os.listdir(D))[:0], D))
    assert shapes == {(32000,)}, shapes
    print('%s: EER %.2f, input lengths seen: %s' % (spec, res['EER'], sorted(shapes)))
print('EVAL SHAPE TEST OK')
