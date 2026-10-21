'''
Command-line / config-file arguments for train.py.
Priority: command line  >  --config file (YAML or JSON)  >  defaults below.

The defaults follow the official large-margin fine-tuning (LM) recipe of ReDimNet2
(PalabraAI/redimnet2, applications/speaker_rec/examples/vox2/recipes/redimnet2/b4/lm.yaml), which is the
WeSpeaker LM recipe (wespeaker examples/voxceleb/v2/conf/resnet_lm.yaml) with the ReDimNet2 loss.
The recipes in configs/ set every value explicitly and give the source of each one (see configs/README.md).
'''

import argparse, json


def _str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('1', 'true', 'yes', 'y', 't'):
        return True
    if v.lower() in ('0', 'false', 'no', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected, got %r' % v)


def _floats(v):
    if isinstance(v, (list, tuple)):
        return [float(x) for x in v]
    return [float(x) for x in str(v).replace(' ', '').replace('[', '').replace(']', '').split(',') if x != '']


def _ints(v):
    return [int(round(x)) for x in _floats(v)]


def _specaug(v):
    v = str(v).lower()
    if v in ('checkpoint', 'ckpt', 'keep'):
        return 'checkpoint'
    return 'true' if _str2bool(v) else 'false'


def build_parser():
    p = argparse.ArgumentParser(description='Fine-tune ReDimNet / ReDimNet2 for speaker verification',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, default='', help='YAML/JSON file with any of the options below (command line overrides it)')

    g = p.add_argument_group('Data')
    g.add_argument('--train_list', type=str, default='', help='Training list, each line "<speaker> <relative/path.wav>"')
    g.add_argument('--train_path', type=str, default='', help='Root folder of the training wavs')
    g.add_argument('--crop_sec', type=_floats, default=[6.015], help='Training crop length(s) in s; one is drawn per batch. '
                   'Official: (num_frms - 1) * 10 ms + 25 ms = 2.015 s (200 frames, pre-training) / 6.015 s (600 frames, LM)')
    g.add_argument('--crop_probs', type=_floats, default=[], help='Probabilities of --crop_sec (empty = uniform)')
    g.add_argument('--min_dur', type=float, default=1.0, help='Skip training files shorter than this (s). Official filter: min_num_frames 100 = 1.0 s')
    g.add_argument('--batch_size', type=int, default=128, help='Global batch per forward pass (split over the GPUs with torchrun)')
    g.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation: effective batch = batch_size * accum_steps')
    g.add_argument('--n_cpu', type=int, default=8, help='DataLoader workers in total (with torchrun they are shared by the GPU processes)')
    g.add_argument('--seed', type=int, default=42, help='Random seed (-1 = do not seed)')

    g = p.add_argument_group('Augmentation (waveform, official wespeaker_lite chains, see dataLoader.py)')
    g.add_argument('--musan_path', type=str, default='', help='MUSAN root (folder with music/ noise/ speech/, or a parent)')
    g.add_argument('--rir_path', type=str, default='', help='RIR root (simulated_rirs is used if found below it)')
    g.add_argument('--aug_chain', type=str, default='wespeaker_lite', choices=['wespeaker', 'wespeaker_lite', 'hard', 'none'],
                   help='wespeaker = upstream WeSpeaker add_reverb_noise (ReDimNet); wespeaker_lite = add_reverb_noise_gain '
                        '(ReDimNet2 VoxCeleb2 recipes); hard = Sequential tree of the ReDimNet2 vb2+vox2+cnc2 recipes')
    g.add_argument('--aug_prob', type=float, default=0.6, help='[wespeaker, wespeaker_lite] probability of augmenting a sample')
    g.add_argument('--aug_preset', type=str, default='default', choices=['default', 'medium', 'hard'], help='[wespeaker_lite] SNR preset')
    g.add_argument('--gain_prob', type=float, default=1.0, help='[wespeaker_lite] probability of the random gain')
    g.add_argument('--gain_db', type=_floats, default=[-18.0, 6.0], help='[wespeaker_lite] gain range (dB)')
    g.add_argument('--hard_prob', type=float, default=0.9, help='[hard] probability of the whole tree (ptn_ha 0.9, LM 0.7)')
    g.add_argument('--hard_noise_prob', type=float, default=0.8, help='[hard] probability of the MUSAN OneOf')
    g.add_argument('--hard_reverb_prob', type=float, default=0.2, help='[hard] probability of reverberation')
    g.add_argument('--lowpass_prob', type=float, default=0.25, help='[hard] probability of the low-pass filter')
    g.add_argument('--lowpass_cutoffs', type=_floats, default=[0.6], help='[hard] cutoff(s) as a fraction of Nyquist (0.6 used, 0.5 = 4 kHz in the official comments)')
    g.add_argument('--lowpass_width', type=float, default=0.08, help='[hard] transition width (fraction of Nyquist)')
    g.add_argument('--speed_perturb', type=_floats, default=[], help='Extra speeds (official pre-training: 0.9,1.1); each one adds a copy of every speaker as new classes. Empty = off (official LM)')
    g.add_argument('--speed_probs', type=_floats, default=[], help='Probabilities of [1.0, *speed_perturb] (empty = uniform, as official)')

    g = p.add_argument_group('SpecAugment (inside the model front-end, training only, no parameters)')
    g.add_argument('--specaug', type=_specaug, default='checkpoint', help='checkpoint = keep the do_spec_aug of the pretrained model_config; true / false = force')
    g.add_argument('--specaug_freq', type=_ints, default=[], help='Frequency mask width [min,max); empty = keep the checkpoint value')
    g.add_argument('--specaug_time', type=_ints, default=[], help='Time mask width [min,max); empty = keep the checkpoint value')

    g = p.add_argument_group('Model & loss')
    g.add_argument('--pretrained', type=str, default='redimnet2:b4-vox2-lm',
                   help='"redimnet:<release name>", "redimnet2:<release name>" or a checkpoint path')
    g.add_argument('--ckpt_dir', type=str, default='', help='Download cache for release checkpoints (default ~/.cache/redimnet-vi-sred)')
    g.add_argument('--max_params', type=float, default=7e6, help='Refuse an encoder with more parameters (0 = no check)')
    g.add_argument('--loss', type=str, default='sf2', choices=['aam', 'sf2'], help='aam = WeSpeaker ArcMarginProduct + CE; sf2 = SphereFace2 (ReDimNet2)')
    g.add_argument('--m', type=float, default=0.3, help='Final margin (SF2-C: 0.2 pre-training / 0.3 LM; AAM: 0.2 / 0.5)')
    g.add_argument('--s', type=float, default=32.0, help='Scale (32 in every official recipe)')
    g.add_argument('--sf2_lanbuda', type=float, default=0.7, help='[sf2] weight of the positive term')
    g.add_argument('--sf2_t', type=int, default=3, help='[sf2] t of the similarity adjustment g(z)')
    g.add_argument('--sf2_margin_type', type=str, default='C', choices=['A', 'C'])
    g.add_argument('--m_start', type=float, default=-1.0, help='Initial margin of the official MarginScheduler (-1 = same as --m, i.e. constant)')
    g.add_argument('--m_increase_start_epoch', type=float, default=1, help='MarginScheduler increase_start_epoch (1-based, as official)')
    g.add_argument('--m_fix_start_epoch', type=float, default=1, help='MarginScheduler fix_start_epoch (margin = --m from here on)')
    g.add_argument('--m_increase_type', type=str, default='exp', choices=['exp', 'linear'])
    g.add_argument('--head_init', type=str, default='centroid', choices=['centroid', 'random'],
                   help='Head rows of speakers without trained rows: centroid = mean embedding of the pretrained encoder (weight imprinting); random = xavier (official fallback)')
    g.add_argument('--head_init_utts', type=int, default=4, help='[centroid] utterances per speaker')
    g.add_argument('--head_init_sec', type=float, default=3.0, help='[centroid] centre crop length (s)')

    g = p.add_argument_group('Optimisation (official: SGD Nesterov + ExponentialDecrease, LR x scale_ratio)')
    g.add_argument('--max_epoch', type=int, default=5, help='Number of epochs (official LM: 5)')
    g.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adamw', 'adam'])
    g.add_argument('--momentum', type=float, default=0.9, help='[sgd]')
    g.add_argument('--nesterov', type=_str2bool, default=True, help='[sgd]')
    g.add_argument('--lr', type=float, default=1e-4, help='initial_lr of the scheduler (before x scale_ratio)')
    g.add_argument('--final_lr', type=float, default=2.5e-5, help='final_lr of the scheduler (before x scale_ratio)')
    g.add_argument('--lr_scale_ratio', type=float, default=-1, help='LR multiplier; -1 = official rule: effective global batch / 64 (world_size * batch_size / 64)')
    g.add_argument('--scheduler', type=str, default='exp', choices=['exp', 'cosine', 'const'], help='exp = official ExponentialDecrease (per iteration)')
    g.add_argument('--warmup_epochs', type=float, default=1.0, help='warm_up_epoch of the scheduler')
    g.add_argument('--warm_from_zero', type=_str2bool, default=True, help='Warm-up from LR 0 (official LM: True)')
    g.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay of the encoder (official LM 1e-4, pre-training 2e-5)')
    g.add_argument('--head_wd_mult', type=float, default=1.0, help='Head weight decay = weight_decay * this (official loss_weight_decay_mult)')
    g.add_argument('--wd_skip_1d', type=_str2bool, default=False, help='No weight decay on biases / norm weights (official: False = decay everything)')
    g.add_argument('--lr_head_mult', type=float, default=1.0, help='Head LR = encoder LR * this (official: 1)')
    g.add_argument('--layer_decay', type=float, default=1.0, help='LR of encoder block k = lr * layer_decay^(depth-k) (1 = official)')
    g.add_argument('--freeze_epochs', type=int, default=0, help='Epochs training only the head (0 = official)')
    g.add_argument('--l2sp', type=float, default=0.0, help='L2-SP (Li et al., ICML 2018): penalty l2sp/2 * ||w - w_anchor||^2 on the encoder, '
                   'which then gets no weight decay (as in the paper, the penalty replaces it for the pre-trained layers; '
                   'the head keeps its weight decay). 0 = official')
    g.add_argument('--l2sp_anchor', type=str, default='', help='[l2sp] encoder whose weights are the anchor (e.g. the pretrained release '
                   '"redimnet:M-vb2+vox2+cnc-ft_mix"); empty = the weights at the start of this run')
    g.add_argument('--bn_mode', type=str, default='train', choices=['train', 'frozen'], help='train = official; frozen = keep pretrained BN running statistics')
    g.add_argument('--grad_clip', type=float, default=0.0, help='Clip grad-norm (0 = off, official)')
    g.add_argument('--amp', type=_str2bool, default=True, help='Mixed precision (fp16) for the encoder; the head always runs in fp32 (official enable_amp: true)')

    g = p.add_argument_group('Evaluation during training')
    g.add_argument('--eval_list', type=str, default='', help='Trial list ("label a b" or "a b target|nontarget"). Empty = no evaluation')
    g.add_argument('--eval_path', type=str, default='', help='Root folder of the evaluation wavs')
    g.add_argument('--eval_meta', type=str, default='', help='Optional val_trials_meta.csv -> subset EERs')
    g.add_argument('--test_step', type=int, default=1, help='Evaluate and save every N epochs')
    g.add_argument('--eval_init', type=_str2bool, default=True, help='Evaluate the starting model before training (logged as epoch 0)')
    g.add_argument('--eval_dur', type=float, default=8.0, help='Validation during training on the first N seconds of every file, '
                   'short files wrap-padded (official eval_dur 8.0, Vox1TestEvalDatasetSimple). 0 = whole utterances')
    g.add_argument('--eval_min_sec', type=float, default=1.0, help='Shorter files are repeat-padded to this length')
    g.add_argument('--eval_max_sec', type=float, default=60.0, help='Longer files are cut (0 = never)')
    g.add_argument('--eval_len_tol', type=float, default=0.02, help='[eval_dur 0] batch files whose lengths differ by <= this fraction')
    g.add_argument('--eval_batch', type=int, default=64, help='Max files per evaluation batch')
    g.add_argument('--eval_amp', type=_str2bool, default=True, help='fp16 validation on CUDA (official: autocast enabled=enable_amp)')
    g.add_argument('--cohort_size', type=int, default=1000, help='Training files used to estimate the mean embedding for "center" scoring (0 = off)')
    g.add_argument('--select_score', type=str, default='cos', choices=['cos', 'center'], help='Which EER selects best.model')
    g.add_argument('--dcf_ptarget', type=_floats, default=[0.05], help='P_target for minDCF (reported only)')

    g = p.add_argument_group('Checkpoints & logging')
    g.add_argument('--save_path', type=str, default='exps/exp1', help='Output folder (score.txt, model/, config.json)')
    g.add_argument('--initial_model', type=str, default='', help='Start from this checkpoint instead of --pretrained (epoch restarts at 1). '
                   'A last.ckpt also gives the head: rows of the same speakers are copied (official reassemble_projection_head)')
    g.add_argument('--resume', type=str, default='auto', help='"auto" = continue from save_path/model/last.ckpt if present, "none", or a path to a last.ckpt')
    g.add_argument('--max_hours', type=float, default=0, help='Stop before this many hours are used (0 = no limit); rerun to resume')
    g.add_argument('--keep_last', type=int, default=0, help='Keep only the N newest model_XXXX.model (0 = all)')
    g.add_argument('--save_best', type=_str2bool, default=True, help='Save model/best.model (lowest EER)')
    g.add_argument('--log_interval', type=int, default=20, help='Print progress every N batches')
    g.add_argument('--device', type=str, default='cuda', help='cuda / cuda:0 / cpu')

    g = p.add_argument_group('Command')
    g.add_argument('--eval', dest='eval', action='store_true', help='Only evaluate --initial_model (or --pretrained) on --eval_list and exit')
    return p


def load_config_file(path):
    with open(path) as f:
        if path.endswith(('.yaml', '.yml')):
            import yaml
            return yaml.safe_load(f) or {}
        return json.load(f)


def parse_args(argv=None):
    parser = build_parser()
    pre, _ = parser.parse_known_args(argv)
    if pre.config:
        cfg = load_config_file(pre.config)
        known = {a.dest for a in parser._actions}
        unknown = [k for k in cfg if k not in known]
        if unknown:
            raise ValueError('Unknown keys in %s: %s' % (pre.config, unknown))
        for a in parser._actions:
            if a.dest in cfg and a.type is not None and cfg[a.dest] is not None:
                cfg[a.dest] = a.type(cfg[a.dest])
        parser.set_defaults(**cfg)
    args = parser.parse_args(argv)
    if args.crop_probs and len(args.crop_probs) != len(args.crop_sec):
        parser.error('--crop_probs needs one value per --crop_sec')
    if len(args.gain_db) != 2:
        parser.error('--gain_db needs 2 values')
    if not args.eval and not (args.train_list and args.train_path):
        parser.error('--train_list and --train_path are required for training')
    if args.m_start < 0:
        args.m_start = args.m
    return args


def save_config(args, path):
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
