'''
Command-line / config-file arguments for train.py.
Priority: command line  >  --config file (YAML or JSON)  >  defaults below.
Defaults = the fine-tuning recipe for ReDimNet-M on Vi-SRED26 (see README).
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
    return [float(x) for x in str(v).replace(' ', '').split(',') if x != '']


def _ints(v):
    return [int(round(x)) for x in _floats(v)]


def _strs(v):
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [x for x in str(v).replace(' ', '').split(',') if x != '']


def build_parser():
    p = argparse.ArgumentParser(description='Fine-tune ReDimNet / ReDimNet2 for speaker verification',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--config', type=str, default='', help='YAML/JSON file with any of the options below (command line overrides it)')

    g = p.add_argument_group('Data')
    g.add_argument('--train_list', type=str, default='', help='Training list, each line "<speaker> <relative/path.wav>"')
    g.add_argument('--train_path', type=str, default='', help='Root folder of the training wavs')
    g.add_argument('--crop_sec', type=_floats, default=[2.0, 3.0], help='Training crop lengths (s); one is drawn per batch')
    g.add_argument('--crop_probs', type=_floats, default=[], help='Probabilities of --crop_sec (empty = uniform)')
    g.add_argument('--batch_size', type=int, default=64, help='Batch size (split over GPUs with --data_parallel)')
    g.add_argument('--n_cpu', type=int, default=8, help='DataLoader workers')
    g.add_argument('--seed', type=int, default=42, help='Random seed (-1 = do not seed)')

    g = p.add_argument_group('Augmentation (waveform)')
    g.add_argument('--musan_path', type=str, default='', help='MUSAN root (folder with music/ noise/ speech/, or a parent). Empty = off')
    g.add_argument('--rir_path', type=str, default='', help='RIR root (simulated_rirs is used if found below it). Empty = off')
    g.add_argument('--aug_probs', type=_floats, default=[1, 1, 1, 1, 1, 1], help='Relative weights of [clean, reverb, babble, music, noise, tv]')
    g.add_argument('--snr_noise', type=_floats, default=[0, 15], help='SNR range (dB) for MUSAN noise')
    g.add_argument('--snr_speech', type=_floats, default=[13, 20], help='SNR range (dB) for babble')
    g.add_argument('--snr_music', type=_floats, default=[5, 15], help='SNR range (dB) for music')
    g.add_argument('--num_noise', type=_ints, default=[1, 1], help='Number of noise files mixed (min,max)')
    g.add_argument('--num_speech', type=_ints, default=[3, 8], help='Number of babble speakers (min,max)')
    g.add_argument('--num_music', type=_ints, default=[1, 1], help='Number of music files mixed (min,max)')
    g.add_argument('--speed_perturb', type=_floats, default=[0.9, 1.1], help='Extra speeds; each one adds a copy of every speaker as new classes. Empty = off')
    g.add_argument('--speed_probs', type=_floats, default=[], help='Probabilities of [1.0, *speed_perturb] (empty = uniform)')
    g.add_argument('--narrowband_prob', type=float, default=0.2, help='Probability of a narrow-band channel (after noise/reverb)')
    g.add_argument('--narrowband_types', type=_strs, default=['8k', 'tel'], help='8k = resample to 8 kHz and back, tel = 300-3400 Hz band')

    g = p.add_argument_group('SpecAugment (inside the model front-end, training only, no parameters)')
    g.add_argument('--specaug', type=_str2bool, default=True, help='Enable SpecAugment')
    g.add_argument('--specaug_freq', type=_ints, default=[], help='Frequency mask width [min,max); empty = keep the checkpoint value')
    g.add_argument('--specaug_time', type=_ints, default=[], help='Time mask width [min,max); empty = keep the checkpoint value')

    g = p.add_argument_group('Model & loss')
    g.add_argument('--pretrained', type=str, default='redimnet:M-vb2+vox2+cnc-ft_mix',
                   help='"redimnet:<release name>", "redimnet2:<release name>" or a checkpoint path')
    g.add_argument('--ckpt_dir', type=str, default='', help='Download cache for release checkpoints (default ~/.cache/redimnet-vi-sred)')
    g.add_argument('--max_params', type=float, default=7e6, help='Refuse an encoder with more parameters (0 = no check)')
    g.add_argument('--m', type=float, default=0.2, help='AAM-softmax margin')
    g.add_argument('--m_start', type=float, default=0.0, help='Margin at the start of training (ramped linearly to --m)')
    g.add_argument('--m_warmup_epochs', type=float, default=0, help='Epochs to ramp the margin from --m_start to --m (0 = constant --m)')
    g.add_argument('--s', type=float, default=30, help='AAM-softmax scale')
    g.add_argument('--n_class', type=int, default=0, help='Ignored: always computed from --train_list (x number of speeds)')
    g.add_argument('--head_init', type=str, default='centroid', choices=['centroid', 'random'],
                   help='centroid = class weights = speaker centroids of the pretrained encoder')
    g.add_argument('--head_init_utts', type=int, default=4, help='[centroid] utterances per speaker')
    g.add_argument('--head_init_sec', type=float, default=3.0, help='[centroid] crop length (s)')

    g = p.add_argument_group('Optimisation')
    g.add_argument('--max_epoch', type=int, default=20, help='Number of epochs')
    g.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'adam', 'sgd'])
    g.add_argument('--lr_backbone', type=float, default=1e-4, help='Peak LR of the encoder (its top layers with --layer_decay)')
    g.add_argument('--lr_head', type=float, default=1e-3, help='Peak LR of the AAM head')
    g.add_argument('--layer_decay', type=float, default=1.0, help='LR of encoder block k = lr_backbone * layer_decay^(depth-k) (1 = same LR everywhere)')
    g.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'exp', 'const'])
    g.add_argument('--lr_min_ratio', type=float, default=0.01, help='[cosine] final LR = peak * ratio')
    g.add_argument('--lr_decay', type=float, default=0.95, help='[exp] LR multiplied by this every epoch')
    g.add_argument('--warmup_epochs', type=float, default=1.0, help='Linear LR warm-up (epochs, after --freeze_epochs)')
    g.add_argument('--freeze_epochs', type=int, default=1, help='Epochs training only the head (encoder frozen, BN statistics fixed)')
    g.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (not applied to biases / norm layers)')
    g.add_argument('--l2sp', type=float, default=1e-4, help='L2-SP: penalty l2sp/2 * ||w - w_start||^2 on the encoder (0 = off)')
    g.add_argument('--accum_steps', type=int, default=1, help='Gradient accumulation: one optimiser step every N batches (effective batch = batch_size * N)')
    g.add_argument('--bn_mode', type=str, default='frozen', choices=['train', 'frozen'], help='frozen = keep the pretrained BatchNorm running statistics (affine weights still train); train = update them from the batches')
    g.add_argument('--grad_clip', type=float, default=5.0, help='Clip grad-norm (0 = off)')
    g.add_argument('--amp', type=_str2bool, default=True, help='Mixed precision (fp16) training on CUDA')

    g = p.add_argument_group('Evaluation during training')
    g.add_argument('--eval_list', type=str, default='', help='Trial list ("label a b" or "a b target|nontarget"). Empty = no evaluation')
    g.add_argument('--eval_path', type=str, default='', help='Root folder of the evaluation wavs')
    g.add_argument('--eval_meta', type=str, default='', help='Optional val_trials_meta.csv -> subset EERs')
    g.add_argument('--test_step', type=int, default=1, help='Evaluate and save every N epochs')
    g.add_argument('--eval_init', type=_str2bool, default=True, help='Evaluate the starting model before training (logged as epoch 0)')
    g.add_argument('--eval_mode', type=str, default='full', choices=['full', 'crops', 'both'])
    g.add_argument('--eval_sec', type=float, default=3.0, help='[crops] crop length (s)')
    g.add_argument('--num_eval', type=int, default=5, help='[crops] crops per utterance')
    g.add_argument('--eval_min_sec', type=float, default=1.0, help='Shorter files are repeat-padded to this length')
    g.add_argument('--eval_max_sec', type=float, default=60.0, help='Longer files are cut (0 = never)')
    g.add_argument('--eval_len_tol', type=float, default=0.05, help='Batch files whose lengths differ by <= this fraction (cut to the shortest); 0 = exact')
    g.add_argument('--eval_batch', type=int, default=64, help='Max files per evaluation batch')
    g.add_argument('--eval_amp', type=_str2bool, default=True, help='fp16 evaluation on CUDA')
    g.add_argument('--cohort_size', type=int, default=1000, help='Training files used to estimate the mean embedding for "center" scoring (0 = off)')
    g.add_argument('--select_score', type=str, default='cos', choices=['cos', 'center'], help='Which EER selects best.model')
    g.add_argument('--dcf_ptarget', type=_floats, default=[0.05], help='P_target for minDCF (reported only)')

    g = p.add_argument_group('Checkpoints & logging')
    g.add_argument('--save_path', type=str, default='exps/exp1', help='Output folder (score.txt, model/, config.json)')
    g.add_argument('--initial_model', type=str, default='', help='Start from this checkpoint instead of --pretrained (epoch restarts at 1); a last.ckpt also restores the head if the classes match')
    g.add_argument('--resume', type=str, default='auto', help='"auto" = continue from save_path/model/last.ckpt if present, "none", or a path to a last.ckpt')
    g.add_argument('--max_hours', type=float, default=0, help='Stop before this many hours are used (0 = no limit); rerun to resume')
    g.add_argument('--keep_last', type=int, default=0, help='Keep only the N newest model_XXXX.model (0 = all)')
    g.add_argument('--save_best', type=_str2bool, default=True, help='Save model/best.model (lowest EER)')
    g.add_argument('--log_interval', type=int, default=20, help='Print progress every N batches')
    g.add_argument('--device', type=str, default='cuda', help='cuda / cuda:0 / cpu')
    g.add_argument('--data_parallel', type=_str2bool, default=False, help='Split every training batch over all visible GPUs (nn.DataParallel); evaluation uses one GPU')

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
    if len(args.aug_probs) != 6:
        parser.error('--aug_probs needs 6 values: clean,reverb,babble,music,noise,tv')
    if args.crop_probs and len(args.crop_probs) != len(args.crop_sec):
        parser.error('--crop_probs needs one value per --crop_sec')
    if not args.eval and not (args.train_list and args.train_path):
        parser.error('--train_list and --train_path are required for training')
    return args


def save_config(args, path):
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
