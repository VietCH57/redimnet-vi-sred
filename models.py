'''
ReDimNet / ReDimNet2 encoders: build, load pretrained checkpoints, save in the original format.

The network code lives unmodified in third_party/ (IDRnD/redimnet and PalabraAI/redimnet2).
This file only wraps it:
  * `--pretrained` accepts  "redimnet:<name>" / "redimnet2:<name>"  (official release checkpoints,
    downloaded once) or a path to any checkpoint saved by the authors or by this repo.
  * Checkpoints written by this repo (model_XXXX.model, best.model) keep the authors' format
    {"model_config", "state_dict"} (+ a few extra keys), so they load with the original code /
    torch.hub as well.
'''

import copy, os, sys, torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in ('third_party/redimnet', 'third_party/redimnet2'):
    if os.path.join(_HERE, _p) not in sys.path:
        sys.path.insert(0, os.path.join(_HERE, _p))

URLS = {
    'redimnet':  'https://github.com/IDRnD/redimnet/releases/download/latest/{name}.pt',
    'redimnet2': 'https://github.com/PalabraAI/redimnet2/releases/download/v1.0.0/{name}.pt',
}
# The checkpoints allowed in Vi-SRED26 (any other release name also works)
KNOWN = {
    'redimnet:S-vb2+vox2+cnc-ft_mix': 3.15e6,
    'redimnet:M-vb2+vox2+cnc-ft_mix': 4.81e6,
    'redimnet2:b4-vox2-lm': 6.66e6,
    'redimnet2:b6-vb2+vox2+cnc2_v0-lm': 12.46e6,
}


def wrap_class(family):
    if family == 'redimnet':
        from redimnet.model import ReDimNetWrap
        return ReDimNetWrap
    if family == 'redimnet2':
        from redimnet2.redimnet2 import ReDimNet2Wrap
        return ReDimNet2Wrap
    raise ValueError('unknown family %r (redimnet / redimnet2)' % family)


def default_cache_dir():
    return os.environ.get('REDIMNET_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'redimnet-vi-sred'))


def resolve(spec, cache_dir=''):
    '''"redimnet:M-vb2+vox2+cnc-ft_mix" -> (local path, family); a path -> (path, None).'''
    if os.path.isfile(spec):
        return spec, None
    if ':' not in spec:
        raise ValueError('--pretrained must be a file or "redimnet:<name>" / "redimnet2:<name>", got %r' % spec)
    family, name = spec.split(':', 1)
    name = name[:-3] if name.endswith('.pt') else name
    url = URLS[family].format(name=name)
    cache_dir = cache_dir or default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, '%s__%s.pt' % (family, name))
    if not os.path.exists(dst):
        print('Downloading', url)
        torch.hub.download_url_to_file(url, dst + '.part', progress=False)
        os.replace(dst + '.part', dst)
    return dst, family


def _torch_load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def build(family, model_config):
    return wrap_class(family)(**copy.deepcopy(model_config))


def _try_build(cfg, sd, family=None):
    '''Build the network of `family` (or guess it) and load `sd` strictly.'''
    errors = []
    for fam in ([family] if family else ['redimnet', 'redimnet2']):
        try:
            net = build(fam, cfg)
            r = net.load_state_dict(sd, strict=False)
            if r.missing_keys or r.unexpected_keys:
                raise RuntimeError('missing %s / unexpected %s' % (r.missing_keys[:3], r.unexpected_keys[:3]))
            return net, fam
        except Exception as e:  # wrong family -> unknown kwargs or key mismatch
            errors.append('%s: %s' % (fam, str(e)[:300]))
    raise RuntimeError('Could not load checkpoint into ReDimNet / ReDimNet2:\n  ' + '\n  '.join(errors))


class Encoder(nn.Module):
    '''waveform [B, T] (16 kHz) -> embedding [B, D]. Holds the original network in `.net`.'''

    def __init__(self, net, family, model_config):
        super().__init__()
        self.net = net
        self.family = family
        self.model_config = copy.deepcopy(model_config)

    def forward(self, wav):
        return self.net(wav)

    @property
    def embed_dim(self):
        return self.model_config.get('embed_dim', 192)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    # ---- checkpoints in the authors' format ----
    def export(self, path, **extra):
        ck = {'model_config': self.model_config, 'state_dict': self.net.state_dict(), 'family': self.family}
        ck.update(extra)
        tmp = path + '.tmp'
        torch.save(ck, tmp)
        os.replace(tmp, path)


def load_encoder(spec, cache_dir='', specaug=None, verbose=True):
    '''Load an encoder from a release name, an authors' .pt, a model saved by this repo
    (model_XXXX.model / best.model) or a training checkpoint (last.ckpt).

    specaug: None = keep the checkpoint's SpecAugment settings, otherwise a dict merged into
    model_config["spec_params"] (e.g. {"do_spec_aug": True, "freq_mask_width": [0, 10]}).
    SpecAugment only runs in training mode and has no parameters, so this never changes inference.
    Returns (encoder, extra) where extra holds additional keys of the file (e.g. center_mean).'''
    path, family = resolve(spec, cache_dir)
    ck = _torch_load(path)
    if 'encoder' in ck and isinstance(ck['encoder'], dict):      # last.ckpt of this repo
        ck = ck['encoder']
    if 'model_config' not in ck or 'state_dict' not in ck:
        raise ValueError('%s is not a ReDimNet checkpoint (needs model_config + state_dict)' % path)
    family = family or ck.get('family')
    cfg = copy.deepcopy(ck['model_config'])
    net, family = _try_build(cfg, ck['state_dict'], family)
    if specaug:
        sp = dict(cfg.get('spec_params') or {})
        sp.update(specaug)
        cfg['spec_params'] = sp
        net2 = build(family, cfg)
        r = net2.load_state_dict(net.state_dict(), strict=False)
        assert not r.missing_keys and not r.unexpected_keys, r
        net = net2
    enc = Encoder(net, family, cfg)
    extra = {k: v for k, v in ck.items() if k not in ('model_config', 'state_dict', 'family')}
    if verbose:
        print('Encoder: %s (%s) | %.3fM params | embed %d | spec_params %s'
              % (spec, family, enc.n_params() / 1e6, enc.embed_dim, cfg.get('spec_params')))
    return enc, extra


def encoder_state_from_file(path):
    '''(family, model_config, state_dict) of any supported checkpoint, without building the network.'''
    ck = _torch_load(path)
    if 'encoder' in ck and isinstance(ck['encoder'], dict):
        ck = ck['encoder']
    return ck.get('family'), ck['model_config'], ck['state_dict']
