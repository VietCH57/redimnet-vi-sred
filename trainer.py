'''
Fine-tuning loop for ReDimNet / ReDimNet2 (the role ECAPAModel.py plays in the ECAPA repo).

  * encoder = unmodified ReDimNet(2) network (models.Encoder), head = AAM-softmax (loss.py)
  * head warm-up: --freeze_epochs epochs with the encoder frozen (BN statistics fixed)
  * per-iteration LR schedule (linear warm-up, then cosine / exponential / constant),
    separate LRs for head and encoder, optional layer-wise decay inside the encoder
  * L2-SP regularisation towards the starting weights, margin ramp, fp16 AMP, grad clipping
  * nn.DataParallel kept OUTSIDE the registered modules -> checkpoint keys do not change
'''

import math, os, random, sys, time
import numpy, torch
import torch.nn as nn
import torch.nn.functional as F
import models
from loss import AAMsoftmax
from dataLoader import load_wav, SR
from inference import parse_trials, extract_embeddings, score_trials, mean_embedding, subset_metrics
from tools import compute_metrics


def _specaug_override(args):
    sp = {'do_spec_aug': bool(args.specaug)}
    if args.specaug_freq:
        sp['freq_mask_width'] = list(args.specaug_freq)
    if args.specaug_time:
        sp['time_mask_width'] = list(args.specaug_time)
    return sp


class Trainer(nn.Module):
    def __init__(self, args, n_class, init_spec=None):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device if (torch.cuda.is_available() or not args.device.startswith('cuda')) else 'cpu')
        spec = init_spec or args.pretrained
        self.encoder, self.extra = models.load_encoder(spec, args.ckpt_dir, specaug=_specaug_override(args))
        n = self.encoder.n_params()
        if args.max_params and n > args.max_params:
            raise SystemExit('Encoder has %d parameters > --max_params %d' % (n, args.max_params))
        self.encoder.to(self.device)
        self.head = AAMsoftmax(n_class, self.encoder.embed_dim, m=args.m, s=args.s).to(self.device)
        self.n_class = n_class
        self._train_encoder = [self.encoder]
        if args.data_parallel and self.device.type == 'cuda' and torch.cuda.device_count() > 1:
            self._train_encoder = [nn.DataParallel(self.encoder)]
            print('DataParallel on %d GPUs' % torch.cuda.device_count())
        self._build_optimizer()
        self.amp = bool(args.amp) and self.device.type == 'cuda'
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.anchor = None
        self.center = None          # (mean_full, mean_crops) of the last evaluation
        print(time.strftime('%m-%d %H:%M:%S') + ' Encoder %.3fM params (%s), head %d x %d'
              % (n / 1e6, self.encoder.family, n_class, self.encoder.embed_dim))

    # ------------------------------------------------------------------ optimiser / schedule
    def _build_optimizer(self):
        a = self.args
        bb = [(n, c) for n, c in self.encoder.net.backbone.named_children() if any(True for _ in c.parameters())]
        depth_of, top = {}, len(bb)
        for d, (n, c) in enumerate(bb):
            for p in c.parameters():
                depth_of[id(p)] = d
        groups = {}
        for name, p in self.encoder.named_parameters():
            d = depth_of.get(id(p), top)          # pool / bn / linear = top
            no_wd = p.ndim <= 1
            key = ('enc', d, no_wd)
            if key not in groups:
                groups[key] = {'params': [], 'base_lr': a.lr_backbone * a.layer_decay ** (top - d),
                               'weight_decay': 0.0 if no_wd else a.weight_decay, 'kind': 'enc'}
            groups[key]['params'].append(p)
        groups[('head',)] = {'params': list(self.head.parameters()), 'base_lr': a.lr_head,
                             'weight_decay': a.weight_decay, 'kind': 'head'}
        pg = list(groups.values())
        for g in pg:
            g['lr'] = g['base_lr']
        if a.optimizer == 'adamw':
            self.optim = torch.optim.AdamW(pg, lr=a.lr_backbone)
        elif a.optimizer == 'adam':
            self.optim = torch.optim.Adam(pg, lr=a.lr_backbone)
        else:
            self.optim = torch.optim.SGD(pg, lr=a.lr_backbone, momentum=0.9, nesterov=True)

    def lr_factor(self, t):
        '''t = epochs elapsed (float, 0 = start of epoch 1). Factor of the peak LR, after the freeze phase.'''
        a = self.args
        t = t - a.freeze_epochs
        if t < 0:
            return 1.0                                    # head only, at its peak LR
        if a.warmup_epochs > 0 and t < a.warmup_epochs:
            return (t + 1e-3) / a.warmup_epochs
        T = max(a.max_epoch - a.freeze_epochs - a.warmup_epochs, 1e-6)
        u = min(max((t - a.warmup_epochs) / T, 0.0), 1.0)
        if a.scheduler == 'cosine':
            return a.lr_min_ratio + (1 - a.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * u))
        if a.scheduler == 'exp':
            return a.lr_decay ** (t - a.warmup_epochs)
        return 1.0

    def margin_at(self, t):
        a = self.args
        if a.m_warmup_epochs <= 0:
            return a.m
        return a.m_start + (a.m - a.m_start) * min(1.0, t / a.m_warmup_epochs)

    def _set_lr(self, t, frozen):
        f = self.lr_factor(t)
        for g in self.optim.param_groups:
            g['lr'] = 0.0 if (frozen and g['kind'] == 'enc') else g['base_lr'] * f
        return self.optim.param_groups[-1]['lr'], max(g['lr'] for g in self.optim.param_groups if g['kind'] == 'enc')

    # ------------------------------------------------------------------ L2-SP
    def set_anchor(self, path, state=None):
        '''Anchor = encoder weights at the start of the run. Saved to `path` and inside last.ckpt;
        on resume it is read back from the checkpoint (state) or from `path`.'''
        if self.args.l2sp <= 0:
            return
        if state is not None:
            sd = state
        elif os.path.exists(path):
            sd = torch.load(path, map_location=self.device)
        else:
            sd = {k: v.detach().clone() for k, v in self.encoder.named_parameters()}
            torch.save(sd, path)
        self.anchor = [(p, sd[k].to(self.device)) for k, p in self.encoder.named_parameters()]

    def l2sp_loss(self):
        if self.anchor is None:
            return None
        return 0.5 * self.args.l2sp * sum(((p - p0) ** 2).sum() for p, p0 in self.anchor)

    # ------------------------------------------------------------------ training
    def _set_frozen(self, frozen):
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        self.encoder.train(not frozen)
        if not frozen and self.args.bn_mode == 'frozen':  # keep the pretrained BN statistics
            for mod in self.encoder.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        self.encoder.net.spec.train(True)                 # SpecAugment stays on
        self.head.train(True)

    def _mem(self):
        if self.device.type != 'cuda':
            return ''
        return ' | GPU peak ' + '/'.join('%.1f' % (torch.cuda.max_memory_allocated(i) / 2 ** 30) for i in range(torch.cuda.device_count())) + ' GB'

    def train_network(self, epoch, loader):
        a = self.args
        frozen = epoch <= a.freeze_epochs
        self._set_frozen(frozen)
        if hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)
        n_batches = len(loader)
        tot_loss = tot_acc = tot_reg = 0.0
        n_reg = 0
        t0 = time.time()
        accum = max(1, a.accum_steps)
        self.optim.zero_grad(set_to_none=True)
        for num, (data, labels) in enumerate(loader, start=1):
            t = epoch - 1 + (num - 1) / n_batches
            lr_head, lr_enc = self._set_lr(t, frozen)
            self.head.set_margin(self.margin_at(t))
            data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                emb = self._train_encoder[0](data)
            loss, acc = self.head(emb.float(), labels)
            step = num % accum == 0 or num == n_batches
            reg = self.l2sp_loss() if (step and not frozen) else None     # once per optimiser step
            total = loss / accum if reg is None else loss / accum + reg
            self.scaler.scale(total).backward()
            if step:
                if a.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_([p for g in self.optim.param_groups for p in g['params']], a.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                if reg is not None:
                    tot_reg += reg.item(); n_reg += 1
            tot_loss += loss.item(); tot_acc += acc.item()
            if num % a.log_interval == 0 or num == n_batches:
                el = time.time() - t0
                sys.stderr.write(time.strftime('%m-%d %H:%M:%S') +
                                 ' [%2d] %s%.1f%% | lr_enc %.2e lr_head %.2e m %.3f | loss %.4f acc %.2f%% l2sp %.2e | %.1f min/epoch%s\r'
                                 % (epoch, 'FROZEN ' if frozen else '', 100 * num / n_batches, lr_enc, lr_head, self.head.m,
                                    tot_loss / num, tot_acc / num, tot_reg / max(n_reg, 1), el / num * n_batches / 60, self._mem()))
                sys.stderr.flush()
        sys.stderr.write('\n')
        return {'loss': tot_loss / n_batches, 'acc': tot_acc / n_batches, 'l2sp': tot_reg / max(n_reg, 1),
                'lr_enc': lr_enc, 'lr_head': lr_head, 'margin': self.head.m, 'frozen': frozen}

    # ------------------------------------------------------------------ head initialisation
    @torch.no_grad()
    def init_head_centroids(self, trainset, utts_per_spk=4, sec=3.0, seed=0):
        '''Class weights = mean embedding of a few utterances of each speaker (centre crops, no noise).
        Speed-perturbed classes get centroids of the speed-perturbed audio, so that the copies of a
        speaker are distinguishable from the start.'''
        rng = random.Random(seed)
        by_spk = {}
        for i, l in enumerate(trainset.labels):
            by_spk.setdefault(l, []).append(i)
        pick = [(l, i) for l, idx in sorted(by_spk.items()) for i in rng.sample(idx, min(utts_per_spk, len(idx)))]
        L = int(sec * SR)
        items = [(k, l, i) for k in range(len(trainset.speeds)) for l, i in pick]
        speeds, files, crop = trainset.speeds, trainset.files, trainset._speed

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, j):
                k, _, i = items[j]
                x = crop(load_wav(files[i]).astype(numpy.float64), speeds[k], L, center=True)
                return torch.from_numpy(x.astype(numpy.float32))
        dl = torch.utils.data.DataLoader(_DS(), batch_size=64, num_workers=self.args.n_cpu)
        self.encoder.eval()
        embs = []
        for x in dl:
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                embs.append(F.normalize(self.encoder(x.to(self.device)).float(), dim=1).cpu())
        E = torch.cat(embs)
        cls = torch.tensor([l + k * trainset.n_spk for k, l, _ in items])
        C = torch.zeros(self.n_class, E.shape[1]).index_add_(0, cls, E)
        self.head.init_from_centroids(F.normalize(C, dim=1).to(self.device))
        print('Head initialised with class centroids (%d utterances x %d speeds, %d classes)' % (len(pick), len(speeds), self.n_class))

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def eval_network(self, eval_list, eval_path, cohort=None, p_targets=(0.05,), eval_meta=''):
        '''Returns {"EER": .., "minDCF..": .., ["EER_center": ..], [subset EERs]}; cohort = (files, root) for center scoring.'''
        a = self.args
        enroll, test, labels = parse_trials(eval_list)
        if labels[0] is None:
            raise ValueError('%s has no labels' % eval_list)
        files = sorted(set(enroll) | set(test))
        kw = dict(device=self.device, mode=a.eval_mode, eval_sec=a.eval_sec, num_eval=a.num_eval,
                  min_sec=a.eval_min_sec, max_sec=a.eval_max_sec, len_tol=a.eval_len_tol,
                  max_batch=a.eval_batch, amp=a.eval_amp, workers=a.n_cpu, progress=False)
        emb = extract_embeddings(self.encoder, files, eval_path, **kw)
        scores = score_trials(emb, enroll, test, a.eval_mode)
        res = compute_metrics(scores, labels, p_targets)
        if eval_meta:
            res.update(subset_metrics(scores, labels, enroll, test, eval_meta, p_targets[0]))
        if cohort and cohort[0]:
            cemb = extract_embeddings(self.encoder, cohort[0], cohort[1], **kw)
            self.center = mean_embedding(cemb)
            res['EER_center'] = compute_metrics(score_trials(emb, enroll, test, a.eval_mode, self.center), labels, [])['EER']
        return res

    # ------------------------------------------------------------------ checkpoints
    def save_model(self, path, **meta):
        '''Encoder only, in the authors' format (+ family, center_mean, meta).'''
        extra = {'meta': meta}
        if self.center is not None:
            extra['center_mean'] = {'full': self.center[0], 'crops': self.center[1]}
        self.encoder.export(path, **extra)

    def save_checkpoint(self, path, epoch, extra=None):
        ck = {'encoder': {'model_config': self.encoder.model_config, 'state_dict': self.encoder.net.state_dict(),
                          'family': self.encoder.family},
              'head': self.head.state_dict(), 'n_class': self.n_class, 'optim': self.optim.state_dict(),
              'scaler': self.scaler.state_dict(), 'epoch': epoch, 'center': self.center}
        if self.anchor is not None:
            ck['anchor'] = {k: p0 for (k, _), (_, p0) in zip(self.encoder.named_parameters(), self.anchor)}
        if extra:
            ck.update(extra)
        torch.save(ck, path + '.tmp')
        os.replace(path + '.tmp', path)

    def load_checkpoint(self, path, with_optim=True):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder.net.load_state_dict(ck['encoder']['state_dict'])
        head_ok = ck.get('n_class') == self.n_class and 'head' in ck
        if head_ok:
            self.head.load_state_dict(ck['head'])
        if with_optim and head_ok:
            self.optim.load_state_dict(ck['optim'])
            if 'scaler' in ck:
                self.scaler.load_state_dict(ck['scaler'])
        self.center = ck.get('center')
        return ck, head_ok
