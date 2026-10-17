'''
Fine-tuning loop for ReDimNet / ReDimNet2, following the official training code
(PalabraAI/redimnet2 applications/speaker_rec/wespeaker_lite: bin/train.py, utils/executor.py,
utils/schedulers.py, utils/utils.py; itself a fork of WeSpeaker).

  * encoder = unmodified ReDimNet(2) network (models.Encoder); head = AAM or SphereFace2 (loss.py),
    always computed in fp32, encoder forward under fp16 autocast (executor.run_epoch)
  * optimiser: two parameter groups, encoder (weight_decay) and head (weight_decay * head_wd_mult =
    loss_weight_decay_mult), SGD momentum 0.9 Nesterov (train.py)
  * LR: ExponentialDecrease, updated every optimiser step, x scale_ratio = world * batch / 64, warm-up
    from 0 (schedulers.ExponentialDecrease / BaseClass.get_multi_process_coeff)
  * margin: MarginScheduler (constant margin when m_start == m, as in the LM recipes)
  * head of a new speaker set: rows of speakers present in the initial checkpoint are copied
    (utils.reassemble_projection_head); the other rows are speaker centroids of the pretrained encoder
    (weight imprinting, Qi et al. CVPR 2018) or xavier (official fallback when no row can be copied)
Options that are NOT part of the official recipes and are off by default: --freeze_epochs, --l2sp
(Li et al., ICML 2018), --layer_decay, --bn_mode frozen, --grad_clip, --wd_skip_1d, AdamW.
Multi-GPU: DistributedDataParallel when launched with torchrun; the wrapper is kept outside the
registered modules -> checkpoint keys do not change. Only rank 0 writes files.
'''

import contextlib, math, os, random, sys, time
import numpy, torch
import torch.nn as nn
import torch.nn.functional as F
import models
import dist_utils as du
from loss import build_head
from dataLoader import load_wav, SR
from inference import parse_trials, extract_embeddings, score_trials, mean_embedding, subset_metrics
from tools import compute_metrics


def specaug_override(args):
    '''None = keep the model_config of the checkpoint (official: use the config the model was trained with).'''
    sp = {}
    if args.specaug != 'checkpoint':
        sp['do_spec_aug'] = args.specaug == 'true'
    if args.specaug_freq:
        sp['freq_mask_width'] = list(args.specaug_freq)
    if args.specaug_time:
        sp['time_mask_width'] = list(args.specaug_time)
    return sp or None


class _EncHead(nn.Module):
    '''encoder + head in one module, so that DDP synchronises the gradients of both.'''

    def __init__(self, enc, head, amp):
        super().__init__()
        self.enc, self.head, self.amp = enc, head, amp

    def forward(self, x, y):
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
            e = self.enc(x)
        return self.head(e.float(), y)


class Schedule:
    '''Official LR and margin schedules, as functions of the optimiser-step index `it` (0-based).'''

    def __init__(self, args, epoch_iter, scale_ratio):
        self.a, self.epoch_iter, self.scale = args, max(int(epoch_iter), 1), float(scale_ratio)
        self.freeze_iter = args.freeze_epochs * self.epoch_iter
        self.max_iter = max((args.max_epoch - args.freeze_epochs) * self.epoch_iter, 1)
        self.warm_iter = args.warmup_epochs * self.epoch_iter
        m = args
        self.inc_start = (m.m_increase_start_epoch - 1) * self.epoch_iter
        self.fix_start = (m.m_fix_start_epoch - 1) * self.epoch_iter

    def coeff(self, it):                                   # BaseClass.get_multi_process_coeff
        c = self.scale
        if it < self.warm_iter:
            if self.a.warm_from_zero:
                c = self.scale * it / self.warm_iter
            elif self.scale > 1:
                c = (self.scale - 1) * it / self.warm_iter + 1.0
        return c

    def lr(self, it):
        '''Encoder LR at optimiser step `it` (during --freeze_epochs: the head LR at its peak).'''
        a = self.a
        if it < self.freeze_iter:
            return self.scale * a.lr
        it = it - self.freeze_iter
        u = it / self.max_iter
        if a.scheduler == 'exp':                           # ExponentialDecrease.get_current_lr
            base = a.lr * math.exp(u * math.log(a.final_lr / a.lr))
        elif a.scheduler == 'cosine':
            base = a.final_lr + (a.lr - a.final_lr) * 0.5 * (1 + math.cos(math.pi * min(u, 1.0)))
        else:
            base = a.lr
        return self.coeff(it) * base

    def margin(self, it):                                  # MarginScheduler.step
        a = self.a
        if it >= self.fix_start:
            return a.m
        if it >= self.inc_start:
            cur, inc = it - self.inc_start, max(self.fix_start - self.inc_start, 1)
            if a.m_increase_type == 'exp':
                ratio = 1.0 - math.exp((cur / inc) * math.log(1e-3 / (1.0 + 1e-6)))
            else:
                ratio = cur / inc
            return a.m_start + (a.m - a.m_start) * ratio
        return a.m_start


class Trainer(nn.Module):
    def __init__(self, args, n_class, init_spec=None):
        super().__init__()
        self.args = args
        self.device = torch.device(args.device if (torch.cuda.is_available() or not args.device.startswith('cuda')) else 'cpu')
        spec = init_spec or args.pretrained
        self.encoder, self.extra = models.load_encoder(spec, args.ckpt_dir, specaug=specaug_override(args))
        n = self.encoder.n_params()
        if args.max_params and n > args.max_params:
            raise SystemExit('Encoder has %d parameters > --max_params %d' % (n, args.max_params))
        self.encoder.to(self.device)
        self.head = build_head(args, n_class, self.encoder.embed_dim).to(self.device)
        self.n_class = n_class
        self.ddp = du.is_dist()
        if self.ddp:
            print('DistributedDataParallel: %d processes (this is rank %d, device %s)' % (du.world(), du.rank(), self.device))
        self._build_optimizer()
        self.amp = bool(args.amp) and self.device.type == 'cuda'
        self._combo = [_EncHead(self.encoder, self.head, self.amp)]   # list: not a registered sub-module
        self._ddp_model, self._ddp_frozen = None, None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)
        self.anchor = None
        self.center = None          # (mean_full, mean_crops) of the last evaluation
        self.sched = None
        self.step = 0               # optimiser steps done (drives the LR / margin schedules)
        print(time.strftime('%m-%d %H:%M:%S') + ' Encoder %.3fM params (%s), head %s %d x %d'
              % (n / 1e6, self.encoder.family, args.loss, n_class, self.encoder.embed_dim))

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
            d = depth_of.get(id(p), top)          # front-end / pool / bn / linear = top
            no_wd = a.wd_skip_1d and p.ndim <= 1
            key = ('enc', d, no_wd)
            if key not in groups:
                groups[key] = {'params': [], 'lr_mult': a.layer_decay ** (top - d),
                               'weight_decay': 0.0 if no_wd else a.weight_decay, 'kind': 'enc'}
            groups[key]['params'].append(p)
        groups[('head',)] = {'params': list(self.head.parameters()), 'lr_mult': a.lr_head_mult,
                             'weight_decay': a.weight_decay * a.head_wd_mult, 'kind': 'head'}
        pg = list(groups.values())
        for g in pg:
            g['lr'] = 0.0
        if a.optimizer == 'adamw':
            self.optim = torch.optim.AdamW(pg, lr=a.lr)
        elif a.optimizer == 'adam':
            self.optim = torch.optim.Adam(pg, lr=a.lr)
        else:
            self.optim = torch.optim.SGD(pg, lr=a.lr, momentum=a.momentum, nesterov=a.nesterov)

    def set_schedule(self, epoch_iter, scale_ratio):
        self.sched = Schedule(self.args, epoch_iter, scale_ratio)

    def _apply_schedule(self, frozen):
        lr = self.sched.lr(self.step)
        for g in self.optim.param_groups:
            g['lr'] = 0.0 if (frozen and g['kind'] == 'enc') else lr * g['lr_mult']
        self.head.update(self.sched.margin(self.step))
        return lr

    # ------------------------------------------------------------------ L2-SP (optional, not official)
    def set_anchor(self, path, state=None):
        if self.args.l2sp <= 0:
            return
        if state is not None:
            sd = state
        elif du.is_main() and os.path.exists(path):
            sd = torch.load(path, map_location=self.device)
        else:   # start of the run: identical weights on every rank
            sd = {k: v.detach().clone() for k, v in self.encoder.named_parameters()}
            if du.is_main() and not os.path.exists(path):
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
        self.encoder.net.spec.train(True)                 # SpecAugment (if enabled) stays on
        self.head.train(True)

    def _mem(self):
        if self.device.type != 'cuda':
            return ''
        return ' | GPU%d peak %.1f GB' % (self.device.index or 0, torch.cuda.max_memory_allocated(self.device) / 2 ** 30)

    def _forward_fn(self, frozen):
        '''Callable (data, labels) -> (loss, acc). With DDP the wrapper is rebuilt when the set of trainable
        parameters changes (frozen encoder -> full network), because DDP registers them at construction.'''
        if not self.ddp:
            return self._combo[0], None
        if self._ddp_model is None or self._ddp_frozen != frozen:
            from torch.nn.parallel import DistributedDataParallel as DDP
            ids = [self.device.index] if self.device.type == 'cuda' else None
            self._ddp_model = DDP(self._combo[0], device_ids=ids, output_device=ids[0] if ids else None,
                                  broadcast_buffers=self.args.bn_mode == 'train')
            self._ddp_frozen = frozen
        return self._ddp_model, self._ddp_model

    def train_network(self, epoch, loader):
        a = self.args
        frozen = epoch <= a.freeze_epochs
        self._set_frozen(frozen)
        if hasattr(loader.batch_sampler, 'set_epoch'):
            loader.batch_sampler.set_epoch(epoch)
        n_batches = len(loader)
        accum = max(1, a.accum_steps)
        self.step = (epoch - 1) * self.sched.epoch_iter         # exact position even after a resume
        tot_loss = tot_acc = tot_reg = 0.0
        n_reg = 0
        t0 = time.time()
        fwd, ddp = self._forward_fn(frozen)
        show = du.is_main()
        lr = self._apply_schedule(frozen)
        self.optim.zero_grad(set_to_none=True)
        for num, (data, labels) in enumerate(loader, start=1):
            data, labels = data.to(self.device, non_blocking=True), labels.to(self.device, non_blocking=True)
            last_of_step = num % accum == 0 or num == n_batches
            n_in_step = accum if num <= (n_batches // accum) * accum else n_batches % accum
            sync = ddp.no_sync() if (ddp is not None and not last_of_step) else contextlib.nullcontext()
            with sync:
                loss, acc = fwd(data, labels)
                reg = self.l2sp_loss() if (last_of_step and not frozen) else None     # once per optimiser step
                total = loss / n_in_step if reg is None else loss / n_in_step + reg
                self.scaler.scale(total).backward()
            if last_of_step:
                if a.grad_clip > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_([p for g in self.optim.param_groups for p in g['params']], a.grad_clip)
                self.scaler.step(self.optim)
                self.scaler.update()
                self.optim.zero_grad(set_to_none=True)
                if reg is not None:
                    tot_reg += reg.item(); n_reg += 1
                self.step += 1
                lr = self._apply_schedule(frozen)
            tot_loss += loss.item(); tot_acc += acc.item()
            if show and (num % a.log_interval == 0 or num == n_batches):
                el = time.time() - t0
                sys.stderr.write(time.strftime('%m-%d %H:%M:%S') +
                                 ' [%2d] %s%.1f%% | lr %.2e m %.3f | loss %.4f acc %.2f%%%s | %.1f min/epoch%s\r'
                                 % (epoch, 'FROZEN ' if frozen else '', 100 * num / n_batches, lr, self.head.m,
                                    tot_loss / num, tot_acc / num, (' l2sp %.2e' % (tot_reg / max(n_reg, 1))) if a.l2sp > 0 else '',
                                    el / num * n_batches / 60, self._mem()))
                sys.stderr.flush()
        if show:
            sys.stderr.write('\n')
        st = torch.tensor([tot_loss / n_batches, tot_acc / n_batches], device=self.device)
        st = du.all_reduce_sum(st) / du.world()          # average over GPUs
        return {'loss': st[0].item(), 'acc': st[1].item(), 'l2sp': tot_reg / max(n_reg, 1),
                'lr': lr, 'margin': self.head.m, 'frozen': frozen}

    # ------------------------------------------------------------------ head initialisation
    @torch.no_grad()
    def reassemble_head(self, ck, spk2id, speeds):
        '''utils.reassemble_projection_head of the official code, generalised to speed classes: the row of
        (speaker, speed) is copied from the checkpoint when both exist there. Returns a bool mask of rows set.'''
        old_spk2id, old_speeds, old_head = ck.get('spk2id'), ck.get('speeds', [1.0]), ck.get('head')
        done = torch.zeros(self.n_class, dtype=torch.bool)
        if not old_spk2id or old_head is None:
            return done
        W_old = old_head['weight'].to(self.device)
        n_old, n_new = len(old_spk2id), len(spk2id)
        matched = 0
        for k, sp in enumerate(speeds):
            ko = next((j for j, s in enumerate(old_speeds) if abs(s - sp) < 1e-6), None)
            if ko is None:
                continue
            for spk, i in spk2id.items():
                j = old_spk2id.get(spk)
                if j is not None:
                    self.head.weight[k * n_new + i] = W_old[ko * n_old + j]
                    done[k * n_new + i] = True
                    matched += k == 0
        if 'bias' in old_head and hasattr(self.head, 'bias') and old_head['bias'].shape == self.head.bias.shape:
            self.head.bias.copy_(old_head['bias'].to(self.device))
        print('Head reassembled from the initial checkpoint: %d/%d speakers matched, %d/%d rows copied'
              % (matched, n_new, int(done.sum()), self.n_class))
        return done

    @torch.no_grad()
    def init_head_centroids(self, trainset, utts_per_spk=4, sec=3.0, seed=0, rows_done=None):
        '''Rows not set yet = mean embedding of a few utterances of the speaker (centre crops, no noise).
        Speed-perturbed classes get centroids of the speed-perturbed audio.'''
        rng = random.Random(seed)
        by_spk = {}
        for i, l in enumerate(trainset.labels):
            by_spk.setdefault(l, []).append(i)
        pick = [(l, i) for l, idx in sorted(by_spk.items()) for i in rng.sample(idx, min(utts_per_spk, len(idx)))]
        L = int(sec * SR)
        all_items = [(k, l, i) for k in range(len(trainset.speeds)) for l, i in pick
                     if rows_done is None or not rows_done[l + k * trainset.n_spk]]
        if not all_items:
            return
        items = all_items[du.rank()::du.world()]           # DDP: every GPU embeds its share
        speeds, files, crop = trainset.speeds, trainset.files, trainset._speed

        class _DS(torch.utils.data.Dataset):
            def __len__(self):
                return len(items)

            def __getitem__(self, j):
                k, _, i = items[j]
                x = crop(load_wav(files[i]).astype(numpy.float64), speeds[k], L, center=True)
                return torch.from_numpy(x.astype(numpy.float32))
        dl = torch.utils.data.DataLoader(_DS(), batch_size=64, num_workers=self.workers())
        self.encoder.eval()
        embs = []
        for x in dl:
            with torch.autocast('cuda', dtype=torch.float16, enabled=self.amp):
                embs.append(F.normalize(self.encoder(x.to(self.device)).float(), dim=1).cpu())
        E = torch.cat(embs) if embs else torch.zeros(0, self.encoder.embed_dim)
        cls = torch.tensor([l + k * trainset.n_spk for k, l, _ in items], dtype=torch.long)
        C = torch.zeros(self.n_class, self.encoder.embed_dim).index_add_(0, cls, E).to(self.device)
        C = du.all_reduce_sum(C)
        rows = sorted({l + k * trainset.n_spk for k, l, _ in all_items})
        W = self.head.weight.detach().clone()
        W[rows] = F.normalize(C[rows], dim=1)
        if rows_done is not None and rows_done.any():
            W[rows_done.to(W.device)] = F.normalize(W[rows_done.to(W.device)], dim=1)
        self.head.weight.copy_(W)
        print('Head: %d rows initialised with class centroids (%d utterances per speaker, %d speeds)'
              % (len(rows), utts_per_spk, len(speeds)))

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
                  max_batch=a.eval_batch, amp=a.eval_amp, workers=self.workers(), progress=False)
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        self.sync_buffers()
        emb = self._embed(files, eval_path, kw)
        cemb = self._embed(cohort[0], cohort[1], kw) if (cohort and cohort[0]) else None
        if not du.is_main():
            return None
        scores = score_trials(emb, enroll, test, a.eval_mode)
        res = compute_metrics(scores, labels, p_targets)
        if eval_meta:
            res.update(subset_metrics(scores, labels, enroll, test, eval_meta, p_targets[0]))
        if cemb is not None:
            self.center = mean_embedding(cemb)
            res['EER_center'] = compute_metrics(score_trials(emb, enroll, test, a.eval_mode, self.center), labels, [])['EER']
        return res

    @torch.no_grad()
    def sync_buffers(self):
        '''DDP: copy rank 0's BatchNorm running statistics to every rank. During training DDP broadcasts them
        at the start of each forward, after which every rank updates its own copy with its own batch, so
        the ranks end an epoch with slightly different statistics. Evaluation is split over the ranks and
        rank 0's model is the one saved, so every rank must embed with rank 0's statistics.'''
        if not self.ddp:
            return
        import torch.distributed as dist
        # only normalisation statistics change during training; the other buffers (e.g. the STFT / mel kernels
        # of ReDimNet2's front-end, some of them non-contiguous, which NCCL refuses) are constants
        for mod in self.encoder.modules():
            if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                for b in mod.buffers(recurse=False):
                    if b.is_contiguous():
                        dist.broadcast(b, src=0)
                    else:
                        t = b.contiguous(); dist.broadcast(t, src=0); b.copy_(t)

    def workers(self):
        '''DataLoader workers of this process: --n_cpu is the total, shared by the DDP processes.'''
        return max(1, self.args.n_cpu // du.world())

    def _embed(self, files, root, kw):
        '''Embeddings of `files`; with DDP every rank embeds files[rank::world] and rank 0 receives all.'''
        if not self.ddp:
            return extract_embeddings(self.encoder, files, root, **kw)
        mine = files[du.rank()::du.world()]
        part = extract_embeddings(self.encoder, mine, root, **kw) if mine else {}
        out = {}
        for p in du.gather_objects(part):
            out.update(p)
        return out

    # ------------------------------------------------------------------ checkpoints
    def save_model(self, path, **meta):
        '''Encoder only, in the authors' format (+ family, center_mean, meta).'''
        extra = {'meta': meta}
        if self.center is not None:
            extra['center_mean'] = {'full': self.center[0], 'crops': self.center[1]}
        self.encoder.export(path, **extra)

    def save_checkpoint(self, path, epoch, spk2id, speeds, extra=None):
        ck = {'encoder': {'model_config': self.encoder.model_config, 'state_dict': self.encoder.net.state_dict(),
                          'family': self.encoder.family},
              'head': self.head.state_dict(), 'loss': self.args.loss, 'n_class': self.n_class,
              'spk2id': dict(spk2id), 'speeds': list(speeds),
              'optim': self.optim.state_dict(), 'scaler': self.scaler.state_dict(), 'epoch': epoch,
              'center': self.center}
        if self.anchor is not None:
            ck['anchor'] = {k: p0 for (k, _), (_, p0) in zip(self.encoder.named_parameters(), self.anchor)}
        if extra:
            ck.update(extra)
        torch.save(ck, path + '.tmp')
        os.replace(path + '.tmp', path)

    def load_resume(self, path, spk2id, speeds):
        '''Exact resume of the same run: encoder, head, optimiser, scaler.'''
        ck = torch.load(path, map_location=self.device, weights_only=False)
        if ck.get('spk2id') != dict(spk2id) or list(ck.get('speeds', [])) != list(speeds) or ck.get('loss') != self.args.loss:
            raise SystemExit('%s belongs to another run (different speakers / speeds / loss): use another --save_path '
                             'or --resume none' % path)
        self.encoder.net.load_state_dict(ck['encoder']['state_dict'])
        self.head.load_state_dict(ck['head'])
        self.optim.load_state_dict(ck['optim'])
        if 'scaler' in ck:
            self.scaler.load_state_dict(ck['scaler'])
        self.center = ck.get('center')
        return ck
