'''
Classification heads used to train ReDimNet / ReDimNet2, ported from the official training code.

  aam : additive angular margin softmax = ArcMarginProduct of WeSpeaker
        (wespeaker/models/projections.py; identical copy in PalabraAI/redimnet2
        applications/speaker_rec/wespeaker_lite/models/projections.py). ReDimNet was trained with
        WeSpeaker and AAM (Yakovlev et al., Interspeech 2024, Sec. 4.1; WeSpeaker conf/redimnet.yaml:
        project_type arc_margin, scale 32, easy_margin False).
  sf2 : SphereFace2 (Han et al., ICASSP 2023 "Exploring binary classification loss for speaker
        verification"; Wen et al. 2022 "SphereFace2"), class SphereFace2 of the ReDimNet2 training code,
        used by every ReDimNet2 recipe (project_type sphereface2, margin_type C, scale 32, lanbuda 0.7, t 3).
        Only the one-hot path is kept (no mixup / cutmix in our recipes); for one-hot targets the official
        `dynamic_margin` branch reduces exactly to this formula (see the comment in the official code).

Both heads return (loss, accuracy %) and expose update(margin) like the official classes, so the
official MarginScheduler logic applies unchanged. Changes w.r.t. the official code (no effect on the
values computed): sqrt argument clamped to [0, 1] (avoids NaN when |cos| rounds above 1), softplus
written with torch.nn.functional.softplus (= log(1 + exp(x)), numerically stable).
Ported code: Apache License 2.0, (c) 2021-2023 the WeSpeaker authors (Shuai Wang, Zhengyang Chen,
Hongji Wang, Bing Han) and PalabraAI; see the headers of the original files.
'''

import math, torch
import torch.nn as nn
import torch.nn.functional as F


class _Head(nn.Module):
    def __init__(self, n_class, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embed_dim))
        nn.init.xavier_uniform_(self.weight)          # as the official projections

    @property
    def m(self):
        return self.margin

    def set_margin(self, m):                          # old name used in this repo
        self.update(m)

    @torch.no_grad()
    def init_from_centroids(self, centroids):
        '''Class weights = (normalised) speaker centroids of the pretrained encoder (weight imprinting).'''
        assert centroids.shape == self.weight.shape, (centroids.shape, self.weight.shape)
        self.weight.copy_(F.normalize(centroids.float(), dim=1))


class AAMsoftmax(_Head):
    '''WeSpeaker ArcMarginProduct (easy_margin=False) + cross-entropy.'''

    def __init__(self, n_class, embed_dim=192, m=0.2, s=32.0):
        super().__init__(n_class, embed_dim)
        self.scale = float(s)
        self.ce = nn.CrossEntropyLoss()
        self.update(m)

    def update(self, margin=0.2):
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mmm = 1.0 + math.cos(math.pi - self.margin)

    def logits(self, x, label):
        cosine = F.linear(F.normalize(x.float()), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mmm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        return self.scale * (one_hot * phi + (1.0 - one_hot) * cosine)

    def forward(self, x, label):
        out = self.logits(x, label)
        loss = self.ce(out, label)
        acc = (out.detach().argmax(1) == label).float().mean() * 100
        return loss, acc


class SphereFace2(_Head):
    '''Official ReDimNet2 SphereFace2 (one-hot targets). margin_type C: cos(theta) - m, A: cos(theta + m).'''

    def __init__(self, n_class, embed_dim=192, m=0.2, s=32.0, lanbuda=0.7, t=3, margin_type='C'):
        super().__init__(n_class, embed_dim)
        assert margin_type in ('A', 'C'), margin_type
        self.scale, self.lanbuda, self.t, self.margin_type = float(s), float(lanbuda), int(t), margin_type
        self.bias = nn.Parameter(torch.zeros(1, 1))
        self.update(m)

    def update(self, margin=0.2):
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        self.th = math.cos(math.pi - self.margin)
        self.mmm = 1.0 + math.cos(math.pi - self.margin)

    def fun_g(self, z):
        return 2 * torch.pow((z + 1) / 2, self.t) - 1

    def forward(self, x, label):
        cos = F.linear(F.normalize(x.float()), F.normalize(self.weight))
        target = torch.zeros_like(cos)
        target.scatter_(1, label.view(-1, 1).long(), 1.0)
        nontarget = 1.0 - target
        b = self.bias[0][0]
        if self.margin_type == 'A':
            sin = torch.sqrt((1.0 - cos * cos).clamp(0, 1))
            p = self.scale * self.fun_g(torch.where(cos > self.th, cos * self.cos_m - sin * self.sin_m, cos - self.mmm)) + b
            n = self.scale * self.fun_g(cos * self.cos_m + sin * self.sin_m) + b
        else:
            p = self.scale * (self.fun_g(cos) - self.margin) + b
            n = self.scale * (self.fun_g(cos) + self.margin) + b
        loss_p = self.lanbuda * F.softplus(-p)          # lanbuda * log(1 + exp(-p))
        loss_n = (1 - self.lanbuda) * F.softplus(n)     # (1 - lanbuda) * log(1 + exp(n))
        loss = (target * loss_p + nontarget * loss_n).sum(1).mean()
        out = self.scale * (cos - self.margin * target)     # accuracy meter, as the official code
        acc = (out.detach().argmax(1) == label).float().mean() * 100
        return loss, acc


def build_head(args, n_class, embed_dim):
    if args.loss == 'aam':
        return AAMsoftmax(n_class, embed_dim, m=args.m, s=args.s)
    if args.loss == 'sf2':
        return SphereFace2(n_class, embed_dim, m=args.m, s=args.s, lanbuda=args.sf2_lanbuda, t=args.sf2_t,
                           margin_type=args.sf2_margin_type)
    raise ValueError('unknown --loss %r (aam / sf2)' % args.loss)
