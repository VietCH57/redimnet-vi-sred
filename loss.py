'''
Additive angular margin softmax (AAM / ArcFace), as in voxceleb_trainer and TaoRuijie/ECAPA-TDNN,
with two additions: the embedding size is a parameter and the margin can be changed during
training (margin scheduler, as in WeSpeaker / ReDimNet recipes).
'''

import math, torch
import torch.nn as nn
import torch.nn.functional as F


class AAMsoftmax(nn.Module):
    def __init__(self, n_class, embed_dim=192, m=0.2, s=30.0):
        super().__init__()
        self.s = s
        self.weight = nn.Parameter(torch.FloatTensor(n_class, embed_dim))
        nn.init.xavier_normal_(self.weight, gain=1)
        self.ce = nn.CrossEntropyLoss()
        self.set_margin(m)

    def set_margin(self, m):
        self.m = float(m)
        self.cos_m, self.sin_m = math.cos(self.m), math.sin(self.m)
        self.th = math.cos(math.pi - self.m)
        self.mm = math.sin(math.pi - self.m) * self.m

    @torch.no_grad()
    def init_from_centroids(self, centroids):
        '''Initialise the class weights with (normalised) speaker centroids of the pretrained encoder.'''
        assert centroids.shape == self.weight.shape, (centroids.shape, self.weight.shape)
        self.weight.copy_(F.normalize(centroids.float(), dim=1))

    def forward(self, x, label):
        x = x.float()
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = self.s * ((one_hot * phi) + ((1.0 - one_hot) * cosine))
        loss = self.ce(output, label)
        acc = (output.detach().argmax(1) == label).float().mean() * 100
        return loss, acc
