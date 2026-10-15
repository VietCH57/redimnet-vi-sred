'''
Metrics. EER follows voxceleb_trainer / TaoRuijie/ECAPA-TDNN (tuneThresholdfromScore: the larger of
FPR and FNR at the ROC point where they are closest), so numbers are comparable with the ECAPA runs.
minDCF is a vectorised version of the original ComputeMinDcf (same result).
'''

import numpy
from sklearn import metrics


def eer(scores, labels):
    fpr, tpr, _ = metrics.roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    i = numpy.nanargmin(numpy.absolute(fnr - fpr))
    return float(max(fpr[i], fnr[i]) * 100)


def min_dcf(scores, labels, p_target=0.05, c_miss=1, c_fa=1):
    scores = numpy.asarray(scores, dtype=numpy.float64)
    labels = numpy.asarray(labels, dtype=numpy.int64)
    order = numpy.argsort(scores, kind='mergesort')
    lab = labels[order]
    fnrs = numpy.cumsum(lab) / max(lab.sum(), 1)
    fprs = 1 - numpy.cumsum(1 - lab) / max((1 - lab).sum(), 1)
    c_det = c_miss * fnrs * p_target + c_fa * fprs * (1 - p_target)
    return float(c_det.min() / min(c_miss * p_target, c_fa * (1 - p_target)))


def compute_metrics(scores, labels, p_targets=(0.05,), prefix=''):
    res = {prefix + 'EER': eer(scores, labels)}
    for p in p_targets:
        res[prefix + 'minDCF%g' % p] = min_dcf(scores, labels, p)
    return res
