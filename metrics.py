'''
Detection metrics that stay comparable across test sets with different attack rates.

AP depends heavily on prevalence (malicious / total): a random ranking scores AP equal to the
prevalence, which is ~0.0016 on the LANL random split but ~0.00026 on the dirty split. These
metrics normalize for that or translate it into analyst workload.

  ap_lift        AP / prevalence (1.0 = random ranking)
  prec_r{R}      precision at the score threshold that first reaches R% recall
  fp_day_r{R}    benign edges flagged per day of test traffic at that threshold

Thresholds treat ties conservatively: every edge scoring at or above the threshold is flagged,
so tied scores can't make a detector look better than an analyst would experience it.

Numpy only, so it can be used by the data-only sanity checks as well as training code.
'''
import math

import numpy as np

DAY = 60 * 60 * 24


def prevalence(labels):
    y = np.asarray(labels).astype(bool)
    return y.sum() / y.size if y.size else float('nan')


def threshold_metrics(labels, scores, recall=0.5):
    '''(precision, false positives, flagged) at the first threshold reaching `recall`. Higher score = more anomalous.'''
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float('nan'), float('nan'), float('nan')

    order = np.argsort(-s, kind='stable')
    target = math.ceil(recall * n_pos)
    k = int(np.searchsorted(np.cumsum(y[order]), target, side='left'))  # first rank reaching target
    thr = s[order[k]]

    flagged = s >= thr
    n_flagged = int(flagged.sum())
    tp = int((flagged & y).sum())
    return tp / n_flagged, n_flagged - tp, n_flagged


def test_days(ts, day=DAY):
    if ts is None or len(ts) == 0:
        return float('nan')
    ts = np.asarray(ts, dtype=np.float64)
    span = ts.max() - ts.min()
    return span / day if span > 0 else float('nan')


def detection_metrics(labels, scores, ap, ts=None, recall=0.5, prefix='te_'):
    '''Extra metrics for one labeled split, given its already-computed AP.'''
    r = int(round(recall * 100))
    prev = prevalence(labels)
    prec, fp, _ = threshold_metrics(labels, scores, recall)
    days = test_days(ts)
    return {
        f'{prefix}ap_lift': ap / prev if prev and prev > 0 else float('nan'),
        f'{prefix}prec_r{r}': prec,
        f'{prefix}fp_day_r{r}': fp / days if days == days and days > 0 else float('nan'),
    }