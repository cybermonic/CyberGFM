import numpy as np
import numba

# Stolen from https://github.com/diditforlulz273/fastauc

def fast_auc(y_true: np.array, y_score: np.array, sample_weight: np.array=None) -> float:
    """a function to calculate AUC via python + numba.

    Args:
        y_true (np.array): 1D numpy array as true labels.
        y_score (np.array): 1D numpy array as probability predictions.
        sample_weight (np.array): 1D numpy array as sample weights, optional.

    Returns:
        AUC score as float
    """

    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    if sample_weight is None:
        return fast_numba_auc_nonw(y_true=y_true, y_score=y_score)
    else:
        sample_weight = sample_weight[desc_score_indices]
        return fast_numba_auc_w(y_true=y_true, y_score=y_score, sample_weight=sample_weight)


@numba.njit
def trapezoid_area(x1: float, x2: float, y1: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    return dx * y1 + dy * dx / 2.0


@numba.njit
def fast_numba_auc_nonw(y_true: np.array, y_score: np.array) -> float:
    y_true = (y_true == 1)

    prev_fps = 0
    prev_tps = 0
    last_counted_fps = 0
    last_counted_tps = 0
    auc = 0.0
    for i in range(len(y_true)):
        tps = prev_tps + y_true[i]
        fps = prev_fps + (1 - y_true[i])
        if i == len(y_true) - 1 or y_score[i+1] != y_score[i]:
            auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
            last_counted_fps = fps
            last_counted_tps = tps
        prev_tps = tps
        prev_fps = fps
    return auc / (prev_tps*prev_fps)

@numba.njit
def fast_numba_auc_w(y_true: np.array, y_score: np.array, sample_weight: np.array) -> float:
    y_true = (y_true == 1)

    prev_fps = 0
    prev_tps = 0
    last_counted_fps = 0
    last_counted_tps = 0
    auc = 0.0
    for i in range(len(y_true)):
        weight = sample_weight[i]
        tps = prev_tps + y_true[i] * weight
        fps = prev_fps + (1 - y_true[i]) * weight
        if i == len(y_true) - 1 or y_score[i+1] != y_score[i]:
            auc += trapezoid_area(last_counted_fps, fps, last_counted_tps, tps)
            last_counted_fps = fps
            last_counted_tps = tps
        prev_tps = tps
        prev_fps = fps
    return auc / (prev_tps * prev_fps)


# Adding in extra functions for AP
def fast_ap(y_true, y_score):
    desc_score_indices = np.argsort(y_score)[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    return fast_numba_ap_nonw(y_true)

@numba.njit
def fast_numba_ap_nonw(y_true: np.array) -> float:
    y_true = (y_true == 1)
    ap = 0

    tot_true = y_true.sum()
    tps = 0
    fps = 0

    prev_re = 0
    for i in range(len(y_true)):
        tps += y_true[i]
        fps += 1-y_true[i]

        if y_true[i]: # Change in Re
            pr = tps / (tps + fps)
            re = tps / tot_true
            ap += (re - prev_re) * pr

            prev_re = re

        if re == 1:
            break

    return ap