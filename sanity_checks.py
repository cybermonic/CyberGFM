'''
Data-only checks on the train/val/test splits (no GPU or model code; needs torch and scikit-learn).

For each dataset:
  * split sizes, test attack rate, the AP of a random ranking, and the test period in days
  * whether validation is a later time window or sampled from the training period
  * how often (src, dst) pairs in val / benign test / malicious test already appear in training;
    validation far above benign test means validation is easier than the real task
  * how many test edges touch a host never seen in training
  * an "unseen-host" baseline that flags only those edges, scored with the same metrics as
    common.evaluate (AUC, AP, AP lift, precision and benign flags per day at 50% recall)

    python sanity_checks.py
    python sanity_checks.py --datasets optc-ts lanl14argus-dirtyts
'''
from argparse import ArgumentParser
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from metrics import detection_metrics, prevalence, test_days


def load(ds, split):
    fn = f'data/{ds}_tgraph_{split}.pt'
    return torch.load(fn, weights_only=False) if os.path.exists(fn) else None


def pair_ids(g, n):
    return g.src.long() * n + g.col.long()


def seen_nodes(tr):
    seen = torch.zeros(tr.x.size(0), dtype=torch.bool)
    seen[tr.src] = True
    seen[tr.col] = True
    return seen


def pct(x):
    return f'{x.float().mean().item():.1%}' if x.numel() else '-'


def check_split(ds, tr, va, te):
    y = te.label.bool()
    n_mal = int(y.sum())
    days = test_days(te.ts.numpy())
    print(f'  test: {te.col.size(0):,} edges, {n_mal:,} malicious (rate {prevalence(y.numpy()):.2e} '
          f'= AP of random ranking), {days:.1f} days')

    tr_lo, tr_hi = int(tr.ts.min()), int(tr.ts.max())
    if va is not None:
        va_lo, va_hi = int(va.ts.min()), int(va.ts.max())
        kind = ('later time window' if va_lo >= tr_hi
                else 'sampled from the training period' if tr_lo <= va_lo and va_hi <= tr_hi
                else 'overlaps the end of training')
        print(f'  time ranges: train [{tr_lo}, {tr_hi}], val [{va_lo}, {va_hi}] ({kind}), '
              f'test [{int(te.ts.min())}, {int(te.ts.max())}]')


def check_repeats(tr, va, te):
    n = tr.x.size(0)
    seen = torch.unique(pair_ids(tr, n))
    y = te.label.bool()
    te_seen = torch.isin(pair_ids(te, n), seen)
    va_part = f'val {pct(torch.isin(pair_ids(va, n), seen))} | ' if va is not None else ''
    print(f'  (src, dst) pair already in training: {va_part}'
          f'test benign {pct(te_seen[~y])} | test malicious {pct(te_seen[y])}')


def check_unseen_hosts(tr, te):
    seen = seen_nodes(tr)
    unseen = ~(seen[te.src] & seen[te.col])
    y = te.label.bool()
    print(f'  test edges touching a host never seen in training: {pct(unseen)} overall '
          f'({int(unseen.sum()):,} edges), {pct(unseen[y])} of malicious ({int(unseen[y].sum()):,} edges)')

    labels = y.numpy()
    if labels.sum() == 0:
        return
    rng = np.random.default_rng(0)
    scores = unseen.float().numpy() + rng.random(labels.size) * 1e-3  # random order within each group
    ap = average_precision_score(labels, scores)
    m = detection_metrics(labels, scores, ap, ts=te.ts.numpy(), recall=0.5)
    print(f'  unseen-host baseline: AUC {roc_auc_score(labels, scores):.4f}  AP {ap:.4f}  '
          f'lift {m["te_ap_lift"]:.1f}x  @50% recall: precision {m["te_prec_r50"]:.4f}, '
          f'benign flagged/day {m["te_fp_day_r50"]:,.1f}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['lanl14argus', 'lanl14argus-dirtyts', 'optc-ts', 'unsw'])
    args = ap.parse_args()

    for ds in args.datasets:
        tr, va, te = load(ds, 'tr'), load(ds, 'va'), load(ds, 'te')
        if tr is None or te is None:
            print(f'\n== {ds}: files not found, skipping')
            continue
        print(f'\n== {ds}: {tr.col.size(0):,} train edges, '
              f'{va.col.size(0) if va is not None else 0:,} val edges, {te.col.size(0):,} test edges')
        check_split(ds, tr, va, te)
        check_repeats(tr, va, te)
        check_unseen_hosts(tr, te)