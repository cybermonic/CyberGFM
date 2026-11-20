'''
Out-of-vocabulary breakdown on LANL-41h: how CyberGFM and Argus score test edges whose endpoints
were never seen in the training window, versus edges between known hosts.

A host is "seen" if it is an endpoint of any edge in the training window (the train + validation
files of the split, i.e. everything before the cutoff). Each test edge falls in one group:

    both seen        src seen, dst seen
    new dst          src seen, dst unseen
    new src          src unseen, dst seen
    both new         neither seen

For each method and group the table gives the edges and attack edges in it, and AP / AUC of the
method's ranking WITHIN the group (mean +- standard error over seeds). The last column,
"mal pct", is the median percentile of the group's attack edges in the method's FULL ranking
(100 = ranked most anomalous of all test edges), which shows where in the alert list those
attacks actually land.

Reads the score files written by the --save-scores runs (run_oov.sh); text and numbers only,
fine on the login node.

    python oov_breakdown.py
    python oov_breakdown.py --root results-oov --csv results-oov/oov_breakdown.csv
'''
import argparse
import csv
import glob
import math
import os

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

GROUPS = ['both seen', 'new dst', 'new src', 'both new', 'any new']


def load(path):
    return torch.load(path, weights_only=False)


def src_of(g):
    if 'src' in g:
        return g.src.long()
    return torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])


def groups(src, dst, seen):
    s, d = seen[src], seen[dst]
    return {'both seen': s & d, 'new dst': s & ~d, 'new src': ~s & d, 'both new': ~s & ~d,
            'any new': ~(s & d)}


def group_metrics(score, y, masks):
    order = np.argsort(-score, kind='stable')
    pct = np.empty(len(score))
    pct[order] = 100 * (1 - np.arange(len(score)) / len(score))     # 100 = top of the list
    out = {}
    for name, m in masks.items():
        ym, sm = y[m], score[m]
        r = dict(edges=int(m.sum()), attacks=int(ym.sum()))
        if ym.any() and (~ym.astype(bool)).any():
            r['ap'], r['auc'] = average_precision_score(ym, sm), roc_auc_score(ym, sm)
            r['base'] = ym.mean()
        if ym.any():
            r['mal_pct'] = float(np.median(pct[m & (y == 1)]))
        out[name] = r
    return out


def mean_se(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    se = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs)) if len(xs) > 1 else float('nan')
    return m, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='lanl14argus-ts')
    ap.add_argument('--data', default='data')
    ap.add_argument('--root', default='results-oov')
    ap.add_argument('--argus-exp', default='lanl14argus-dirtyts-41h',
                    help="argus_test_lanl_ts.py's experiment name at the 41 h cutoff")
    ap.add_argument('--csv')
    args = ap.parse_args()

    tr = load(f'{args.data}/{args.dataset}_tgraph_tr.pt')
    va = load(f'{args.data}/{args.dataset}_tgraph_va.pt')
    te = load(f'{args.data}/{args.dataset}_tgraph_te.pt')
    n = tr.x.size(0)
    seen = torch.zeros(n, dtype=torch.bool)
    for g in (tr, va):
        seen[src_of(g)] = True
        seen[g.col.long()] = True
    te_src, te_dst, te_y = src_of(te), te.col.long(), te.label.long().clamp(0, 1)
    new_hosts = ~seen[torch.unique(torch.cat([te_src, te_dst]))]
    print(f'{args.dataset}: {int(seen.sum()):,} hosts seen before the cutoff; '
          f'{int(new_hosts.sum()):,} of the {new_hosts.numel():,} hosts active in test are new')
    base = groups(te_src, te_dst, seen)
    print('test edges by group (CyberGFM\'s test file):')
    for k in GROUPS:
        m = base[k]
        print(f'  {k:<10} {int(m.sum()):>10,} edges ({m.float().mean():.1%}), '
              f'{int(te_y[m].sum()):>4} attacks ({te_y[m].sum() / max(1, te_y.sum()):.1%} of all attacks)')

    runs = {}   # method -> list of per-seed group metrics
    for path in sorted(glob.glob(f'{args.root}/lp-*/{args.dataset}/*_scores.npz')):
        mode = path.split(os.sep)[-3].split('-', 1)[1]
        z = np.load(path)
        score, y = z['score'], z['label'].astype(int)
        if len(score) != len(te_y) or not np.array_equal(y.clip(0, 1), te_y.numpy()):
            print(f'  skipping {path}: does not match the test file edge for edge')
            continue
        masks = {k: v.numpy() for k, v in groups(te_src, te_dst, seen).items()}
        runs.setdefault(f'CyberGFM ({mode[0].upper()})', []).append(group_metrics(score, y, masks))
    for path in sorted(glob.glob(f'{args.root}/argus/scores/{args.argus_exp}_s*.pt')):
        k = load(path)
        score, y = k['score'].double().numpy(), k['label'].long().clamp(0, 1).numpy()
        if k.get('weight') is not None:
            print(f'  note: {path} has deduplicated rows; groups count rows, not events')
        masks = {g: v.numpy() for g, v in groups(k['src'].long(), k['dst'].long(), seen).items()}
        runs.setdefault('Argus', []).append(group_metrics(score, y, masks))
    if not runs:
        print(f'no score files under {args.root} -- run run_oov.sh first')
        return

    rows = []
    for method, per_seed in runs.items():
        for gname in GROUPS:
            rec = dict(method=method, group=gname, seeds=len(per_seed),
                       edges=per_seed[0][gname]['edges'], attacks=per_seed[0][gname]['attacks'])
            for metric in ('ap', 'auc', 'base', 'mal_pct'):
                m, se = mean_se([r[gname].get(metric) for r in per_seed])
                rec[metric], rec[metric + '_se'] = m, se
            rows.append(rec)

    def f(m, se, d=4):
        if m is None:
            return '-'
        return f'{m:.{d}f}' + ('' if se is None or se != se else f' ± {se:.{d}f}')

    hdr = ['method', 'group', 'n', 'edges', 'attacks', 'base rate', 'AP', 'AUC', 'mal pct']
    table = [[r['method'], r['group'], str(r['seeds']), f'{r["edges"]:,}', str(r['attacks']),
              f(r['base'], None, 5) if r['base'] is not None else '-', f(r['ap'], r['ap_se']),
              f(r['auc'], r['auc_se']), f(r['mal_pct'], r['mal_pct_se'], 1)] for r in rows]
    w = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(hdr)]
    print()
    print('  '.join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    last = None
    for t in table:
        if last and t[0] != last:
            print()
        last = t[0]
        print('  '.join(c.ljust(w[i]) for i, c in enumerate(t)))

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            cw = csv.DictWriter(fh, fieldnames=list(rows[0]))
            cw.writeheader()
            cw.writerows(rows)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()
