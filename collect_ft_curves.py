'''
Collect per-epoch fine-tuning curves for the "AP per fine-tuning epoch" figure
(fig:pretrain_convergence): pretrained vs. randomly initialised static CyberGFM.

Reads the per-run CSVs ResultTracker writes (one row per evaluation) for every pretraining seed
and writes two files:

  ft_curves_runs.csv    one row per (dataset, init, seed, epoch) -- the raw traces
  ft_curves_summary.csv one row per (dataset, init, epoch): mean +- SE over seeds of te_ap,
                        te_auc, va_auc, va_ap, plus each run's best-validation epoch

Cells use the paper's walk-length rule: wl 4 on LANL and OpTC, wl 16 on UNSW. LANL and UNSW use
link-prediction fine-tuning (results/lp-static), OpTC the classifier (results/cls-static).
Epoch 0 (the pretrained model before any fine-tuning) is kept; random-init runs have no epoch 0.

Only finished runs (CSV ends with ResultTracker's "# BEST" line) are used unless --partial.
Missing or unfinished cells are listed at the end, with the command that queues them.

    python collect_ft_curves.py
    python collect_ft_curves.py --root results --out-dir figdata
    python collect_ft_curves.py --unsw-wl 4          # if the wl16 random-init runs do not exist
    python collect_ft_curves.py --select-by ap       # best epoch by va_ap instead of va_auc

Run from the repo root. Standard library only, so it is safe on the login node.
'''
from argparse import ArgumentParser
import csv
import math
import os
import sys

SEEDS = [1, 2, 3, 4, 5]
METRICS = ['te_ap', 'te_auc', 'va_auc', 'va_ap']


def cells(unsw_wl):
    # dataset, fine-tune kind, walk length
    return [('lanl14argus', 'lp', 4), ('optc', 'cls', 4), ('unsw', 'lp', unsw_wl)]


def run_path(root, ds, kind, wl, init, seed, size):
    pre = 'rand_init_' if init == 'random' else ''
    return os.path.join(root, f'{kind}-static', ds, f'{pre}{kind}_ps{seed}_{size}_wl{wl}_ps{seed}_s0.csv')


def read_run(path):
    '''Return (rows, finished). rows: list of dicts with epoch + metrics as floats.'''
    rows, finished = [], False
    with open(path) as f:
        lines = f.read().splitlines()
    data = [l for l in lines if l and not l.startswith('#')]
    finished = any(l.startswith('# BEST') for l in lines)
    if not data:
        return [], finished
    for r in csv.DictReader(data):
        row = {'epoch': int(r['epoch'])}
        for m in METRICS:
            v = r.get(m, '')
            row[m] = float(v) if v not in ('', None) else math.nan
        rows.append(row)
    return rows, finished


def best_epoch(rows, key):
    # same rule as ResultTracker: first epoch with the strictly highest validation score.
    # Epoch 0 (no fine-tuning) is a candidate there too, so it is here.
    best = None
    for r in rows:
        if best is None or r[key] > best[key]:
            best = r
    return best['epoch'] if best else None


def mean_se(xs):
    xs = [x for x in xs if not math.isnan(x)]
    n = len(xs)
    if n == 0:
        return math.nan, math.nan, 0
    mu = sum(xs) / n
    if n == 1:
        return mu, math.nan, 1
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    return mu, sd / math.sqrt(n), n


def main():
    ap = ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--size', default='tiny')
    ap.add_argument('--unsw-wl', type=int, default=16)
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--partial', action='store_true', help='also use runs without a "# BEST" line')
    ap.add_argument('--out-dir', default='.')
    args = ap.parse_args()
    key = f'va_{args.select_by}'

    os.makedirs(args.out_dir, exist_ok=True)
    runs_out = os.path.join(args.out_dir, 'ft_curves_runs.csv')
    summ_out = os.path.join(args.out_dir, 'ft_curves_summary.csv')

    traces = {}   # (ds, init) -> {seed: rows}
    missing = []  # (ds, kind, wl, init, seed, reason)
    with open(runs_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'init', 'seed', 'wl', 'epoch'] + METRICS + ['is_best_val'])
        for ds, kind, wl in cells(args.unsw_wl):
            for init in ('pretrained', 'random'):
                for s in SEEDS:
                    p = run_path(args.root, ds, kind, wl, init, s, args.size)
                    if not os.path.exists(p):
                        missing.append((ds, kind, wl, init, s, 'no file'))
                        continue
                    rows, finished = read_run(p)
                    if not finished and not args.partial:
                        missing.append((ds, kind, wl, init, s, f'unfinished ({len(rows)} evals)'))
                        continue
                    if not rows:
                        missing.append((ds, kind, wl, init, s, 'empty'))
                        continue
                    traces.setdefault((ds, init), {})[s] = rows
                    b = best_epoch(rows, key)
                    for r in rows:
                        w.writerow([ds, init, s, wl, r['epoch']] + [r[m] for m in METRICS]
                                   + [int(r['epoch'] == b)])

    with open(summ_out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['dataset', 'init', 'epoch', 'n']
                   + [f'{m}_{s}' for m in METRICS for s in ('mean', 'se')]
                   + ['best_epochs', 'best_epoch_mean', 'te_ap_at_best_mean', 'te_ap_at_best_se'])
        for (ds, init), by_seed in traces.items():
            bests = {s: best_epoch(rows, key) for s, rows in by_seed.items()}
            at_best = [next(r['te_ap'] for r in rows if r['epoch'] == bests[s]) for s, rows in by_seed.items()]
            bmu, bse, _ = mean_se(at_best)
            best_list = ' '.join(str(bests[s]) for s in sorted(bests))
            best_mean = sum(bests.values()) / len(bests)
            epochs = sorted({r['epoch'] for rows in by_seed.values() for r in rows})
            for e in epochs:
                vals = {m: [r[m] for rows in by_seed.values() for r in rows if r['epoch'] == e] for m in METRICS}
                stats = [mean_se(vals[m]) for m in METRICS]
                n = stats[0][2]
                w.writerow([ds, init, e, n] + [x for mu, se, _ in stats for x in (mu, se)]
                           + [best_list, best_mean, bmu, bse])

    # console summary: the numbers the figure shows
    print(f'best epoch selected by {key}\n')
    for (ds, init), by_seed in traces.items():
        epochs = sorted({r['epoch'] for rows in by_seed.values() for r in rows})
        aps = []
        for e in epochs:
            mu, _, _ = mean_se([r['te_ap'] for rows in by_seed.values() for r in rows if r['epoch'] == e])
            aps.append(f'{e}:{mu:.3f}')
        bests = [best_epoch(rows, key) for rows in by_seed.values()]
        print(f'{ds:12s} {init:10s} n={len(by_seed)}  best epochs {bests}  te_ap  ' + '  '.join(aps))

    print(f'\nwrote {runs_out}\nwrote {summ_out}')
    if missing:
        print(f'\n{len(missing)} run(s) skipped:')
        for ds, kind, wl, init, s, why in missing:
            print(f'  {kind}-static/{ds} wl{wl} {init:10s} ps{s}: {why}')
        need_rand = sorted({(ds, wl) for ds, _, wl, init, _, why in missing if init == 'random' and why == 'no file'})
        for ds, wl in need_rand:
            print(f'  -> queue with: WLS={wl} DATASETS={ds} ./run_all_hpc.sh --random')


if __name__ == '__main__':
    sys.exit(main())