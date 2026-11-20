'''
Aggregate fine-tuning results across seeds.

Reads results/<method>-<mode>/<dataset>/*_wl<W>_<ckpt>_s<seed>.csv (one CSV per run, one row per
epoch) and groups runs that differ only by seed. For each group it reports mean ± standard error
over seeds for two ways of picking a row from each run:

  final : the last epoch of a finished run
  best  : the epoch with the highest validation AUC (or --select-by ap)

A run counts as finished once its CSV has the "# BEST" summary line. Runs still in progress are
left out of the summary (the n column shows finished/total) unless --include-running is given,
in which case their latest epoch stands in for "final".

    python summarize_results.py
    python summarize_results.py --dataset optc-ts --csv optc_summary.csv
    python summarize_results.py --filter cls-static --curves curves/
'''
from argparse import ArgumentParser
from collections import defaultdict
import csv
import glob
import math
import os
import re

DEFAULT_SHOW = ('te_auc', 'te_ap', 'te_ap_lift', 'te_prec_r50', 'te_fp_day_r50', 'va_auc')
# <prefix>_wl<N>[_<variant>...][_s<seed>][.run<k>].csv
#
# Everything between the walk length and the seed is the "variant": zero or more tag segments,
# e.g. final, best, 2e9, 2e9_best. Runs are grouped by (prefix, walk length, variant), so any
# --tag you give lp_finetune.py (other than the seed) shows up as its own row.
#   lp_wl4_2e9_tiny_wl16_2e9_best_s0.csv  -> prefix lp_wl4_2e9_tiny, wl 16, variant 2e9_best, seed 0
#   lp_snap200Mtok_tiny_wl16_s2.csv       -> prefix lp_snap200Mtok_tiny, wl 16, variant -, seed 2
NAME_RE = re.compile(
    r'^(?P<prefix>.+)_wl(?P<wl>\d+)'
    r'(?P<variant>(?:_[A-Za-z0-9+-]+)*?)'
    r'(?:_s(?P<seed>\d+))?(?:\.run(?P<rerun>\d+))?\.csv$'
)


def read_run(path):
    '''Rows keyed by the CSV header, so files with and without newer metric columns both work.'''
    rows, finished, cols = [], False, None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('#'):
                finished = finished or line.startswith('# BEST')
                continue
            if line.startswith('epoch'):
                cols = line.split(',')
                continue
            parts = line.split(',')
            if cols is None or len(parts) != len(cols):
                continue  # partially written line
            try:
                row = {c: (int(v) if c in ('epoch', 'updates') else float(v) if v else float('nan'))
                       for c, v in zip(cols, parts)}
            except ValueError:
                continue
            rows.append(row)
    return rows, finished


def metric_names(groups):
    '''All metric columns seen, in first-seen order.'''
    names = []
    for runs in groups.values():
        for r in runs:
            for row in r['rows'][:1]:
                names += [k for k in row if k not in ('epoch', 'updates') and k not in names]
    return names


def mean_se(values):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    n = len(vals)
    if n == 0:
        return float('nan'), float('nan'), 0
    mu = sum(vals) / n
    if n == 1:
        return mu, float('nan'), 1
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var / n), n


def fmt(mu, se):
    if math.isnan(mu):
        return '-'
    return f'{mu:.4f}' if math.isnan(se) else f'{mu:.4f} ± {se:.4f}'


def collect(root, dataset=None, filt=None):
    groups = defaultdict(list)
    skipped = []
    for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.csv'))):
        exp_dir, ds = path.split(os.sep)[-3:-1]
        if dataset and ds != dataset:
            continue
        if filt and filt not in path:
            continue
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            skipped.append(path)
            continue
        key = (exp_dir, ds, m['prefix'], int(m['wl']), m['variant'].lstrip('_') or '-')
        rows, finished = read_run(path)
        groups[key].append(dict(seed=int(m['seed'] or 0), rows=rows, finished=finished, path=path))
    return groups, skipped


def pick(rows, how, select_key):
    if not rows:
        return None
    if how == 'final':
        return rows[-1]
    best = rows[0]
    for r in rows[1:]:
        if r[select_key] > best[select_key]:  # strict: earliest epoch wins ties, as in ResultTracker
            best = r
    return best


def summarize(groups, include_running, select_key, metrics):
    table = []
    for key in sorted(groups, key=lambda k: (k[0], k[1], k[2], k[4], k[3])):
        runs = groups[key]
        usable = [r for r in runs if r['rows'] and (r['finished'] or include_running)]
        row = dict(zip(('experiment', 'dataset', 'prefix', 'walk_len', 'variant'), key))
        row['n'] = f'{len(usable)}/{len(runs)}'
        row['epochs'] = '/'.join(sorted({str(r['rows'][-1]['epoch']) for r in usable})) or '-'
        for how in ('final', 'best'):
            picked = [pick(r['rows'], how, select_key) for r in usable]
            for m in metrics:
                mu, se, _ = mean_se([p.get(m) for p in picked])
                row[f'{how}_{m}'] = mu
                row[f'{how}_{m}_se'] = se
            if how == 'best':
                mu, se, _ = mean_se([p['epoch'] for p in picked])
                row['best_epoch'] = mu
        table.append(row)
    return table


def fmt_metric(name, mu, se):
    if 'fp_day' in name and not math.isnan(mu):  # counts: fewer decimals
        return f'{mu:.1f}' if math.isnan(se) else f'{mu:.1f} ± {se:.1f}'
    if 'lift' in name and not math.isnan(mu):
        return f'{mu:.1f}' if math.isnan(se) else f'{mu:.1f} ± {se:.1f}'
    return fmt(mu, se)


BASE_COLS = ['experiment', 'dataset', 'prefix', 'variant', 'walk_len', 'n', 'epochs', 'best_epoch']


def _row_cells(r, how, show):
    cells = [str(r[c]) for c in BASE_COLS[:-1]]
    be = r.get('best_epoch') if how == 'best' else None
    if be is None:
        cells.append('')
    elif isinstance(be, float) and math.isnan(be):
        cells.append('-')
    else:
        cells.append(f'{be:.1f}')
    cells += [fmt_metric(m, r.get(f'{how}_{m}', float('nan')), r.get(f'{how}_{m}_se', float('nan')))
              for m in show]
    return cells


def print_table(table, how, select_key, show):
    title = 'FINAL EPOCH' if how == 'final' else f'BEST EPOCH BY {select_key.upper()}'
    header = BASE_COLS + list(show)
    lines = [_row_cells(r, how, show) for r in table]

    # Size columns over both tables so FINAL and BEST line up with each other
    both = lines + [_row_cells(r, 'best' if how == 'final' else 'final', show) for r in table]
    widths = [max(len(h), *(len(l[i]) for l in both)) if both else len(h) for i, h in enumerate(header)]

    print(f'\n== {title} (mean ± standard error over seeds) ==')
    print('  '.join(h.ljust(w) for h, w in zip(header, widths)))
    prev = None
    for r, l in zip(table, lines):
        group = (r['experiment'], r['dataset'], r['prefix'], r['variant'])
        if prev is not None and group != prev:
            print()
        prev = group
        print('  '.join(c.ljust(w) for c, w in zip(l, widths)))


def write_csv(table, path, metrics):
    fields = ['experiment', 'dataset', 'prefix', 'variant', 'walk_len', 'n', 'epochs', 'best_epoch']
    for how in ('final', 'best'):
        for m in metrics:
            fields += [f'{how}_{m}', f'{how}_{m}_se']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        w.writerows(table)
    print(f'\nWrote {path}')


def write_curves(groups, out_dir, include_running, metrics):
    '''Per-epoch mean ± standard error for each group, over the runs that reached that epoch.'''
    os.makedirs(out_dir, exist_ok=True)
    for (exp_dir, ds, prefix, wl, ckpt), runs in groups.items():
        usable = [r for r in runs if r['rows'] and (r['finished'] or include_running)]
        if not usable:
            continue
        by_epoch = defaultdict(list)
        for r in usable:
            for row in r['rows']:
                by_epoch[row['epoch']].append(row)
        path = os.path.join(out_dir, f'{exp_dir}_{ds}_{prefix}_wl{wl}_{ckpt}.csv')
        with open(path, 'w') as f:
            f.write('epoch,updates,n,' + ','.join(f'{m},{m}_se' for m in metrics) + '\n')
            for e in sorted(by_epoch):
                rows = by_epoch[e]
                cells = [str(e), str(rows[0]['updates']), str(len(rows))]
                for m in metrics:
                    mu, se, _ = mean_se([x.get(m) for x in rows])
                    cells += ['' if math.isnan(mu) else f'{mu}', '' if math.isnan(se) else f'{se}']
                f.write(','.join(cells) + '\n')
    print(f'Wrote per-epoch curves to {out_dir}/')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--dataset', help='only this dataset, e.g. optc-ts')
    ap.add_argument('--filter', help='only files whose path contains this string, e.g. cls-temporal')
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc', help='validation metric for "best"')
    ap.add_argument('--show', choices=['final', 'best', 'both'], default='both')
    ap.add_argument('--include-running', action='store_true',
                    help='include unfinished runs, using their latest epoch as "final"')
    ap.add_argument('--metrics', nargs='+', help=f'columns to print (default: {" ".join(DEFAULT_SHOW)}); '
                                                 'the CSV always gets every metric')
    ap.add_argument('--csv', help='also write the summary table to this CSV')
    ap.add_argument('--curves', help='also write per-epoch mean ± SE curves to this directory')
    args = ap.parse_args()

    select_key = f'va_{args.select_by}'
    groups, skipped = collect(args.root, args.dataset, args.filter)
    if skipped:
        print(f'Skipped {len(skipped)} file(s) whose names do not fit '
              f'<prefix>_wl<N>[_<variant>...][_s<seed>].csv:')
        for path in skipped[:5]:
            print(f'  {path}')
        if len(skipped) > 5:
            print(f'  ... and {len(skipped) - 5} more')
    if not groups:
        raise SystemExit(f'No result CSVs found under {args.root}/')

    metrics = metric_names(groups)
    show = [m for m in (args.metrics or DEFAULT_SHOW) if m in metrics]
    table = summarize(groups, args.include_running, select_key, metrics)
    for how in (('final', 'best') if args.show == 'both' else (args.show,)):
        print_table(table, how, select_key, show)
    if args.csv:
        write_csv(table, args.csv, metrics)
    if args.curves:
        write_curves(groups, args.curves, args.include_running, metrics)