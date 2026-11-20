'''
Dirty-vs-clean comparison for the poisoning experiment, aggregated over PRETRAINING seeds.

show_results.py groups runs that differ only by fine-tuning seed, so each _ps<seed> checkpoint
gets its own row and the arms sit far apart in the table. Here the pretraining seed is the unit of
replication -- poisoning acts at pretraining, so the error bars have to span pretraining runs --
and the two arms are put side by side with their difference.

Reduction, from the inside out:
  1. one number per RUN      : the final epoch, or the best epoch by validation AUC (--how)
  2. one number per CHECKPOINT: mean over fine-tuning seeds of that pretraining run
  3. one number per ARM      : mean +- standard error over pretraining runs

So the standard error is between pretraining runs, with fine-tuning noise averaged out inside each
one, which is the variance the dirty/clean manipulation actually acts through. Step 2 matters as
soon as FT_SEEDS has more than one seed in it: treating 4 checkpoints x 3 fine-tuning seeds as
n=12 would claim replicates the experiment does not have.

The difference column is reported two ways. "delta" is unpaired. "paired" differences the arms
within each pretraining seed index first, which is legitimate here -- seed_everything(s) gives
both arms the same initialisation and the same sampling stream, so pairing removes that shared
variance -- and is usually the tighter of the two. |d|/SE is given instead of a p-value: with only
a handful of pretraining runs a p-value reads as more than it is. The critical value for the
n you have is printed beneath each table.

Cells that are missing or unfinished are listed first, as a seed x walk-length grid per arm, since
an arm silently short a few checkpoints is the easy way to misread this comparison. Use
fill_missing.sh to fill them.

    python show_poison_results.py
    python show_poison_results.py --metrics te_ap te_auc --how final
    python show_poison_results.py --per-seed             # every checkpoint, no aggregation
    python show_poison_results.py --coverage-only        # just what is missing
    python show_poison_results.py --csv poison_summary.csv

Run from the repo root.
'''
from argparse import ArgumentParser
from collections import defaultdict
import csv
import glob
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from show_results import read_run, mean_se  # same reader, same "# BEST" finished rule

DEFAULT_SHOW = ('te_ap', 'te_auc', 'te_prec_r50', 'te_fp_day_r50')

# <prefix>_wl<N><tag>.csv ; the pretraining seed is read out of the tag segments, not positionally,
# because it appears both as --pretrain-tag (inside prefix) and as part of --tag.
NAME_RE = re.compile(r'^(?P<prefix>.+?)_wl(?P<wl>\d+)(?P<tag>.*)\.csv$')
# two-sided 0.05 critical values, df 1..10; beyond that the normal approximation is close enough
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}


def t_crit(n):
    '''Critical |mean| / SE for n observations, two-sided alpha = 0.05.'''
    return T_CRIT.get(n - 1, 1.96) if n > 1 else float('nan')


def arm_of(dataset, dirty, clean):
    if dirty in dataset:
        return 'dirty'
    if clean in dataset:
        return 'clean'
    return None


def collect(root, dirty, clean, ps_re, filt=None):
    '''(experiment, wl, arm, pseed) -> list of runs, one per fine-tuning seed.'''
    runs = defaultdict(list)
    skipped = []
    for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.csv'))):
        exp, ds = path.split(os.sep)[-3:-1]
        arm = arm_of(ds, dirty, clean)
        if arm is None or (filt and filt not in path):
            continue
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            skipped.append(path)
            continue
        ps = ps_re.search(os.path.basename(path))
        if not ps:
            skipped.append(path)  # not part of the pretraining-seed matrix
            continue
        seed = re.search(r'_s(\d+)$', m['tag'])
        rows, finished = read_run(path)
        runs[(exp, int(m['wl']), arm, int(ps[1]))].append(
            dict(seed=int(seed[1]) if seed else 0, rows=rows, finished=finished,
                 path=path, dataset=ds))
    return runs, skipped


def pick_row(rows, how, select_key):
    '''show_results.pick, but tolerant of a run whose CSV predates the selection metric.'''
    if not rows:
        return None
    if how == 'final' or select_key not in rows[0]:
        return rows[-1]
    best = rows[0]
    for r in rows[1:]:
        if r[select_key] > best[select_key]:  # strict: earliest epoch wins ties, as ResultTracker does
            best = r
    return best


def metric_names(runs):
    names = []
    for rs in runs.values():
        for r in rs:
            for row in r['rows'][:1]:
                names += [k for k in row if k not in ('epoch', 'updates') and k not in names]
    return names


def per_checkpoint(runs, how, select_key, metrics, include_running):
    '''Step 1 and 2: (experiment, wl, arm, pseed) -> {metric: mean over fine-tuning seeds}.'''
    out = {}
    for key, rs in runs.items():
        usable = [r for r in rs if r['rows'] and (r['finished'] or include_running)]
        if not usable:
            continue
        picked = [pick_row(r['rows'], how, select_key) for r in usable]
        vals = {m: mean_se([p.get(m) for p in picked])[0] for m in metrics}
        vals['_n_ft'] = len(usable)
        vals['_epoch'] = mean_se([p['epoch'] for p in picked])[0]
        out[key] = vals
    return out


def coverage(runs, cells, wls, pseeds, include_running):
    '''Per (experiment, arm) grid: ok / partial / missing, plus the list of what to rerun.'''
    report, gaps = [], []
    for exp in sorted({k[0] for k in runs} | {k[0] for k in cells}):
        for arm in ('dirty', 'clean'):
            grid = {}
            for ps in pseeds:
                for wl in wls:
                    rs = runs.get((exp, wl, arm, ps), [])
                    if (exp, wl, arm, ps) in cells:
                        grid[(ps, wl)] = 'ok'
                    elif rs:
                        grid[(ps, wl)] = 'part'
                        gaps.append((exp, arm, ps, wl, 'unfinished'))
                    else:
                        grid[(ps, wl)] = '.'
                        gaps.append((exp, arm, ps, wl, 'missing'))
            report.append((exp, arm, grid))
    return report, gaps


def print_coverage(report, wls, pseeds, include_running):
    missing = [g for exp, arm, grid in report for g in grid.values() if g != 'ok']
    if not missing:
        print(f'coverage: complete -- {len(pseeds)} pretraining seeds x {len(wls)} walk lengths, '
              f'both arms, every experiment')
        return
    print('== coverage (ok = finished, part = CSV exists but no "# BEST" line, . = no CSV) ==')
    if include_running:
        print('   --include-running is on, so "part" runs are still counted in the tables below')
    w = max(4, max(len(str(wl)) for wl in wls))
    for exp, arm, grid in report:
        n_ok = sum(1 for v in grid.values() if v == 'ok')
        print(f'\n  {exp} / {arm}   {n_ok}/{len(grid)} finished')
        print('      ps  ' + '  '.join(f'wl{wl}'.ljust(w) for wl in wls))
        for ps in pseeds:
            print(f'      {ps:<3} ' + '  '.join(grid[(ps, wl)].ljust(w) for wl in wls))
    print('\n  to fill the gaps:  ./fill_missing.sh --list   then   ./fill_missing.sh')


def fmt_val(name, mu, se=None, n=None):
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        return '-'
    d = 1 if ('fp_day' in name or 'lift' in name) else 4
    s = f'{mu:.{d}f}'
    if se is not None and not math.isnan(se):
        s += f' ± {se:.{d}f}'
    if n is not None:
        s += f' ({n})'
    return s


def compare(cells, metric):
    '''Step 3: (experiment, wl) -> dirty/clean means over pretraining runs, and the difference.'''
    out = {}
    keys = sorted({(e, wl) for e, wl, _, _ in cells}, key=lambda k: (k[0], k[1]))
    for exp, wl in keys:
        by_arm = {}
        for arm in ('dirty', 'clean'):
            vals = {ps: cells[(exp, wl, arm, ps)][metric]
                    for (e, w, a, ps) in cells if (e, w, a) == (exp, wl, arm)}
            vals = {ps: v for ps, v in vals.items() if v is not None and not math.isnan(v)}
            by_arm[arm] = vals
        d, c = by_arm['dirty'], by_arm['clean']
        row = dict(experiment=exp, walk_len=wl)
        for arm, vals in (('dirty', d), ('clean', c)):
            mu, se, n = mean_se(list(vals.values()))
            row[f'{arm}'], row[f'{arm}_se'], row[f'{arm}_n'] = mu, se, n
        # unpaired: difference of means, standard errors added in quadrature
        if d and c:
            row['delta'] = row['dirty'] - row['clean']
            se_d, se_c = row['dirty_se'], row['clean_se']
            row['delta_se'] = math.sqrt(sum(s ** 2 for s in (se_d, se_c) if not math.isnan(s))) \
                if not (math.isnan(se_d) and math.isnan(se_c)) else float('nan')
        # paired on the seed index: same init and sampling stream in both arms
        shared = sorted(set(d) & set(c))
        if len(shared) >= 1:
            diffs = [d[ps] - c[ps] for ps in shared]
            mu, se, n = mean_se(diffs)
            row['paired'], row['paired_se'], row['paired_n'] = mu, se, n
        out[(exp, wl)] = row
    return out


def print_table(rows, metric, how, select_key, per_seed_cells=None, wls=None):
    label = 'final epoch' if how == 'final' else f'best epoch by {select_key}'
    hdr = ['experiment', 'wl', 'dirty (n)', 'clean (n)', 'delta (dirty-clean)', '|d|/SE', 'paired delta', '|d|/SE']
    lines = []
    ns = set()
    for (exp, wl), r in sorted(rows.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        def ratio(mu, se, n):
            # With n=2 the SE of the difference is |d1 - d2| / 2, which lands near zero whenever the
            # two runs happen to agree, sending |d|/SE to absurd values that say nothing about the
            # effect. Below n=3 the ratio is not reported at all rather than reported misleadingly.
            if n is None or n < 3:
                return 'n<3'
            if mu is None or se is None or math.isnan(mu) or math.isnan(se) or se == 0:
                return '-'
            return f'{abs(mu) / se:.1f}'
        ns.update(x for x in (r.get('dirty_n'), r.get('clean_n'), r.get('paired_n')) if x)
        lines.append([
            exp, str(wl),
            fmt_val(metric, r['dirty'], r['dirty_se'], r['dirty_n']),
            fmt_val(metric, r['clean'], r['clean_se'], r['clean_n']),
            fmt_val(metric, r.get('delta'), r.get('delta_se')),
            # the unpaired comparison is only as replicated as its smaller arm
            ratio(r.get('delta'), r.get('delta_se'), min(r.get('dirty_n') or 0, r.get('clean_n') or 0)),
            fmt_val(metric, r.get('paired'), r.get('paired_se'), r.get('paired_n')),
            ratio(r.get('paired'), r.get('paired_se'), r.get('paired_n')),
        ])
    widths = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)] if lines else []
    print(f'\n== {metric} | {label} | mean ± SE over PRETRAINING seeds ==')
    print('  '.join(h.ljust(w) for h, w in zip(hdr, widths)))
    prev = None
    for (exp, wl), l in zip(sorted(rows, key=lambda k: (k[0], k[1])), lines):
        if prev is not None and exp != prev:
            print()
        prev = exp
        print('  '.join(c.ljust(w) for c, w in zip(l, widths)))
    crits = ', '.join(f'n={n}: {t_crit(n):.2f}' for n in sorted(ns) if n > 2)
    if crits:
        print(f'  |d|/SE is significant at p<0.05 two-sided above  {crits}   (t, df = n-1; the')
        print(f'  unpaired column is scored on the smaller of the two arm n)')
    if any(n == 2 for n in ns):
        print('  "n<3": only two runs, so the SE of the difference is one number and the ratio '
              'would be noise')

    if per_seed_cells is not None:
        print(f'\n  per pretraining seed ({metric}, {label}):')
        for exp in sorted({k[0] for k in per_seed_cells}):
            pseeds = sorted({k[3] for k in per_seed_cells if k[0] == exp})
            w = max(8, *(len(f'wl{wl}') for wl in wls))
            print(f'    {exp}')
            print('      arm    ps  ' + '  '.join(f'wl{wl}'.ljust(w) for wl in wls))
            for arm in ('dirty', 'clean'):
                for ps in pseeds:
                    cells = [per_seed_cells.get((exp, wl, arm, ps), {}).get(metric) for wl in wls]
                    if all(v is None for v in cells):
                        continue
                    print(f'      {arm:<6} {ps:<3} ' +
                          '  '.join(fmt_val(metric, v).ljust(w) for v in cells))


def write_csv(path, tables, metrics):
    fields = ['metric', 'experiment', 'walk_len', 'dirty', 'dirty_se', 'dirty_n',
              'clean', 'clean_se', 'clean_n', 'delta', 'delta_se', 'paired', 'paired_se', 'paired_n']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for m in metrics:
            for key in sorted(tables[m], key=lambda k: (k[0], k[1])):
                w.writerow(dict(tables[m][key], metric=m))
    print(f'\nWrote {path}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--dirty', default='dirtyts', help='substring identifying the dirty dataset')
    ap.add_argument('--clean', default='cleants', help='substring identifying the clean dataset')
    ap.add_argument('--filter', help='only files whose path contains this string, e.g. lp-static')
    ap.add_argument('--ps-pattern', default=r'_ps(\d+)',
                    help='how the pretraining seed appears in the filename (default: _ps<N>)')
    ap.add_argument('--how', choices=['final', 'best'], default='best',
                    help='which epoch of each run to use (default: best, by validation --select-by)')
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--metrics', nargs='+', help=f'default: {" ".join(DEFAULT_SHOW)}')
    ap.add_argument('--wl', type=int, nargs='+', help='only these walk lengths')
    ap.add_argument('--pseeds', type=int, nargs='+', help='only these pretraining seeds')
    ap.add_argument('--per-seed', action='store_true', help='also print every checkpoint separately')
    ap.add_argument('--include-running', action='store_true',
                    help='count unfinished runs, using their latest epoch as "final"')
    ap.add_argument('--coverage-only', action='store_true', help='report what is missing and stop')
    ap.add_argument('--csv', help='also write the comparison to this CSV')
    args = ap.parse_args()

    select_key = f'va_{args.select_by}'
    runs, skipped = collect(args.root, args.dirty, args.clean, re.compile(args.ps_pattern), args.filter)
    if args.wl:
        runs = {k: v for k, v in runs.items() if k[1] in args.wl}
    if args.pseeds:
        runs = {k: v for k, v in runs.items() if k[3] in args.pseeds}
    if not runs:
        raise SystemExit(f'No matching result CSVs under {args.root}/ '
                         f'(looking for {args.dirty}/{args.clean} datasets with {args.ps_pattern})')
    if skipped:
        print(f'Ignored {len(skipped)} CSV(s) with no pretraining seed in the name '
              f'(runs outside this matrix)\n')

    metrics_avail = metric_names(runs)
    metrics = [m for m in (args.metrics or DEFAULT_SHOW) if m in metrics_avail]
    if not metrics:
        raise SystemExit(f'None of the requested metrics are in these CSVs. Available: '
                         f'{" ".join(metrics_avail)}')

    cells = per_checkpoint(runs, args.how, select_key, metrics_avail, args.include_running)
    wls = sorted({k[1] for k in runs})
    pseeds = sorted({k[3] for k in runs})
    report, gaps = coverage(runs, cells, wls, pseeds, args.include_running)
    print_coverage(report, wls, pseeds, args.include_running)
    if args.coverage_only:
        raise SystemExit(0)

    n_ft = {v['_n_ft'] for v in cells.values()}
    if n_ft and max(n_ft) > 1:
        print(f'\nfine-tuning seeds per checkpoint: {min(n_ft)}-{max(n_ft)}, averaged within each '
              f'checkpoint before the arm means')

    tables = {m: compare(cells, m) for m in metrics}
    for m in metrics:
        print_table(tables[m], m, args.how, select_key,
                    per_seed_cells=cells if args.per_seed else None, wls=wls)

    if args.csv:
        write_csv(args.csv, tables, metrics)

    print('\nNote: the walk length with the largest gap is not the walk length to report. Picking it '
          '\nfrom this table is selection on the test set; choose it on validation (or report the '
          '\nsweep) and let these numbers say whether the arms separate at all.')