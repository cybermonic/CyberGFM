'''
Main-table results, averaged over PRETRAINING seeds.

show_results.py groups runs that differ only by fine-tuning seed, so every _ps<N> checkpoint lands
in its own row and the five replicates never meet. This collapses them: one number per pretraining
run, then mean +- standard error across runs. That is the error bar the new protocol is for --
between-pretraining-run AP spread is 25-40% of the mean, so a single checkpoint cannot stand
behind a claim.

Reduction, inside out:
  1. per RUN        : final epoch, or best epoch by validation AUC (--how)
  2. per CHECKPOINT : mean over fine-tuning seeds of that pretraining run
  3. per CELL       : mean +- SE over pretraining runs

Step 2 only matters when FT_SEEDS has more than one entry; with one it is a no-op, and the script
says which it was. Counting 5 checkpoints x 3 fine-tuning seeds as n=15 would claim replicates
the experiment does not have.

    python show_main_results.py                      # every cell, every walk length
    python show_main_results.py --table              # the paper-shaped view, matched walk length
    python show_main_results.py --wl 4 --metrics te_auc te_ap
    python show_main_results.py --per-seed           # every checkpoint separately
    python show_main_results.py --progress           # what is still running
    python show_main_results.py --csv main.csv
    python show_main_results.py --no-rand-init       # leave out the from-random-init controls
    python show_main_results.py --root results-ntokens --table   # the pretraining-length sweep
    python show_main_results.py --dataset lanl14argus --wl 4 --filter lp-static   # sizes: tiny vs
                                                     # lp-static_mini / _med / _baseline rows

Pretraining variants named in the file (the token budget from run_ntokens.sh, e.g. lp_1e8_ps1_tiny...,
or a poisoning level) become their own experiment, `<experiment>_<variant>` (lp-static_1e8), so
different budgets are never averaged together. The main runs carry no variant and are unchanged.

Runs fine-tuned from random initialisation (lp_finetune.py --from-random, files named rand_init_*)
are their own experiment, `<experiment>-rand` (e.g. lp-static-rand), so they sit next to the
pretrained rows rather than being averaged into them.

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

DEFAULT_SHOW = ('te_auc', 'te_ap', 'te_prec_r50', 'te_fp_day_r50')
NAME_RE = re.compile(r'^(?P<prefix>.+?)_wl(?P<wl>\d+)(?P<tag>.*)\.csv$')
T_CRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23}

# The walk length each dataset is PRETRAINED at, from pretrain.defaults(): LANL is 4 with edge
# features, everything else 64. Fine-tuning at the pretraining length is the selection rule, so
# mark that row and use it for --table. Keep in step with pretrain.py.
PT_WL = {'lanl': 4}
DEFAULT_PT_WL = 64


def pt_wl(dataset):
    for prefix, wl in PT_WL.items():
        if dataset.startswith(prefix):
            return wl
    return DEFAULT_PT_WL


RAND_PREFIX = 'rand_init_'   # what lp_finetune.py / cls_finetune.py prepend under --from-random


def variant_of(prefix, ps_re):
    '''
    What distinguishes this run's pretraining beyond its seed: the part of the file prefix left
    after removing rand_init_, the lp/cls kind, the pretraining seed and the default model size
    (tiny). Any other size stays, so each model size is its own cell.

        lp_ps1_tiny          -> ''        (the main runs)
        lp_ps1_mini          -> '_mini'   (run_sizes.sh)
        lp_1e8_ps1_tiny      -> '_1e8'    (run_ntokens.sh)
        lp_10poisoned_ps3_tiny -> '_10poisoned'

    A non-empty variant is appended to the experiment name, so runs pretrained differently are
    separate cells instead of being averaged together as if they were extra seeds.
    '''
    v = prefix[len(RAND_PREFIX):] if prefix.startswith(RAND_PREFIX) else prefix
    v = re.sub(r'^(lp|cls)', '', v)
    v = ps_re.sub('', v, count=1)
    v = re.sub(r'_tiny$', '', v)
    return v


def nat(s):
    '''Natural sort key: lp-static_3e8 before lp-static_11e8.'''
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', str(s))]


def collect(root, ps_re, filt=None, dataset=None, rand_init=True):
    '''(experiment, dataset, wl, pseed) -> list of runs, one per fine-tuning seed.'''
    runs, skipped = defaultdict(list), []
    for path in sorted(glob.glob(os.path.join(root, '*', '*', '*.csv'))):
        exp, ds = path.split(os.sep)[-3:-1]
        if (dataset and ds != dataset) or (filt and filt not in path):
            continue
        # A from-random run shares every other part of its name with the pretrained run of the
        # same seed, so without this it would be averaged in as another fine-tuning seed.
        if os.path.basename(path).startswith(RAND_PREFIX):
            if not rand_init:
                continue
            exp = f'{exp}-rand'
        m = NAME_RE.match(os.path.basename(path))
        ps = ps_re.search(os.path.basename(path)) if m else None
        if not ps:
            skipped.append(path)   # not part of the pretraining-seed matrix
            continue
        exp += variant_of(m['prefix'], ps_re)
        seed = re.search(r'_s(\d+)$', m['tag'])
        rows, finished = read_run(path)
        runs[(exp, ds, int(m['wl']), int(ps[1]))].append(
            dict(seed=int(seed[1]) if seed else 0, rows=rows, finished=finished, path=path))
    return runs, skipped


def pick_row(rows, how, select_key):
    if not rows:
        return None
    if how == 'final' or select_key not in rows[0]:
        return rows[-1]
    best = rows[0]
    for r in rows[1:]:
        if r[select_key] > best[select_key]:   # strict: earliest epoch wins ties, as ResultTracker
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
    '''Steps 1 and 2.'''
    out = {}
    for key, rs in runs.items():
        usable = [r for r in rs if r['rows'] and (r['finished'] or include_running)]
        if not usable:
            continue
        picked = [pick_row(r['rows'], how, select_key) for r in usable]
        vals = {m: mean_se([p.get(m) for p in picked])[0] for m in metrics}
        vals['_n_ft'] = len(usable)
        vals['_epoch'] = mean_se([float(p['epoch']) for p in picked])[0]
        out[key] = vals
    return out


def aggregate(cells, metrics):
    '''Step 3: (experiment, dataset, wl) -> {metric: (mean, se, n)}.'''
    out = {}
    for exp, ds, wl, _ in cells:
        key = (exp, ds, wl)
        if key in out:
            continue
        vals = [v for (e, d, w, _), v in cells.items() if (e, d, w) == key]
        row = {'n': len(vals)}
        for m in metrics:
            row[m] = mean_se([v[m] for v in vals])
        row['_epoch'] = mean_se([v['_epoch'] for v in vals])
        out[key] = row
    return out


def fmt(name, mu, se=None):
    if mu is None or (isinstance(mu, float) and math.isnan(mu)):
        return '-'
    d = 1 if ('fp_day' in name or 'lift' in name) else 4
    s = f'{mu:.{d}f}'
    if se is not None and not math.isnan(se):
        s += f' ± {se:.{d}f}'
    return s


def print_table(agg, show, how, select_key, expected_n):
    label = 'final epoch' if how == 'final' else f'best epoch by {select_key}'
    hdr = ['experiment', 'dataset', 'wl', 'n', 'epoch'] + list(show)
    lines = []
    for (exp, ds, wl) in sorted(agg, key=lambda k: (k[1], nat(k[0]), k[2])):
        r = agg[(exp, ds, wl)]
        star = '*' if wl == pt_wl(ds) else ''
        lines.append([exp, ds, f'{wl}{star}', f'{r["n"]}/{expected_n}',
                      fmt('epoch', r['_epoch'][0])] +
                     [fmt(m, *r[m][:2]) for m in show])
    if not lines:
        print('no cells to show')
        return
    w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
    print(f'\n== {label} | mean ± SE over pretraining seeds ==')
    print('  '.join(h.ljust(x) for h, x in zip(hdr, w)))
    prev = None
    for (exp, ds, wl), l in zip(sorted(agg, key=lambda k: (k[1], nat(k[0]), k[2])), lines):
        if prev is not None and (ds, exp) != prev:
            print()
        prev = (ds, exp)
        print('  '.join(c.ljust(x) for c, x in zip(l, w)))
    print('  * = walk length matching pretraining (the selection rule; see the plan doc)')
    ns = sorted({r['n'] for r in agg.values() if r['n'] > 2})
    if ns:
        crits = ', '.join(f'n={n}: {T_CRIT.get(n - 1, 1.96):.2f}' for n in ns)
        print(f'  a difference is significant at p<0.05 two-sided when |d|/SE exceeds  {crits}')


def print_paper_table(agg, how, select_key, expected_n):
    '''Table 5 shape: one row per model variant, AUC and AP, at the matched walk length.'''
    label = 'final epoch' if how == 'final' else f'best epoch by {select_key}'
    print(f'\n== paper table | matched walk length | {label} | mean ± SE over pretraining seeds ==')
    datasets = sorted({ds for _, ds, _ in agg})
    for ds in datasets:
        wl = pt_wl(ds)
        rows = []
        for exp in sorted({e for e, d, _ in agg if d == ds}, key=nat):
            r = agg.get((exp, ds, wl))
            if r is None:
                rows.append([exp, f'wl{wl}', '-', '-', f'0/{expected_n}'])
                continue
            variant = 'CyberGFM_T' if 'temporal' in exp else 'CyberGFM_S'
            if '-gpt-' in exp:   # lp-gpt-static, cls-gpt-temporal, ... (run_all_hpc.sh --gpt)
                variant += ', GPT'
            if exp.endswith('-rand'):
                variant += ', random init'
            rows.append([f'{variant} ({exp})', f'wl{wl}',
                         fmt('te_auc', *r.get('te_auc', (float('nan'),) * 2)[:2]),
                         fmt('te_ap', *r.get('te_ap', (float('nan'),) * 2)[:2]),
                         f'{r["n"]}/{expected_n}'])
        if not rows:
            continue
        hdr = [ds, 'wl', 'AUC', 'AP', 'n']
        w = [max(len(h), *(len(x[i]) for x in rows)) for i, h in enumerate(hdr)]
        print()
        print('  '.join(h.ljust(x) for h, x in zip(hdr, w)))
        for x in rows:
            print('  '.join(c.ljust(y) for c, y in zip(x, w)))


def print_progress(runs, cells, expected_n):
    '''Per cell: finished / started / expected, so an in-flight sweep is legible.'''
    print('\n== progress ==')
    keys = sorted({(e, d, w) for e, d, w, _ in runs} | {(e, d, w) for e, d, w, _ in cells},
                  key=lambda k: (k[1], nat(k[0]), k[2]))
    hdr = ['experiment', 'dataset', 'wl', 'finished', 'started', 'seeds done']
    lines = []
    for exp, ds, wl in keys:
        ps_all = sorted({p for e, d, w, p in runs if (e, d, w) == (exp, ds, wl)})
        ps_ok = sorted({p for e, d, w, p in cells if (e, d, w) == (exp, ds, wl)})
        lines.append([exp, ds, str(wl), f'{len(ps_ok)}/{expected_n}', str(len(ps_all)),
                      ','.join(f'ps{p}' for p in ps_ok) or '-'])
    w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
    print('  '.join(h.ljust(x) for h, x in zip(hdr, w)))
    for l in lines:
        print('  '.join(c.ljust(x) for c, x in zip(l, w)))


def print_per_seed(cells, show):
    print('\n== per pretraining seed ==')
    keys = sorted(cells, key=lambda k: (k[1], nat(k[0]), k[2], k[3]))
    hdr = ['experiment', 'dataset', 'wl', 'ps', 'ft seeds'] + list(show)
    lines = [[e, d, str(w), f'ps{p}', str(cells[(e, d, w, p)]['_n_ft'])] +
             [fmt(m, cells[(e, d, w, p)].get(m)) for m in show] for e, d, w, p in keys]
    wid = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
    print('  '.join(h.ljust(x) for h, x in zip(hdr, wid)))
    for l in lines:
        print('  '.join(c.ljust(x) for c, x in zip(l, wid)))


def write_csv(path, agg, metrics, expected_n):
    fields = ['experiment', 'dataset', 'walk_len', 'n', 'n_expected', 'epoch']
    for m in metrics:
        fields += [m, f'{m}_se']
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for (exp, ds, wl) in sorted(agg, key=lambda k: (k[1], nat(k[0]), k[2])):
            r = agg[(exp, ds, wl)]
            row = dict(experiment=exp, dataset=ds, walk_len=wl, n=r['n'],
                       n_expected=expected_n, epoch=r['_epoch'][0])
            for m in metrics:
                row[m], row[f'{m}_se'] = r[m][0], r[m][1]
            w.writerow(row)
    print(f'\nWrote {path}')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--root', default='results')
    ap.add_argument('--dataset', help='only this dataset, e.g. lanl14argus')
    ap.add_argument('--filter', help='only paths containing this, e.g. lp-static')
    ap.add_argument('--ps-pattern', default=r'_ps(\d+)',
                    help='how the pretraining seed appears in the filename (default: _ps<N>)')
    ap.add_argument('--how', choices=['final', 'best'], default='best')
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--metrics', nargs='+', help=f'default: {" ".join(DEFAULT_SHOW)}')
    ap.add_argument('--wl', type=int, nargs='+', help='only these walk lengths')
    ap.add_argument('--expected-seeds', type=int, default=5, help='for the n column (default 5)')
    ap.add_argument('--table', action='store_true', help='also print the paper-shaped table')
    ap.add_argument('--per-seed', action='store_true', help='also print every checkpoint')
    ap.add_argument('--progress', action='store_true', help='print cell progress and stop')
    ap.add_argument('--include-running', action='store_true',
                    help='count unfinished runs, using their latest epoch')
    ap.add_argument('--csv', help='also write the aggregate to this CSV')
    ap.add_argument('--no-rand-init', action='store_true',
                    help='leave out the from-random-init runs (rand_init_* files)')
    args = ap.parse_args()

    select_key = f'va_{args.select_by}'
    runs, skipped = collect(args.root, re.compile(args.ps_pattern), args.filter, args.dataset,
                            rand_init=not args.no_rand_init)
    if args.wl:
        runs = {k: v for k, v in runs.items() if k[2] in args.wl}
    if not runs:
        raise SystemExit(f'No results with {args.ps_pattern} in the name under {args.root}/. '
                         f'(show_results.py covers runs outside the pretraining-seed matrix.)')
    if skipped:
        print(f'Ignored {len(skipped)} CSV(s) with no pretraining seed in the name '
              f'(runs outside this matrix)')

    metrics_avail = metric_names(runs)
    show = [m for m in (args.metrics or DEFAULT_SHOW) if m in metrics_avail]
    if not show:
        raise SystemExit(f'None of those metrics are present. Available: {" ".join(metrics_avail)}')

    cells = per_checkpoint(runs, args.how, select_key, metrics_avail, args.include_running)
    if args.progress:
        print_progress(runs, cells, args.expected_seeds)
        raise SystemExit(0)
    if not cells:
        raise SystemExit('No finished runs yet. --include-running to use partial ones, '
                         'or --progress to see what is in flight.')

    n_ft = {v['_n_ft'] for v in cells.values()}
    if n_ft and max(n_ft) > 1:
        print(f'fine-tuning seeds per checkpoint: {min(n_ft)}-{max(n_ft)}, averaged within each '
              f'checkpoint before the pretraining-seed mean')

    agg = aggregate(cells, metrics_avail)
    print_table(agg, show, args.how, select_key, args.expected_seeds)
    if args.table:
        print_paper_table(agg, args.how, select_key, args.expected_seeds)
    if args.per_seed:
        print_per_seed(cells, show)
    if args.csv:
        write_csv(args.csv, agg, metrics_avail, args.expected_seeds)

    short = {k: v['n'] for k, v in agg.items() if v['n'] < args.expected_seeds}
    if short:
        print(f'\n{len(short)} cell(s) have fewer than {args.expected_seeds} pretraining seeds; '
              f'--progress shows which, and the SEs on those rows are provisional.')