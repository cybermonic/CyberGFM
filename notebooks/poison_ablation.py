'''
Effect of training-window poisoning on detection quality, as a function of walk length.

Run from notebooks/. Reads the fine-tuning CSVs that lp_finetune.py writes under

    ../results/<method>-<mode>/<dataset>/<prefix>_wl<W>[_<variant>][_s<seed>].csv

groups runs that differ only by seed, and plots two panels sharing the walk-length axis:

  top     the metric itself, dirty (solid) against clean (dashed)
  bottom  dirty - clean, the poisoning effect, with a zero reference line

A point below zero in the bottom panel means training on unlabelled attack traffic HURT. The
claim the figure is meant to support is that this is large for GNN baselines and near zero for
CyberGFM, so the bottom panel is the one that carries the argument.

Argus has no walk length, so with --argus-dir its dirty/clean pair is drawn as horizontal
reference bands (mean +/- sem) rather than a line.

    python plot_poison_by_wl.py
    python plot_poison_by_wl.py --metric te_ap_lift --select final
    python plot_poison_by_wl.py --filter lp-static --out ../figs/poison_static.pdf
'''
from argparse import ArgumentParser
from collections import defaultdict
import csv
import glob
import math
import os
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Same convention as show_results.py: <prefix>_wl<N>[_<variant>...][_s<seed>][.run<k>].csv
NAME_RE = re.compile(
    r'^(?P<prefix>.+)_wl(?P<wl>\d+)'
    r'(?P<variant>(?:_[A-Za-z0-9+-]+)*?)'
    r'(?:_s(?P<seed>\d+))?(?:\.run(?P<rerun>\d+))?\.csv$'
)

# Validated categorical order (dataviz skill, references/palette.md slots 1,2,3,7). Passes all
# six checks with --pairs all: worst CVD dE 9.2, worst normal-vision dE 16.3. Colour identifies
# the METHOD and stays with it across both panels; linestyle carries dirty vs clean.
PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#4a3aa7']
INK, INK_MUTED, GRID = '#1a1a19', '#6b6a63', '#d9d8d2'

LOWER_IS_BETTER = ('fp_day',)   # for these, a positive delta means poisoning hurt


def read_run(path):
    '''Rows keyed by the CSV header. Returns (rows, finished) -- finished once "# BEST" appears.'''
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
                continue                        # partially written line
            row = {}
            for c, v in zip(cols, parts):
                try:
                    row[c] = int(v) if c in ('epoch', 'updates') else (float(v) if v else float('nan'))
                except ValueError:
                    row[c] = float('nan')
            rows.append(row)
    return rows, finished


def pick_row(rows, select, select_by):
    '''One row per run: the last epoch, or the one with the best validation score.'''
    if not rows:
        return None
    if select == 'final':
        return rows[-1]
    key = f'va_{select_by}'
    scored = [r for r in rows if key in r and r[key] == r[key]]
    return max(scored, key=lambda r: r[key]) if scored else rows[-1]


def collect(args):
    '''{(method, variant, dataset, wl): [metric values, one per seed]}'''
    out, skipped = defaultdict(list), 0
    pattern = os.path.join(args.results, '**', '*.csv')
    for path in sorted(glob.glob(pattern, recursive=True)):
        m = NAME_RE.match(os.path.basename(path))
        if not m:
            continue
        dataset = os.path.basename(os.path.dirname(path))
        method = os.path.basename(os.path.dirname(os.path.dirname(path)))
        if dataset not in (args.dirty, args.clean):
            continue
        if args.filter and args.filter not in f'{method}/{dataset}/{os.path.basename(path)}':
            continue

        rows, finished = read_run(path)
        if not finished and not args.include_running:
            skipped += 1
            continue
        row = pick_row(rows, args.select, args.select_by)
        if row is None or args.metric not in row or row[args.metric] != row[args.metric]:
            continue
        out[(method, m.group('variant').strip('_'), dataset, int(m.group('wl')))].append(row[args.metric])

    if skipped:
        print(f'skipped {skipped} unfinished run(s); --include-running to include them')
    return out


def agg(values):
    '''(mean, standard error, n). sem is 0 for a single seed -- the n column is the caveat.'''
    n = len(values)
    mean = sum(values) / n
    if n < 2:
        return mean, 0.0, n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var / n), n


def read_argus(args):
    '''{'dirty'|'clean': (mean, sem)} from the argus_results_*.csv the prior_works script writes.'''
    if not args.argus_dir:
        return {}
    col = args.metric[3:] if args.metric.startswith('te_') else args.metric
    found = {}
    for path in sorted(glob.glob(os.path.join(args.argus_dir, 'argus_results_*.csv'))):
        base = os.path.basename(path).lower()
        arm = 'dirty' if 'dirty' in base else ('clean' if 'clean' in base else None)
        if arm is None:
            continue
        stats = {}
        with open(path) as f:
            for row in csv.DictReader(f):
                label = (row.get('') or row.get('Unnamed: 0') or '').strip()
                if label in ('mean', 'sem') and col in row and row[col]:
                    stats[label] = float(row[col])
        if 'mean' in stats:
            found[arm] = (stats['mean'], stats.get('sem', 0.0))
            print(f'argus {arm}: {stats["mean"]:.4f} +/- {stats.get("sem", 0.0):.4f}  ({os.path.basename(path)})')
        else:
            print(f'argus: no "{col}" column in {os.path.basename(path)}, skipping')
    return found


def main():
    ap = ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--results', default='../results')
    ap.add_argument('--dirty', default='lanl14argus-dirtyts')
    ap.add_argument('--clean', default='lanl14argus-cleants')
    ap.add_argument('--metric', default='te_ap',
                    help='te_ap, te_ap_lift, te_auc, te_prec_r50, te_fp_day_r50')
    ap.add_argument('--select', choices=['best', 'final'], default='best',
                    help="'best' picks each run's highest-validation epoch, 'final' its last")
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--filter', default='', help='substring match on method/dataset/filename')
    ap.add_argument('--include-running', action='store_true')
    ap.add_argument('--argus-dir', default='../prior_works',
                    help="draw Argus as horizontal bands; '' to omit")
    ap.add_argument('--out', default='poison_by_wl.png')
    args = ap.parse_args()

    data = collect(args)
    if not data:
        raise SystemExit(f'no runs matched under {args.results} for {args.dirty} / {args.clean}')

    series = sorted({(k[0], k[1]) for k in data})
    argus = read_argus(args)
    lower_better = any(t in args.metric for t in LOWER_IS_BETTER)

    # ---- the table. Printed as well as plotted: the palette's aqua slot carries a contrast
    # ---- WARN, which obliges visible labels or a table view.
    print(f'\n{args.metric}, {args.select} epoch (by va_{args.select_by})\n')
    hdr = f'{"method":<26} {"wl":>3} {"dirty":>18} {"clean":>18} {"dirty - clean":>18}'
    print(hdr)
    print('-' * len(hdr))
    rows = defaultdict(dict)
    for method, variant in series:
        label = method if not variant else f'{method} [{variant}]'
        wls = sorted({k[3] for k in data if k[0] == method and k[1] == variant})
        for wl in wls:
            d = data.get((method, variant, args.dirty, wl))
            c = data.get((method, variant, args.clean, wl))
            if not d or not c:
                continue
            (dm, ds, dn), (cm, cs, cn) = agg(d), agg(c)
            delta, derr = dm - cm, math.hypot(ds, cs)
            rows[label][wl] = (dm, ds, dn, cm, cs, cn, delta, derr)
            print(f'{label:<26} {wl:>3} {dm:>10.4f}+-{ds:<6.4f} {cm:>10.4f}+-{cs:<6.4f} '
                  f'{delta:>10.4f}+-{derr:<6.4f}  n={dn}/{cn}')
    if not rows:
        raise SystemExit('no walk length had BOTH a dirty and a clean run; nothing to compare')

    # ---- figure ---------------------------------------------------------------------------
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.0, 6.4), sharex=True,
                                   gridspec_kw={'height_ratios': [1.35, 1], 'hspace': 0.12})
    for ax in (ax0, ax1):
        ax.set_facecolor('none')
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(axis='y', color=GRID, linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    for i, (label, bywl) in enumerate(sorted(rows.items())):
        colour = PALETTE[i % len(PALETTE)]
        wls = sorted(bywl)
        dm = [bywl[w][0] for w in wls]; ds = [bywl[w][1] for w in wls]
        cm = [bywl[w][3] for w in wls]; cs = [bywl[w][4] for w in wls]
        dl = [bywl[w][6] for w in wls]; de = [bywl[w][7] for w in wls]

        ax0.errorbar(wls, dm, yerr=ds, color=colour, linewidth=2, marker='o', markersize=5,
                     capsize=3, label=label, zorder=3)
        ax0.errorbar(wls, cm, yerr=cs, color=colour, linewidth=2, linestyle='--', marker='s',
                     markersize=5, markerfacecolor='white', capsize=3, zorder=3)
        ax1.errorbar(wls, dl, yerr=de, color=colour, linewidth=2, marker='o', markersize=5,
                     capsize=3, zorder=3)
        # direct label on the last point, so identity is never colour-alone
        ax1.annotate(label, (wls[-1], dl[-1]), textcoords='offset points', xytext=(6, 0),
                     va='center', fontsize=8, color=colour)

    for arm, (mean, sem) in argus.items():
        ax0.axhspan(mean - sem, mean + sem, color=INK_MUTED, alpha=0.12, zorder=1)
        ax0.axhline(mean, color=INK_MUTED, linewidth=1.2,
                    linestyle='-' if arm == 'dirty' else '--', zorder=2)
        ax0.annotate(f'Argus ({arm})', (1.0, mean), xycoords=('axes fraction', 'data'),
                     textcoords='offset points', xytext=(-2, 3), ha='right', fontsize=8,
                     color=INK_MUTED)
    if {'dirty', 'clean'} <= set(argus):
        d, c = argus['dirty'][0], argus['clean'][0]
        derr = math.hypot(argus['dirty'][1], argus['clean'][1])
        ax1.axhspan(d - c - derr, d - c + derr, color=INK_MUTED, alpha=0.12, zorder=1)
        ax1.axhline(d - c, color=INK_MUTED, linewidth=1.2, zorder=2)
        ax1.annotate('Argus', (1.0, d - c), xycoords=('axes fraction', 'data'),
                     textcoords='offset points', xytext=(-2, 3), ha='right', fontsize=8,
                     color=INK_MUTED)

    ax1.axhline(0, color=INK, linewidth=1, zorder=2)
    hurt = 'above 0 = poisoning hurt' if lower_better else 'below 0 = poisoning hurt'

    ax0.set_ylabel(args.metric, color=INK, fontsize=10)
    ax0.set_title(f'Poisoning the training window: {args.metric} by walk length',
                  color=INK, fontsize=11, loc='left', pad=10)
    ax1.set_ylabel(f'dirty - clean\n({hurt})', color=INK, fontsize=9)
    ax1.set_xlabel('walk length', color=INK, fontsize=10)
    all_wls = sorted({w for bywl in rows.values() for w in bywl})
    # walk lengths are usually powers of two; a log2 axis spaces them evenly instead of
    # crowding the small ones against the left edge
    if len(all_wls) > 2 and all(w > 0 and (w & (w - 1)) == 0 for w in all_wls):
        ax1.set_xscale('log', base=2)
    ax1.set_xticks(all_wls)
    ax1.set_xticklabels([str(w) for w in all_wls])
    ax1.minorticks_off()
    ax1.set_xlim(min(all_wls) * 0.85, max(all_wls) * 1.18)   # room for the direct labels

    style_key = [plt.Line2D([], [], color=INK_MUTED, lw=2, marker='o', markersize=5, label='dirty'),
                 plt.Line2D([], [], color=INK_MUTED, lw=2, ls='--', marker='s', markersize=5,
                            markerfacecolor='white', label='clean (scrubbed)')]
    # both legends stacked at the top left, so neither lands on a data line
    leg = ax0.legend(loc='upper left', frameon=False, fontsize=9, labelcolor=INK,
                     bbox_to_anchor=(0.0, 1.0))
    ax0.add_artist(leg)
    ax0.legend(handles=style_key, loc='center left', frameon=False, fontsize=9, labelcolor=INK)

    fig.savefig(args.out, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'\nwrote {args.out}')


if __name__ == '__main__':
    main()