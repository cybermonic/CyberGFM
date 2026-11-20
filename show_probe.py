'''
Which validation metric tracks test AP? Reads the probe files written by lp_finetune.py --val-probe
(<root>/<exp>/<dataset>/probe/*.csv; run_probe.sh) and, for every candidate rule, reports the
test AP at the epoch that rule would select as a fraction of the run's best epoch (1.00 = the rule
always picks the best epoch), averaged over runs.

Candidate rules:
    va_<positives>_<negatives>_<auc|ap>   highest validation AUC / AP, where
        positives  all | newpair (held-out edges whose node pair is not in training)
                       | newedge (held-out edges whose pair + features combination is not in training)
        negatives  uniform | degree (plausible source and destination, never a training pair)
                           | featswap (a real pair with another edge's features)
        va_all_uniform_auc is the rule in use now.
    nll_<positives>                       lowest held-out NLL of the true destination
    epoch 0 ... last epoch                no selection at all (epoch 0 = the pretrained model, not fine-tuned)

Text only; fine on the login node.

    python show_probe.py                      # every probe run under results-probe/
    python show_probe.py --seeds 1 2          # fix the rule on these seeds ...
    python show_probe.py --seeds 3 4 5        # ... and confirm it on the others
    python show_probe.py --detail             # also each run's per-epoch numbers
'''
import argparse
import collections
import csv
import glob
import math
import os
import re


def read(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        try:
            out.append({k: float(v) for k, v in r.items()})
        except (TypeError, ValueError):
            pass
    return out


def pick(rows, key, lower=False):
    '''Index of the selected epoch, or None if the metric is missing / nan everywhere.'''
    vals = [(r.get(key, float('nan')), i) for i, r in enumerate(rows)]
    vals = [(v, i) for v, i in vals if v == v]
    if not vals:
        return None
    return (min(vals, key=lambda t: (t[0], t[1])) if lower else max(vals, key=lambda t: (t[0], -t[1])))[1]


def mean(xs):
    xs = [x for x in xs if x is not None and x == x]
    return sum(xs) / len(xs) if xs else float('nan')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--root', default='results-probe')
    ap.add_argument('--seeds', type=int, nargs='*', help='only these pretraining seeds (_ps<N>)')
    ap.add_argument('--detail', action='store_true')
    ap.add_argument('--top', type=int, default=12, help='rules shown per table, best first (0 = all)')
    args = ap.parse_args()

    groups = collections.defaultdict(list)          # (lr, exp, dataset) -> runs
    for path in sorted(glob.glob(os.path.join(args.root, '*', '*', 'probe', '*.csv'))):
        exp, ds = path.split(os.sep)[-4:-2]
        name = os.path.basename(path)
        ps = re.search(r'_ps(\d+)', name)
        if args.seeds and (not ps or int(ps.group(1)) not in args.seeds):
            continue
        lr = re.search(r'_lr([0-9.e+-]+?)(?:_|\.csv)', name)
        rows = read(path)
        if len(rows) >= 2:
            groups[(lr.group(1) if lr else 'default', exp, ds)].append((name, rows))
    if not groups:
        print(f'no probe files under {args.root}/*/*/probe/ -- run run_probe.sh first')
        return

    for lr in sorted({k[0] for k in groups}):
        cols = sorted(k for k in groups if k[0] == lr)
        rules = collections.OrderedDict()
        keys = sorted({c for k in cols for _, rows in groups[k] for c in rows[0]})
        for c in keys:
            if c.startswith('va_'):
                rules[c] = (c, False)
            elif c.startswith('nll_'):
                rules[c] = (c, True)
        n_ep = max(len(rows) for k in cols for _, rows in groups[k])

        table = collections.defaultdict(dict)       # rule -> column -> (ratio, epoch)
        absolute = {}
        for k in cols:
            ratios, epochs = collections.defaultdict(list), collections.defaultdict(list)
            orc, cur = [], []
            for _, rows in groups[k]:
                te = [r['te_ap'] for r in rows]
                best = max(te)
                orc.append(best)
                for rule, (key, lower) in rules.items():
                    i = pick(rows, key, lower)
                    if i is not None and best > 0:
                        ratios[rule].append(te[i] / best)
                        epochs[rule].append(rows[i]['epoch'])
                last = int(max(r['epoch'] for r in rows))
                for e in range(0, last + 1):
                    j = next((i for i, r in enumerate(rows) if int(r['epoch']) == e), None)
                    if j is not None and best > 0:
                        tag = f'epoch {e}' + (' (no fine-tuning)' if e == 0 else ' (last)' if e == last else '')
                        ratios[tag].append(te[j] / best)
                        epochs[tag].append(e)
                i = pick(rows, 'va_all_uniform_auc')
                cur.append(te[i] if i is not None else float('nan'))
            for rule in ratios:
                table[rule][k] = (mean(ratios[rule]), mean(epochs[rule]))
            absolute[k] = (len(groups[k]), mean(orc), mean(cur))

        order = sorted(table, key=lambda r: -mean([table[r].get(k, (float('nan'),))[0] for k in cols]))
        if args.top:
            keep = set(order[:args.top]) | {'va_all_uniform_auc'} | {r for r in order if r.startswith('epoch ')}
            order = [r for r in order if r in keep]
        heads = [f'{k[1].replace("lp-", "")}/{k[2].replace("lanl14argus", "lanl")}' for k in cols]
        w = max(len(r) for r in order) + 2
        cw = max(14, max(len(h) for h in heads) + 2)
        print(f'\n=== lr {lr}: test AP at the selected epoch / best epoch   (mean selected epoch in brackets) ===')
        print(f'{"rule":<{w}}' + ''.join(f'{h:>{cw}}' for h in heads) + f'{"mean":>8}')
        print(f'{"  runs":<{w}}' + ''.join(f'{absolute[k][0]:>{cw}}' for k in cols))
        for r in order:
            cells, vals = '', []
            for k in cols:
                if k in table[r]:
                    v, e = table[r][k]
                    cells += f'{f"{v:.2f} ({e:.1f})":>{cw}}'
                    vals.append(v)
                else:
                    cells += f'{"-":>{cw}}'
            mark = '   <- current rule' if r == 'va_all_uniform_auc' else ''
            print(f'{r:<{w}}{cells}{mean(vals):>8.2f}{mark}')
        print(f'{"  test AP, best epoch":<{w}}' + ''.join(f'{absolute[k][1]:>{cw}.4f}' for k in cols))
        print(f'{"  test AP, current rule":<{w}}' + ''.join(f'{absolute[k][2]:>{cw}.4f}' for k in cols))

        if args.detail:
            show = ['va_all_uniform_auc', 'va_newpair_degree_auc', 'va_newedge_degree_auc', 'va_all_featswap_auc', 'nll_all']
            for k in cols:
                for name, rows in groups[k]:
                    print(f'\n  {k[1]}/{k[2]}/{name}')
                    print('    epoch   te_ap  ' + '  '.join(f'{c.replace("va_", ""):>20}' for c in show))
                    for r in rows:
                        print(f'    {int(r["epoch"]):>5}  {r["te_ap"]:.4f}  '
                              + '  '.join(f'{r.get(c, float("nan")):>20.4f}' for c in show))

    print('\nRead across a row: a good rule is near 1.00 in EVERY column, not just on average.')
    print('Compare "test AP, best epoch" between the lr tables to see whether the lower rate costs accuracy.')


if __name__ == '__main__':
    main()
