'''
Mean ± standard error over seeds for every <out-dir>/<model>/<dataset>*.csv that
run_baselines.sh produced. Prints one table per dataset and writes the same numbers to a CSV
(one row per dataset x model, <metric>_mean and <metric>_se columns).

    python show_baseline_results.py ../results/baselines              # -> ../results/baselines/summary.csv
    python show_baseline_results.py ../results/baselines --metrics auc ap ap_snooped
    python show_baseline_results.py ../results/baselines --csv baselines.csv
'''
import argparse
import csv
import glob
import math
import os
from collections import defaultdict

MODEL_ORDER = ['sage', 'gin', 'n2v', 'euler', 'pikachu']


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float('nan')
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def model_key(m):
    return (MODEL_ORDER.index(m) if m in MODEL_ORDER else 99, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', nargs='?', default='../results/baselines')
    ap.add_argument('--metrics', nargs='+', default=['auc', 'ap', 'prec_r50', 'fp_day_r50'])
    ap.add_argument('--csv', help='summary CSV to write (default: <root>/summary.csv)')
    args = ap.parse_args()
    out_csv = args.csv or os.path.join(args.root, 'summary.csv')

    table = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(args.root, '*', '*.csv'))):
        model, stem = path.split(os.sep)[-2], os.path.basename(path)[:-4]
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if rows:
            table[stem][model] = rows

    summary = []
    for ds in sorted(table):
        print(f'\n== {ds} ==')
        hdr = ['model', 'n'] + args.metrics
        lines = []
        for model in sorted(table[ds], key=model_key):
            rows = table[ds][model]
            cells = [model, str(len(rows))]
            rec = {'dataset': ds, 'model': model, 'n': len(rows)}
            for k in args.metrics:
                xs = [float(r[k]) for r in rows if r.get(k) not in (None, '')]
                if not xs:
                    cells.append('-')
                    rec[f'{k}_mean'] = rec[f'{k}_se'] = ''
                    continue
                m, se = mean_se(xs)
                rec[f'{k}_mean'], rec[f'{k}_se'] = m, ('' if math.isnan(se) else se)
                cells.append(f'{m:.0f} ± {se:.0f}' if k.startswith('fp_day') else f'{m:.4f} ± {se:.4f}')
            lines.append(cells)
            summary.append(rec)
        w = [max(len(h), *(len(l[i]) for l in lines)) for i, h in enumerate(hdr)]
        print('  '.join(h.ljust(w[i]) for i, h in enumerate(hdr)))
        for l in lines:
            print('  '.join(c.ljust(w[i]) for i, c in enumerate(l)))

    if not summary:
        print(f'no results under {args.root}')
        return
    fields = ['dataset', 'model', 'n'] + [f'{k}_{s}' for k in args.metrics for s in ('mean', 'se')]
    os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    print(f'\nwrote {out_csv}')


if __name__ == '__main__':
    main()
