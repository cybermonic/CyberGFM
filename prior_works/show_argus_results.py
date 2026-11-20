'''
Mean and SE of every Argus experiment in one or more results directories.

    python show_argus_results.py                                   # ../results/argus
    python show_argus_results.py ../results/argus-paper-defaults
    python show_argus_results.py ../results/argus-a ../results/argus-b      # side by side
    python show_argus_results.py ../results/argus --csv argus.csv           # for the spreadsheet

Reads both layouts:

  current   <experiment>.csv, one row per seed, written by every seed job of the experiment
            (lanl14argus.csv, optc-argus.csv, lanl14argus-cleants-172h_b0.csv, ...)
  older     argus_results_<experiment>_s<seed>.csv, one file per seed job, each holding pandas
            `mean` and `sem` rows after the runs. Those two rows are skipped -- otherwise a file's
            own average would be pooled in as if it were another replication.

Everything else in the directory is ignored: speedtest.csv, *_log.csv, logs/, lock files.

--csv writes show_main_results.py --csv's exact columns, so the rows paste under the CyberGFM
ones. `experiment` is the directory name, which keeps two directories apart in one sheet;
walk_len is present but empty, since Argus has none.
'''
from argparse import ArgumentParser
from collections import OrderedDict
import csv
import glob
import math
import os
import re

# show_main_results.py write_csv, exactly
FIELDS = ['experiment', 'dataset', 'walk_len', 'n', 'n_expected', 'epoch']
METRICS = ['te_auc', 'te_ap', 'te_prec_r50', 'te_fp_day_r50']
for _m in METRICS:
    FIELDS += [_m, f'{_m}_se']
# what the Argus scripts record -> show_main_results names
RENAME = {'auc': 'te_auc', 'ap': 'te_ap', 'prec_r50': 'te_prec_r50', 'fp_day_r50': 'te_fp_day_r50'}

# Pooling rows that differ in any of these would average two different experiments together.
HYPER = ['arch', 'lr', 'h_dim', 'z_dim', 'gru_layers', 'margin', 'gamma', 'beta', 'max_pos',
         'epochs', 'patience', 'bs', 'score_mode', 'edge_weight', 'undirected', 'split',
         'cutoff', 'scrub', 'ts_div']

IGNORE = re.compile(r'(^speedtest|^argus_speedtest|_log$|^argus_main$)')


def experiment_name(stem):
    '''lanl14argus.csv -> lanl14argus;  argus_results_optc-argus_s3.csv -> optc-argus.'''
    stem = re.sub(r'^argus_results_', '', stem)
    return re.sub(r'_s\d+$', '', stem)


def read_rows(path):
    '''Per-run rows as dicts, numbers parsed where possible; pandas mean/sem rows skipped.'''
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            index_col = next(iter(row), None)
            if index_col in ('', 'Unnamed: 0'):
                if not str(row[index_col]).strip().isdigit():
                    continue                            # the `mean` / `sem` rows
                row.pop(index_col)
            out = {}
            for k, v in row.items():
                if k is None:
                    continue
                try:
                    out[k] = float(v)
                except (TypeError, ValueError):
                    out[k] = v
            if isinstance(out.get('ap'), float):
                rows.append(out)
    return rows


def mean_se(xs):
    xs = [x for x in xs if isinstance(x, float) and x == x]
    n = len(xs)
    if not n:
        return None, None
    mu = sum(xs) / n
    if n < 2:
        return mu, None
    return mu, math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1) / n)


def load_dir(d):
    '''experiment -> (rows, files) for one directory.'''
    groups = OrderedDict()
    for path in sorted(glob.glob(os.path.join(d, '*.csv'))):
        stem = os.path.basename(path)[:-4]
        if IGNORE.search(stem):
            continue
        rows = read_rows(path)
        if not rows:
            continue
        g = groups.setdefault(experiment_name(stem), ([], []))
        g[0].extend(rows)
        g[1].append(os.path.basename(path))
    return groups


def summarize(d, name, rows, files, expected):
    rec = {'experiment': os.path.basename(os.path.normpath(d)), 'dataset': name, 'walk_len': '',
           'n': len(rows), 'n_expected': expected}
    ep = mean_se([r.get('epoch') for r in rows])[0]
    rec['epoch'] = '' if ep is None else f'{ep:.4f}'
    for src, dst in RENAME.items():
        mu, se = mean_se([r.get(src) for r in rows])
        rec[dst] = '' if mu is None else f'{mu:.4f}'
        rec[f'{dst}_se'] = '' if se is None else f'{se:.4f}'
    mins = mean_se([r.get('minutes') for r in rows])[0]
    rec['_minutes'] = mins

    warnings = []
    seeds = [r.get('seed') for r in rows if r.get('seed') not in (None, '')]
    if len(seeds) != len(set(seeds)):
        warnings.append('duplicate seeds -- the same replication is counted more than once')
    for h in HYPER:
        vals = {r[h] for r in rows if h in r and r[h] != ''}
        if len(vals) > 1:
            warnings.append(f'mixed {h}: {sorted(map(str, vals))} -- these rows are different '
                            f'experiments and should not be pooled')
    rec['_warnings'] = warnings
    return rec


def main():
    ap = ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dirs', nargs='*', default=['results/argus'],
                    help='results directories to read (default results/argus)')
    ap.add_argument('--csv', help="also write show_main_results.py's --csv format to this path")
    ap.add_argument('--expected', type=int, default=5,
                    help='expected replications per experiment, for the n_expected column')
    args = ap.parse_args()

    out = []
    for d in args.dirs:
        if not os.path.isdir(d):
            raise SystemExit(f'{d}: no such directory')
        groups = load_dir(d)
        if not groups:
            print(f'{d}: no Argus results found')
            continue
        for name, (rows, files) in sorted(groups.items()):
            out.append(summarize(d, name, rows, files, args.expected))

    if not out:
        raise SystemExit(1)

    multi = len(args.dirs) > 1
    w = max(len(r['dataset']) for r in out)
    hdr = (f'{"directory":24s} ' if multi else '') + f'{"experiment":{w}s} {"n":>3s} ' + \
        ' '.join(f'{m[3:]:>18s}' for m in METRICS) + f' {"min/run":>8s}'
    print(hdr)
    print('-' * len(hdr))
    for r in out:
        cells = []
        for m in METRICS:
            if r[m] == '':
                cells.append(f'{"--":>18s}')
            elif r[f'{m}_se'] == '':
                cells.append(f'{float(r[m]):>18.4f}')
            else:
                cells.append(f'{float(r[m]):>9.4f} ±{float(r[f"{m}_se"]):7.4f}')
        mins = '' if r['_minutes'] is None else f'{r["_minutes"]:8.1f}'
        short = '' if r['n'] >= args.expected else '*'
        print((f'{r["experiment"]:24s} ' if multi else '') +
              f'{r["dataset"]:{w}s} {r["n"]:>3d}{short}' + ' '.join(cells) + f' {mins:>8s}')
        for wmsg in r['_warnings']:
            print(f'    ! {wmsg}')
    if any(r['n'] < args.expected for r in out):
        print(f'\n* fewer than {args.expected} runs so far; those SEs are provisional')
    if any(r['te_prec_r50'] == '' for r in out):
        print('-- not recorded by that script (argus_test_optc.py records auc and ap only)')

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction='ignore')
            wr.writeheader()
            wr.writerows(out)
        print(f'\nwrote {args.csv}')


if __name__ == '__main__':
    main()