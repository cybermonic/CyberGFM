'''
Average wall-clock runtime per method from LSF job logs (*.out).

Reads the LSF report at the top of each .out ("Subject: Job N: <name> ... Done|Exited",
"Run time : X sec.") and works out the method from the job name:

    argus_<exp>_s<seed>                          Argus            (logs/argus)
    bl_<model>_<dataset>[tag]_s<seed>            sage/gin/n2v/euler/pikachu  (logs/baselines)
    pt_[gpt_]<static|temporal>_<dataset>_ps<s>   CyberGFM pretraining, BERT or GPT  (logs/lsf)
    ft_[gpt_][rand_]<mode>_<dataset>_ps<s>_wl<w> CyberGFM fine-tuning               (logs/lsf)

A pretraining token budget in the name (pt_static_lanl14argus_3e8_ps1, from run_ntokens.sh) goes
in its own `tokens` column, so the budgets are averaged separately.

LSF appends to -o files, so a resubmitted job leaves several reports in one .out; the LAST one
is used, and the number of earlier attempts is counted. Only jobs that finished successfully
are averaged by default (--include-failed to keep the rest); everything skipped is reported.
Text parsing only -- fine on the login node.

    python aggregate_runtimes.py                                  # logs/lsf logs/baselines logs/argus ...
    python aggregate_runtimes.py logs/baselines logs/argus --by method
    python aggregate_runtimes.py --by method phase dataset --csv runtimes.csv   # pool walk lengths
    python aggregate_runtimes.py logs/ntokens --by mode tokens phase            # the token sweep
'''
import argparse
import csv
import glob
import math
import os
import re
from collections import defaultdict

ARGUS_EXP = {'lanl': 'lanl14argus', 'lanl_ts': 'lanl14argus-ts',
             'lanl_ts_dirty': 'lanl14argus-dirtyts', 'lanl_ts_clean': 'lanl14argus-cleants',
             'optc': 'optc', 'optc_argus': 'optc-argus', 'unsw': 'unsw'}

SUBJECT = re.compile(r'^Subject: Job (\d+): <([^>]+)> in cluster <[^>]*> (\w+)', re.M)
RUNTIME = re.compile(r'Run time\s*:\s*([\d.]+) sec')
EXITED = re.compile(r'Exited with exit code (\d+)')


def split_tokens(dataset):
    '''lanl14argus_3e8 (run_ntokens.sh) -> ('lanl14argus', '3e8'); anything else -> (dataset, '').'''
    m = re.fullmatch(r'(.+)_(\d+e\d+)', dataset)
    return (m[1], m[2]) if m else (dataset, '')


def parse_name(name):
    info = _parse_name(name)
    if info is not None:
        info['dataset'], info['tokens'] = split_tokens(info['dataset'])
    return info


def _parse_name(name):
    '''job name -> dict(method, phase, mode, dataset, wl, seed), or None if unrecognized.'''
    m = re.fullmatch(r'argus_(.+)_s(\d+)', name)
    if m:
        return dict(method='argus', phase='train', mode='', dataset=ARGUS_EXP.get(m[1], m[1]),
                    wl='', seed=m[2])
    m = re.fullmatch(r'bl_(sage|gin|n2v|euler|pikachu)_(.+?)_s(\d+)', name)
    if m:
        return dict(method=m[1], phase='train', mode='', dataset=m[2], wl='', seed=m[3])
    m = re.fullmatch(r'pt_(gpt_)?(static|temporal)_(.+)_ps(\d+)', name)
    if m:
        return dict(method='cybergfm-gpt' if m[1] else 'cybergfm', phase='pretrain', mode=m[2],
                    dataset=m[3], wl='', seed=m[4])
    m = re.fullmatch(r'ft_(gpt_)?(rand_)?(static|temporal)_(.+)_ps(\d+)_wl(\d+)', name)
    if m:
        method = ('cybergfm-gpt' if m[1] else 'cybergfm') + ('-randinit' if m[2] else '')
        return dict(method=method, phase='finetune', mode=m[3], dataset=m[4], wl=m[6], seed=m[5])
    return None


def parse_log(path):
    text = open(path, errors='replace').read()
    subjects = list(SUBJECT.finditer(text))
    if not subjects:
        return None
    last = subjects[-1]
    block = text[last.start():]
    rt = RUNTIME.search(block)
    ex = EXITED.search(block)
    status = last[3]                                   # Done / Exited
    ok = status == 'Done' and 'Successfully completed' in block
    return dict(job=last[1], name=last[2], status=status, ok=ok,
                exit=int(ex[1]) if ex else (0 if ok else ''),
                seconds=float(rt[1]) if rt else None, attempts=len(subjects))


def mean_sd(xs):
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else float('nan')
    return m, sd


def hms(sec):
    if sec != sec:                                     # nan
        return '-'
    sec = int(round(sec))
    return f'{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}'


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('dirs', nargs='*',
                    default=['logs/lsf', 'logs/lsf-rand-e20', 'logs/baselines', 'logs/argus'])
    ap.add_argument('--by', nargs='+',
                    default=['source', 'method', 'phase', 'mode', 'dataset', 'tokens', 'wl'],
                    choices=['source', 'method', 'phase', 'mode', 'dataset', 'tokens', 'wl'],
                    help='columns to group by (source = the log directory)')
    ap.add_argument('--include-failed', action='store_true',
                    help='average every job with a run time, not just successful ones')
    ap.add_argument('--csv', help='also write the table here')
    ap.add_argument('--jobs-csv', help='write one row per parsed job here (for checking)')
    args = ap.parse_args()

    jobs, unparsed, unknown = [], [], []
    for d in args.dirs:
        files = sorted(glob.glob(os.path.join(d, '*.out')))
        if not files:
            print(f'(no .out files in {d})')
        for f in files:
            r = parse_log(f)
            if r is None:
                unparsed.append(f)
                continue
            info = parse_name(r['name'])
            if info is None:
                unknown.append(r['name'])
                continue
            jobs.append(dict(source=os.path.basename(os.path.normpath(d)), file=f, **info, **r))

    groups = defaultdict(list)
    skipped = defaultdict(int)
    for j in jobs:
        key = tuple(j[k] for k in args.by)
        if j['seconds'] is None or not (j['ok'] or args.include_failed):
            skipped[key] += 1
            continue
        groups[key].append(j['seconds'])

    rows = []
    def order(key):   # natural order: wl 2 before 10, 3e8 before 11e8
        return tuple(tuple((0, int(t), '') if t.isdigit() else (1, 0, t)
                           for t in re.split(r'(\d+)', str(k))) for k in key)

    for key in sorted(set(groups) | set(skipped), key=order):
        xs = groups.get(key, [])
        m, sd = mean_sd(xs) if xs else (float('nan'), float('nan'))
        rows.append(dict(zip(args.by, key), n=len(xs), skipped=skipped.get(key, 0),
                         mean_sec=round(m, 1) if xs else '', sd_sec=round(sd, 1) if sd == sd else '',
                         mean_hms=hms(m), min_hms=hms(min(xs)) if xs else '-',
                         max_hms=hms(max(xs)) if xs else '-'))

    hdr = list(args.by) + ['n', 'skipped', 'mean', 'sd', 'min', 'max']
    table = [[str(r[k]) for k in args.by] + [str(r['n']), str(r['skipped']), r['mean_hms'],
             hms(r['sd_sec']) if r['sd_sec'] != '' else '-', r['min_hms'], r['max_hms']]
             for r in rows]
    w = [max(len(h), *(len(t[i]) for t in table)) if table else len(h) for i, h in enumerate(hdr)]
    print('  '.join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for t in table:
        print('  '.join(c.ljust(w[i]) for i, c in enumerate(t)))

    n_fail = sum(1 for j in jobs if not j['ok'])
    n_retry = sum(1 for j in jobs if j['attempts'] > 1)
    print(f'\n{len(jobs)} jobs parsed; {n_fail} did not finish successfully'
          f'{" (excluded)" if not args.include_failed else ""}; {n_retry} .out files hold more '
          f'than one attempt (last one used)')
    if unparsed:
        print(f'{len(unparsed)} .out files had no LSF report (still running?): '
              + ', '.join(os.path.basename(f) for f in unparsed[:5]) + (' ...' if len(unparsed) > 5 else ''))
    if unknown:
        print(f'{len(unknown)} job names not recognized: ' + ', '.join(sorted(set(unknown))[:5]))

    if args.csv:
        with open(args.csv, 'w', newline='') as f:
            cw = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else hdr)
            cw.writeheader()
            cw.writerows(rows)
        print(f'wrote {args.csv}')
    if args.jobs_csv:
        fields = ['source', 'file', 'name', 'job', 'method', 'phase', 'mode', 'dataset', 'tokens', 'wl',
                  'seed', 'status', 'ok', 'exit', 'seconds', 'attempts']
        with open(args.jobs_csv, 'w', newline='') as f:
            cw = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            cw.writeheader()
            cw.writerows(jobs)
        print(f'wrote {args.jobs_csv}')


if __name__ == '__main__':
    main()