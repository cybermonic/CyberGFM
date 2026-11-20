'''
Train-window contamination splits: the same window twice, once with the attack edges inside it
left in (dirty, unlabeled) and once with them removed (clean). Both arms share one test set, so
any score difference comes from training on unlabeled attack traffic.

LANL  lanl14argus-dirtyts{D}d / lanl14argus-cleants{D}d for each --lanl-days D. The window is
      the first D days, split exactly as temporal_split.py's dirty LANL split is:
          train  ts <  0.9 * D days          validation  [0.9 * D days, D days)
          test   ts >= D days
      D=7 reproduces lanl14argus-dirtyts / -cleants edge for edge (checked with --check-7).

OpTC  {base}-dirty{H}h / {base}-clean{H}h for each --optc-hours H. The window runs from the start
      of the data to H hours after the first malicious event (default 24: one attack day), with
      the same 90/10 train/validation rule; test is everything after.

Labels are saved on the test files only, as in the existing dirty split. A report per split (also
written to data/contamination_report.json) gives the malicious edges in train / val / test, and how
many test attack edges repeat a (src, dst) pair that already appears as an attack edge in the
training window -- the pairs a model fitted to the dirty window learns as normal.

Data processing only, no model; it holds the full CSR in memory (a few GB for LANL).

    python contamination_splits.py                                  # LANL 8-12 d, OpTC +24 h
    python contamination_splits.py --lanl-days 7 --check-7          # verify against the old split
    python contamination_splits.py --optc-base optc-ts --optc-hours 24 48 --no-lanl
'''
import argparse
import json
import os

import torch
from torch_geometric.data import Data

DAY, HOUR = 86400, 3600
FIRST_RED = {'optc-argus': 573290, 'optc-ts': 579776, 'optc': 579776}   # temporal_split.py


def reindex(idxptr, subset_mask):
    cum = torch.cat([torch.zeros(1, dtype=torch.long), subset_mask.long().cumsum(0)])
    return cum[idxptr]


def build(g, mask, label=None):
    new_ptr = reindex(g.idxptr, mask)
    row = torch.arange(g.x.size(0)).repeat_interleave(new_ptr[1:] - new_ptr[:-1])
    data = Data(x=g.x, idxptr=new_ptr, col=g.col[mask], src=row, ts=g.ts[mask],
                edge_attr=g.edge_attr[mask])
    for k in ('edge_weight', 'raw_edge_attr'):
        if getattr(g, k, None) is not None:
            data[k] = g[k][mask]
    if label is not None:
        data.label = label[mask].float()
    return data


def masks(ts, non_te):
    tr_ts = non_te * 0.9
    return ts < tr_ts, (ts >= tr_ts) & (ts < non_te), ts >= non_te


def pair_overlap(src, dst, mal, window, test):
    '''Test attack edges whose (src, dst) is also an attack edge inside the training window.'''
    n = int(max(src.max(), dst.max())) + 1
    key = src * n + dst
    seen = key[mal & window].unique()
    te_mal = mal & test
    rep = torch.isin(key[te_mal], seen)
    return int(rep.sum()), int(te_mal.sum()), int(key[te_mal].unique().numel())


def write(g, name, out_dir, tr, va, te, label, mal, src):
    for split, m in (('tr', tr), ('va', va), ('te', te)):
        torch.save(build(g, m, label if split == 'te' else None), f'{out_dir}/{name}_tgraph_{split}.pt')
    r = dict(train=int(tr.sum()), val=int(va.sum()), test=int(te.sum()),
             mal_train=int((mal & tr).sum()), mal_val=int((mal & va).sum()), mal_test=int((mal & te).sum()))
    print(f'  {name:<28} train {r["train"]:>10,}  val {r["val"]:>9,}  test {r["test"]:>10,}  | '
          f'malicious: train {r["mal_train"]}, val {r["mal_val"]}, test {r["mal_test"]}')
    return r


def contaminate(g, label, src, non_te, dirty_name, clean_name, out_dir, report):
    mal = label.bool()
    tr, va, te = masks(g.ts, non_te)
    rep, n_te, n_pairs = pair_overlap(src, g.col, mal, tr | va, te)
    report[dirty_name] = write(g, dirty_name, out_dir, tr, va, te, label, mal, src)
    report[clean_name] = write(g, clean_name, out_dir, tr & ~mal, va & ~mal, te, label, mal, src)
    for k in (dirty_name, clean_name):
        report[k].update(window_end_s=int(non_te), test_attack_edges_repeating_a_window_attack_pair=rep)
    if n_te == 0:
        print('    WARNING: no attack edges left in the test window -- this cutoff cannot be scored')
    else:
        print(f'    test attack edges whose (src, dst) is already an attack edge in the window: '
              f'{rep}/{n_te} ({rep / n_te:.1%}); {n_pairs} distinct test attack pairs')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='data')
    ap.add_argument('--lanl-days', type=int, nargs='*', default=[8, 9, 10, 11, 12])
    ap.add_argument('--no-lanl', action='store_true')
    ap.add_argument('--check-7', action='store_true',
                    help='also rebuild D=7 in memory and compare with lanl14argus-dirtyts / -cleants')
    ap.add_argument('--optc-base', default='optc-argus', choices=sorted(FIRST_RED))
    ap.add_argument('--optc-hours', type=int, nargs='*', default=[24],
                    help='hours after the first malicious event that the training window ends')
    ap.add_argument('--no-optc', action='store_true')
    args = ap.parse_args()
    out = args.data
    report_path = f'{out}/contamination_report.json'
    report = json.load(open(report_path)) if os.path.exists(report_path) else {}

    if not args.no_lanl:
        g = torch.load(f'{out}/lanl14argus_tgraph_csr.pt', weights_only=False)
        label = torch.zeros(g.col.size(0))
        label[g.is_mal] = 1
        src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
        span = (int(g.ts.max()) + 1) / DAY
        print(f'LANL: {g.col.size(0):,} edges over {span:.2f} days, {int(label.sum())} malicious')
        if args.check_7:
            tr, va, te = masks(g.ts, 7 * DAY)
            mal = label.bool()
            for name, ms in (('lanl14argus-dirtyts', (tr, va, te)),
                             ('lanl14argus-cleants', (tr & ~mal, va & ~mal, te))):
                for split, m in zip(('tr', 'va', 'te'), ms):
                    old = torch.load(f'{out}/{name}_tgraph_{split}.pt', weights_only=False)
                    same = old.col.size(0) == int(m.sum()) and torch.equal(old.ts, g.ts[m]) \
                        and torch.equal(old.col, g.col[m])
                    print(f'  check D=7 {name} {split}: {"identical" if same else "DIFFERENT"}')
        for d in args.lanl_days:
            if d * DAY >= span * DAY:
                print(f'  skipping {d} d: the data only covers {span:.2f} days')
                continue
            contaminate(g, label, src, d * DAY, f'lanl14argus-dirtyts{d}d', f'lanl14argus-cleants{d}d',
                        out, report)
        del g

    if not args.no_optc:
        base = args.optc_base
        g = torch.load(f'{out}/{base}_tgraph_csr.pt', weights_only=False)
        label = g.label.float().clamp(0, 1)
        src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
        first = int(g.ts[label.bool()].min()) if label.any() else FIRST_RED[base]
        print(f'{base}: {g.col.size(0):,} edges over {(int(g.ts.max()) + 1) / DAY:.2f} days, '
              f'{int(label.sum())} malicious; first malicious at {first} s '
              f'(temporal_split.py uses {FIRST_RED[base]})')
        for h in args.optc_hours:
            contaminate(g, label, src, first + h * HOUR, f'{base}-dirty{h}h', f'{base}-clean{h}h',
                        out, report)

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=1)
    print(f'\nreport -> {report_path}')


if __name__ == '__main__':
    main()