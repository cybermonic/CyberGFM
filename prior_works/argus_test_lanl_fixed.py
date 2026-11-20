'''
Argus on the fixed-test-set LANL windows (fixed_test_splits.py): the test set is every edge from
--test-day on, and the training window is the last --window days before it.

This is argus_test_lanl_ts.py with a window that need not start at hour 0; everything else is
that script's protocol. Every hour of the window is training data, with a random --val-frac of
each hour's EDGES held out for model selection, and everything from the cutoff on is test,
embedded by running the GRU on from the training window. Attack edges inside the window are
kept, unlabeled (dirty); --scrub removes them from train and validation (clean).

It reads the raw CSR, not the split files, because Argus needs its own edge features. The test
set is the same edges CyberGFM, Euler and PIKACHU score on lanl14argus-fixed{T}-{arm}{K}d, and
the results go to <out-dir>/lanl14argus-fixed{T}-{arm}{K}d.csv, under that same name.
--window equal to --test-day is the split argus_test_lanl_ts.py --cutoff 24*T builds.

    python argus_test_lanl_fixed.py --device 0 --seed 1 --window 4            # dirty, days 8-12
    python argus_test_lanl_fixed.py --device 0 --seed 1 --window 4 --scrub    # clean
'''
import torch

from argus_common import arguments, build_once, finish, make_snapshots, run

HOUR = 3600


def build(args):
    g = torch.load(args.csr, weights_only=False)
    src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    feats = g.raw_edge_attr if 'raw_edge_attr' in g else g.edge_attr
    hour = g.ts // HOUR
    mal = torch.zeros(g.col.size(0), dtype=torch.long)
    mal[g.is_mal] = 1
    cutoff = 24 * args.test_day
    start = cutoff - 24 * args.window

    # per-hour random edge holdout ("the 5% validation split occurs within each snapshot")
    gen = torch.Generator().manual_seed(args.split_seed)
    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    for h in range(start, cutoff):
        idx = (hour == h).nonzero().squeeze(1)
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        cut = int(idx.numel() * (1 - args.val_frac))
        tr[idx[:cut]], va[idx[cut:]] = True, True
    te = hour >= cutoff

    in_window = int(mal[tr | va].sum())
    if args.scrub:
        tr &= mal == 0
        va &= mal == 0
    print(f'hours {start}-{cutoff}: train {int(tr.sum()):,}, val {int(va.sum()):,}, test '
          f'{int(te.sum()):,} edges ({int(mal[te].sum())} malicious); {in_window} malicious in '
          f'the training window, {"removed" if args.scrub else "kept, unlabeled"}')

    def snaps(m, axis, labelled):
        return make_snapshots(src[m], g.col[m], hour[m], axis, feats[m], seconds=HOUR,
                              num_nodes=g.x.size(0), ew=g.edge_weight[m] if 'edge_weight' in g
                              else None, y=mal[m] if labelled else None)

    window, after = torch.arange(start, cutoff), torch.arange(cutoff, int(hour.max()) + 1)
    return snaps(tr, window, False), snaps(va, window, False), snaps(te, after, True)


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0], beta=0.01, gamma=0.01, bs=24, undirected=False)
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_raw_csr.pt')
    ap.add_argument('--test-day', type=int, default=12, help='the test set is every edge from this day on')
    ap.add_argument('--window', type=int, required=True, help='days of training data before the test day')
    ap.add_argument('--scrub', action='store_true', help='remove malicious edges from the window')
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--split-seed', type=int, default=0)
    args = finish(ap.parse_args())
    if not 1 <= args.window <= args.test_day:
        raise SystemExit(f'--window must be 1 to {args.test_day} days')

    name = f'lanl14argus-fixed{args.test_day}-{"clean" if args.scrub else "dirty"}{args.window}d'
    tr, va, te = build_once(f'{args.cache}/argus_{name}_vf{args.val_frac:g}_v6.pt',
                            lambda: build(args))
    # the same columns argus_test_lanl_ts.py records; the window is in the dataset name
    run(args, tr, va, te, temporal=True, experiment=name, cutoff=24 * args.test_day,
        scrub=int(args.scrub))