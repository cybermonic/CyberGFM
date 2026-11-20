'''
Argus on a temporal LANL split -- the paper's protocol at the default --cutoff 41 (147600 s, the
first redteam event), and the poisoning arms at later cutoffs.

Every hour before the cutoff is training data, with a random --val-frac of each hour's EDGES held
out for model selection, so train and validation span the same hours. Everything from the cutoff
on is test, embedded by running the GRU on from the training window.

At a later cutoff the training window contains redteam edges, unlabeled (dirty); --scrub removes
them from train and validation (clean). Both arms score the identical test set. The cutoff must
match CyberGFM's dirtyts/cleants files (temporal_split.py, tr_days=7 -> 168 h).

    python argus_test_lanl_ts.py --device 0 --seed 1                        # paper, 41 h
    python argus_test_lanl_ts.py --device 0 --seed 1 --cutoff 168           # dirty
    python argus_test_lanl_ts.py --device 0 --seed 1 --cutoff 168 --scrub   # clean
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
    cutoff = int(args.cutoff)

    # per-hour random edge holdout ("the 5% validation split occurs within each snapshot")
    gen = torch.Generator().manual_seed(args.split_seed)
    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    for h in range(cutoff):
        idx = (hour == h).nonzero().squeeze(1)
        idx = idx[torch.randperm(idx.numel(), generator=gen)]
        cut = int(idx.numel() * (1 - args.val_frac))
        tr[idx[:cut]], va[idx[cut:]] = True, True
    te = hour >= cutoff

    in_window = int(mal[tr | va].sum())
    if args.scrub:
        tr &= mal == 0
        va &= mal == 0
    print(f'cutoff {cutoff} h: train {int(tr.sum()):,}, val {int(va.sum()):,}, test '
          f'{int(te.sum()):,} edges ({int(mal[te].sum())} malicious); {in_window} malicious in '
          f'the training window, {"removed" if args.scrub else "kept, unlabeled"}')

    def snaps(m, axis, labelled):
        return make_snapshots(src[m], g.col[m], hour[m], axis, feats[m], seconds=HOUR,
                              num_nodes=g.x.size(0), ew=g.edge_weight[m] if 'edge_weight' in g
                              else None, y=mal[m] if labelled else None)

    window, after = torch.arange(cutoff), torch.arange(cutoff, int(hour.max()) + 1)
    return snaps(tr, window, False), snaps(va, window, False), snaps(te, after, True)


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0], beta=0.01, gamma=0.01, bs=24, undirected=False)
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_raw_csr.pt')
    ap.add_argument('--cutoff', type=int, default=41, help='train/test boundary in hours')
    ap.add_argument('--scrub', action='store_true', help='remove malicious edges before cutoff')
    ap.add_argument('--val-frac', type=float, default=0.05)
    ap.add_argument('--split-seed', type=int, default=0)
    args = finish(ap.parse_args())

    name = f'lanl14argus-{"clean" if args.scrub else "dirty"}ts-{args.cutoff}h'
    tr, va, te = build_once(f'{args.cache}/argus_{name}_vf{args.val_frac:g}_v6.pt',
                            lambda: build(args))
    run(args, tr, va, te, temporal=True, experiment=name, cutoff=args.cutoff,
        scrub=int(args.scrub))