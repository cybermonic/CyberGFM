'''
Argus on UNSW-NB15 (random edge split, preprocess_unsw.py: seed 0, 80/10/10, malicious edges in
test). Argus never evaluated UNSW, so the choices here are ours:

  * snapshots are distinct timestamps (--ts-div 1), all three splits on one shared axis;
  * edge features are scaled by the TRAINING maximum of each column, for every split;
  * training snapshots are deduplicated per (src, dst) like their loaders (mean features,
    std_edge_w(count) weights); validation and test stay one row per event, so the metrics
    count events, as CyberGFM's do;
  * beta 0.01 and gamma 0.1 (their smaller beta, and their OpTC gamma).

    python argus_test_unsw.py --device 0 --seed 1
'''
import torch

from argus_common import arguments, build_once, finish, make_snapshots, run


def build(args):
    gs = {s: torch.load(f'{args.data}/unsw_tgraph_{s}_raw.pt', weights_only=False)
          for s in ('tr', 'va', 'te')}
    for g in gs.values():
        g.bucket = g.ts.long() // args.ts_div
    axis = torch.cat([g.bucket for g in gs.values()]).unique()
    scale = gs['tr'].raw_edge_attr.float().max(dim=0).values
    scale[scale == 0] = 1.0

    def snaps(g, train, labelled):
        return make_snapshots(g.src, g.col, g.bucket, axis, g.raw_edge_attr.float() / scale,
                              seconds=args.ts_div, num_nodes=g.x.size(0),
                              y=g.label if labelled else None, dedup=train)

    return snaps(gs['tr'], True, False), snaps(gs['va'], False, False), snaps(gs['te'], False, True)


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0], beta=0.01, gamma=0.1, bs=64, undirected=False)
    ap.add_argument('--data', default='../data')
    ap.add_argument('--ts-div', type=int, default=1, help='seconds per snapshot')
    args = finish(ap.parse_args())

    tr, va, te = build_once(f'{args.cache}/argus_unsw_d{args.ts_div}_v6.pt', lambda: build(args))
    run(args, tr, va, te, temporal=False, experiment='unsw', ts_div=args.ts_div)