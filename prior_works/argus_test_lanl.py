'''
Argus on the random LANL split: CyberGFM's own split files, so both models score the same edges
(lanl_preprocess/nontemporal_split.py -- seed 0, 80/10/10, every malicious edge in test).

Hourly snapshots over the whole 14.7 days. Validation and test edges are held out of the graph
and scored against the training graph's embedding for their hour.

    python argus_test_lanl.py --device 0 --seed 1
'''
import torch

from argus_common import arguments, build_once, finish, make_snapshots, run

HOUR = 3600


def build(args):
    gs = {s: torch.load(args.split_files.format(split=s), weights_only=False)
          for s in ('tr', 'va', 'te')}
    last = max(int(g.ts.max()) for g in gs.values()) // HOUR
    axis = torch.arange(last + 1)

    def snaps(g, labelled):
        feats = g.raw_edge_attr if 'raw_edge_attr' in g else g.edge_attr
        return make_snapshots(g.src, g.col, g.ts // HOUR, axis, feats, seconds=HOUR,
                              num_nodes=g.x.size(0), ew=g.get('edge_weight'),
                              y=g.label if labelled else None)

    return snaps(gs['tr'], False), snaps(gs['va'], False), snaps(gs['te'], True)


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0], beta=0.01, gamma=0.01, bs=24, undirected=False)
    ap.add_argument('--split-files', default='../data/lanl14argus_tgraph_raw_{split}.pt')
    args = finish(ap.parse_args())

    tr, va, te = build_once(f'{args.cache}/argus_lanl14argus_v6.pt', lambda: build(args))
    run(args, tr, va, te, temporal=False, experiment='lanl14argus')