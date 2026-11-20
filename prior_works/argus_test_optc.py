'''
Argus on OpTC: `optc-argus` (default; Argus's own data and temporal split) or `optc` (the random
edge split). Which kind of split it is comes from the timestamps: temporal if every test edge is
at or after the last training edge.

As their loader does: hourly snapshots (--ts-div 3600, their --delta 1), one row per (src, dst)
per snapshot with the max label and the event count, std_edge_w(count) as the GCN edge weight,
and their Argus_OPTC encoder (--arch optc). Test metrics are per event (count-weighted, as
CyberGFM's test rows are); auc_uniq / ap_uniq are per row, which is how their score_stats counts.

load_argus_data.py adds reverse edges before the attack only, so training is undirected and test
is not. The default undirected propagation (argus.undirected_view) fixes the graph the test
window is embedded with; the scored rows stay the directed ones.

    python argus_test_optc.py --device 0 --seed 1
    python argus_test_optc.py --device 0 --seed 1 --dataset optc
'''
import torch

from argus_common import arguments, build_once, finish, make_snapshots, run


def build(args):
    gs = {s: torch.load(f'{args.data}/{args.dataset}_tgraph_{s}.pt', weights_only=False)
          for s in ('tr', 'va', 'te')}
    for g in gs.values():
        g.bucket = g.ts.long() // args.ts_div
    temporal = int(gs['te'].ts.min()) >= int(gs['tr'].ts.max())

    span = lambda *ss: torch.arange(min(int(gs[s].bucket.min()) for s in ss),
                                    max(int(gs[s].bucket.max()) for s in ss) + 1)
    tr_axis = span('tr', 'va') if temporal else span('tr', 'va', 'te')
    te_axis = span('te') if temporal else tr_axis

    def snaps(g, axis, labelled):
        return make_snapshots(g.src, g.col, g.bucket, axis, torch.ones(g.col.size(0), 1),
                              seconds=args.ts_div, num_nodes=g.x.size(0),
                              y=g.label if labelled else None, dedup=True)

    return (snaps(gs['tr'], tr_axis, False), snaps(gs['va'], tr_axis, False),
            snaps(gs['te'], te_axis, True), temporal)


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0], beta=0.5, gamma=0.1, bs=2, undirected=True,
                   arch='optc')
    ap.add_argument('--dataset', default='optc-argus', help='data/<dataset>_tgraph_{tr,va,te}.pt')
    ap.add_argument('--data', default='../data')
    ap.add_argument('--ts-div', type=int, default=3600, help='seconds per snapshot (theirs: 3600)')
    args = finish(ap.parse_args())

    tr, va, te, temporal = build_once(f'{args.cache}/argus_{args.dataset}_d{args.ts_div}_v6.pt',
                                      lambda: build(args))
    run(args, tr, va, te, temporal=temporal, experiment=args.dataset, ts_div=args.ts_div)