'''
Euler baseline: a 2-layer GCN on every snapshot (one-hot node features, shared weights), a GRU
over the snapshot embeddings, and a dot-product decoder trained against uniformly random node
pairs. The edges of snapshot t are scored with the embedding of t; anomaly score = -<z_u, z_v>.

Each snapshot is a simple graph (repeated flows between a pair collapse to one edge, as in the
static GNN baselines); --no-dedup keeps the multigraph. Snapshots are processed `--bs` at a time with the GRU state carried across pieces (truncated
backprop through time, one optimizer step per piece). For temporal splits the GRU runs on past
the training window over the observed val/test graph; see baseline_common.Timeline.

On OpTC, messages propagate over the undirected view of each snapshot by default (--directed to
turn it off). optc-argus's split files hold both directions of every edge in the training window
but only one in the test window (check_symmetry.py), so a directed GCN embeds test snapshots from
a graph shaped unlike anything it trained on. Argus's OpTC script propagates undirected for the
same reason; the loss and the scored edges stay directed either way.

    python euler_test.py --dataset lanl14argus --device 0 --seed 1
    python euler_test.py --dataset optc-argus --device 0 --seed 1
'''
import time

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops

from baseline_common import (Selector, Timeline, arguments, dataset_kind, finish, load_splits, local_pairs,
                             log_epoch, random_pairs, run, test_days, test_metrics, to_undirected,
                             val_auc)


class Euler(nn.Module):
    def __init__(self, num_nodes, hidden, emb_dim):
        super().__init__()
        self.gcn1 = GCNConv(num_nodes, hidden)
        self.gcn2 = GCNConv(hidden, hidden)
        self.rnn = nn.GRU(hidden, emb_dim)               # (T, N, hidden): time-major

    def forward(self, x, eis, h=None):
        zs = []
        for ei in eis:
            z = torch.relu(self.gcn1(x, ei))
            zs.append(torch.relu(self.gcn2(z, ei)))
        out, h = self.rnn(torch.stack(zs), h)
        return out, h                                     # out[t]: (N, emb_dim)


def score(z, ei):
    return -(z[ei[0]] * z[ei[1]]).sum(1)


def fit(args, tl, num_nodes, days):
    dev = args.device
    x = torch.eye(num_nodes, device=dev)

    if not args.no_dedup:   # one row per (src, dst) per snapshot, for propagation and the loss alike
        tl.graph = [g.unique(dim=1) if g.size(1) else g for g in tl.graph]
        tl.tr = [g.unique(dim=1) if g.size(1) else g for g in tl.tr]

    def mp(t):
        ei = tl.graph[t]
        ei = to_undirected(ei) if args.undirected else ei
        return add_remaining_self_loops(ei.to(dev), num_nodes=num_nodes)[0]

    graphs = [mp(t) for t in range(len(tl))]
    gen = torch.Generator().manual_seed(0)
    va_neg = [random_pairs(e.size(1), num_nodes, gen) for e in tl.va]
    te_y = torch.cat(tl.y)
    bs = args.bs if args.bs > 0 else len(tl)

    model = Euler(num_nodes, args.hidden, args.emb_dim).to(dev)
    opt = Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    sel = Selector(args.patience)

    for e in range(args.epochs):
        t0 = time.time()
        model.train()
        h, losses = None, []
        for a in range(0, tl.n_train, bs):
            b = min(a + bs, tl.n_train)
            opt.zero_grad()
            zs, h = model(x, graphs[a:b], h)
            h = h.detach()
            loss, n = 0, 0
            for i, t in enumerate(range(a, b)):
                pos = tl.tr[t].to(dev)
                if pos.size(1) == 0:
                    continue
                neg = (local_pairs(pos, dev) if args.neg == 'local'
                       else random_pairs(pos.size(1), num_nodes, device=dev))
                logits = torch.cat([-score(zs[i], pos), -score(zs[i], neg)])
                labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(dev)
                loss, n = loss + bce(logits, labels), n + 1
            if n:
                (loss / n).backward()
                opt.step()
                losses.append(loss.item() / n)

        model.eval()
        vp, vn, ts = [], [], []
        with torch.no_grad():
            h = None
            for a in range(0, len(tl), bs):
                zs, h = model(x, graphs[a:a + bs], h)
                for i, t in enumerate(range(a, min(a + bs, len(tl)))):
                    if tl.va[t].size(1):
                        vp.append(score(zs[i], tl.va[t].to(dev)).cpu())
                        vn.append(score(zs[i], va_neg[t].to(dev)).cpu())
                    ts.append(score(zs[i], tl.te[t].to(dev)).cpu())
        va_m = val_auc(torch.cat(vp), torch.cat(vn))
        test = test_metrics(te_y, torch.cat(ts), days)
        improved = sel.update(e, va_m, test)
        log_epoch(e, sum(losses) / max(1, len(losses)), va_m, test, improved, t0)
        if sel.stop:
            break
    return sel.result()


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0])
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--emb-dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--bs', type=int, default=32, help='snapshots per piece; 0 = all at once')
    ap.add_argument('--undirected', dest='undirected', action='store_true', default=None,
                    help='propagate messages both ways (loss and scores stay on directed edges). '
                         'Default: on for OpTC, off elsewhere, as in the Argus scripts')
    ap.add_argument('--directed', dest='undirected', action='store_false')
    ap.add_argument('--no-dedup', action='store_true',
                    help='keep repeated (src, dst) rows within a snapshot. By default each snapshot '
                         'is a simple graph (as gnn_test.py and the Argus scripts use); with the '
                         'repeats, GCNConv and the loss weight a pair by its flow count, which on '
                         'UNSW (50 nodes, thousands of flows per pair) inverts the ranking. '
                         'Val/test rows always stay one per event')
    ap.add_argument('--neg', choices=['uniform', 'local'], default='uniform',
                    help='negative pairs: uniform over all nodes, or (the original scripts) '
                         'within the id range of the snapshot\'s own sources / destinations')
    ap.add_argument('--axis', choices=['all', 'train'], default='all',
                    help='snapshots: every bucket any split touches, or (the original scripts) '
                         'only buckets with a training edge, dropping val/test edges elsewhere')
    args = finish(ap.parse_args())
    if args.undirected is None:
        # optc-argus's training window carries both directions of every edge (load_argus_data.py
        # adds reverses before the attack only) and its test window does not. Propagating over the
        # directed graph would embed the test window from a differently shaped graph than training.
        args.undirected = dataset_kind(args.dataset) == 'optc'

    tr, va, te, n, temporal = load_splits(args.dataset, args.data)
    tl = Timeline(tr, va, te, args.ts_div, temporal, axis_from=args.axis)
    days = test_days(te)
    run(args, 'euler', lambda: fit(args, tl, n, days),
        ['hidden', 'emb_dim', 'lr', 'epochs', 'patience', 'bs', 'ts_div', 'undirected'])