'''
GraphSAGE / GIN link-prediction baseline: a 2-layer GNN over the static training graph (one-hot
node features), trained with a dot-product decoder against uniformly random node pairs, full
batch. Validation and test edges are held out and scored against the training graph's
embedding; anomaly score = -<z_u, z_v>. See baseline_common.py for data, selection and metrics.

    python gnn_test.py --dataset lanl14argus --device 0 --seed 1
    python gnn_test.py --dataset optc-argus --gin --device 0 --seed 1
'''
import time

import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.nn import GIN, GraphSAGE
from torch_geometric.utils import add_remaining_self_loops

from baseline_common import (Selector, arguments, finish, load_splits, log_epoch, random_pairs,
                             run, test_days, test_metrics, to_undirected, val_auc)


class GNN(nn.Module):
    def __init__(self, num_nodes, hidden, emb_dim, gin):
        super().__init__()
        cls = GIN if gin else GraphSAGE
        self.gnn = cls(num_nodes, hidden, 2, out_channels=emb_dim)

    def forward(self, x, ei):
        return self.gnn(x, ei)


def score(z, ei, bs=1 << 20):
    '''Anomaly score: low dot product = unlikely edge = anomalous.'''
    return torch.cat([-(z[e[0]] * z[e[1]]).sum(1) for e in ei.split(bs, dim=1)])


def fit(args, tr, va, te, num_nodes, days):
    dev = args.device
    x = torch.eye(num_nodes, device=dev)
    pos = tr.ei.unique(dim=1).to(dev)                    # loss targets: the directed edges
    mp = to_undirected(pos) if args.undirected else pos  # what messages propagate over
    mp = add_remaining_self_loops(mp, num_nodes=num_nodes)[0]

    gen = torch.Generator().manual_seed(0)                # fixed validation negatives
    va_pos, va_neg = va.ei.to(dev), random_pairs(len(va), num_nodes, gen, dev)
    te_ei, te_y = te.ei.to(dev), te.y

    model = GNN(num_nodes, args.hidden, args.emb_dim, args.gin).to(dev)
    opt = Adam(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    sel = Selector(args.patience)

    for e in range(args.epochs):
        t0 = time.time()
        model.train()
        opt.zero_grad()
        z = model(x, mp)
        neg = random_pairs(pos.size(1), num_nodes, device=dev)
        logits = torch.cat([-score(z, pos), -score(z, neg)])
        labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(dev)
        loss = bce(logits, labels)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            z = model(x, mp)
            va_m = val_auc(score(z, va_pos), score(z, va_neg))
            test = test_metrics(te_y, score(z, te_ei).cpu(), days)
        improved = sel.update(e, va_m, test)
        log_epoch(e, loss.item(), va_m, test, improved, t0)
        if sel.stop:
            break
    return sel.result()


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0])
    ap.add_argument('--gin', action='store_true', help='GIN instead of GraphSAGE')
    ap.add_argument('--hidden', type=int, default=128)
    ap.add_argument('--emb-dim', type=int, default=64)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--patience', type=int, default=25)
    ap.add_argument('--undirected', action='store_true',
                    help='propagate messages both ways (loss and scores stay on directed edges)')
    args = finish(ap.parse_args())

    tr, va, te, n, _ = load_splits(args.dataset, args.data)
    days = test_days(te)
    run(args, 'gin' if args.gin else 'sage', lambda: fit(args, tr, va, te, n, days),
        ['hidden', 'emb_dim', 'lr', 'epochs', 'patience', 'undirected'])
