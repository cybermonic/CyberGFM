'''
node2vec baseline: node2vec with p = q = 1 (baseline_common.skipgram: uniform walks, PyG's
Node2Vec loss, no pyg-lib needed) on the static training graph, then a small MLP on
the Hadamard product z_u * z_v trained to tell training edges from uniformly random node pairs.
Validation and test edges are held out; anomaly score = -logit.

    python n2v_test.py --dataset lanl14argus --device 0 --seed 1
'''
import time

import torch
from torch import nn
from torch.optim import Adam

from baseline_common import (Selector, arguments, finish, load_splits, log_epoch, random_pairs,
                             run, skipgram, test_days, test_metrics, val_auc)


class EdgeMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(), nn.Linear(d // 2, 1))

    def forward(self, z, ei):
        return self.net(z[ei[0]] * z[ei[1]]).squeeze(-1)

    def score(self, z, ei, bs=1 << 18):
        return torch.cat([-self(z, e) for e in ei.split(bs, dim=1)])


def train_n2v(args, ei, num_nodes):
    walk_ei = torch.cat([ei, ei.flip(0)], 1) if args.undirected else ei
    return skipgram(walk_ei, num_nodes, dim=args.dim, walk_len=args.walk_len, ctx=args.ctx,
                    epochs=args.n2v_epochs, lr=args.n2v_lr, bs=args.n2v_bs, device=args.device,
                    name='n2v')


def fit(args, tr, va, te, num_nodes, days):
    dev = args.device
    # By default walks run on the simple graph (each pair once), like every other baseline:
    # weighting transitions by flow count makes the heaviest pairs "most normal", which inverts
    # the ranking on UNSW (see euler_test.py --no-dedup). The classifier trains on each distinct
    # training edge once either way.
    simple = tr.ei.unique(dim=1)
    z = train_n2v(args, tr.ei if args.multigraph else simple, num_nodes)
    pos_all = simple.to(dev)

    gen = torch.Generator().manual_seed(0)
    va_pos, va_neg = va.ei.to(dev), random_pairs(len(va), num_nodes, gen, dev)
    te_ei = te.ei.to(dev)

    mlp = EdgeMLP(args.dim).to(dev)
    opt = Adam(mlp.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    sel = Selector(args.patience)
    for e in range(args.epochs):
        t0 = time.time()
        mlp.train()
        tot, nb = 0.0, 0
        for idx in torch.randperm(pos_all.size(1), device=dev).split(args.bs):
            pos = pos_all[:, idx]
            neg = random_pairs(pos.size(1), num_nodes, device=dev)
            logits = torch.cat([mlp(z, pos), mlp(z, neg)])
            labels = torch.cat([torch.ones(pos.size(1)), torch.zeros(neg.size(1))]).to(dev)
            opt.zero_grad()
            loss = bce(logits, labels)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1

        mlp.eval()
        with torch.no_grad():
            va_m = val_auc(mlp.score(z, va_pos), mlp.score(z, va_neg))
            test = test_metrics(te.y, mlp.score(z, te_ei).cpu(), days)
        improved = sel.update(e, va_m, test)
        log_epoch(e, tot / nb, va_m, test, improved, t0)
        if sel.stop:
            break
    return sel.result()


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0])
    ap.add_argument('--dim', type=int, default=128)
    ap.add_argument('--walk-len', type=int, default=500)
    ap.add_argument('--ctx', type=int, default=5)
    ap.add_argument('--n2v-epochs', type=int, default=25)
    ap.add_argument('--n2v-lr', type=float, default=0.01)
    ap.add_argument('--n2v-bs', type=int, default=512, help='walk start nodes per step')
    ap.add_argument('--lr', type=float, default=0.01, help='edge classifier')
    ap.add_argument('--bs', type=int, default=100_000, help='edges per classifier step')
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--undirected', action='store_true', help='walk the undirected training graph')
    ap.add_argument('--multigraph', action='store_true',
                    help='walk the multigraph, repeated edges weighting the transitions (the old '
                         'script\'s behaviour)')
    args = finish(ap.parse_args())

    tr, va, te, n, _ = load_splits(args.dataset, args.data)
    days = test_days(te)
    run(args, 'n2v', lambda: fit(args, tr, va, te, n, days),
        ['dim', 'walk_len', 'ctx', 'n2v_epochs', 'n2v_lr', 'n2v_bs', 'lr', 'bs', 'epochs',
         'patience', 'undirected'])
