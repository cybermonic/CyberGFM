'''
PIKACHU baseline (Paudel & Huang, 2022), reimplemented in PyTorch from their Keras code:

  1. a skip-gram embedding of every snapshot from uniform random walks on its undirected graph
     (their CTDNE / word2vec step: 128 dims, walk length 500, window 5, 5 epochs);
  2. a two-layer GRU autoencoder over each node's sequence of snapshot embeddings, trained on
     the training window; its encoder output is the node's temporal embedding z_t;
  3. an anomaly detector: z_u averaged with `--s` sampled neighbours of u in snapshot t,
     then a linear layer and softmax over all nodes predicts u's neighbours. The anomaly score
     of (u, v) is ((1 - p_u[v]) + (1 - p_v[u])) / 2.

For temporal splits, steps 1-2 also embed the snapshots after the training window from the
observed val/test graph (the GRU runs on); losses only use the training window.

    python pikachu_test.py --dataset lanl14argus --device 0 --seed 1
'''
import time

import torch
from torch import nn
from torch.optim import Adam

from baseline_common import (Selector, Timeline, arguments, csr, finish, load_splits, log_epoch,
                             random_pairs, run, skipgram, test_days, test_metrics, to_undirected,
                             val_auc)

# ---- step 1: skip-gram per snapshot (baseline_common.skipgram) --------------------------------
def embed_snapshot(ei, num_nodes, args):
    # to_undirected also de-duplicates: walks run on a simple graph, as their networkx-based
    # code did (and as Euler needs on UNSW, see euler_test.py --no-dedup)
    return skipgram(to_undirected(ei), num_nodes, dim=args.w2v_dim, walk_len=args.walk_len,
                    ctx=args.ctx, epochs=args.w2v_epochs, lr=args.w2v_lr, bs=args.w2v_bs,
                    device=args.device)


# ---- step 2: GRU autoencoder -------------------------------------------------------------------
class GRUStack(nn.Module):
    def __init__(self, i, h, o):
        super().__init__()
        self.g1, self.drop, self.g2 = nn.GRU(i, h), nn.Dropout(0.3), nn.GRU(h, o)

    def forward(self, x):                                 # (T, N, d), time-major
        return self.g2(self.drop(self.g1(x)[0]))[0]


class Autoencoder(nn.Module):
    def __init__(self, d, hidden, latent):
        super().__init__()
        self.enc, self.dec = GRUStack(d, hidden, latent), GRUStack(latent, hidden, d)

    def loss(self, x):
        return nn.functional.mse_loss(self.dec(self.enc(x)), x)


def temporal_embeddings(embs, n_train, args):
    '''embs (T, N, d) -> z (T, N, latent); trained on the first n_train snapshots.'''
    dev = args.device
    ae = Autoencoder(embs.size(-1), args.ae_hidden, args.ae_latent).to(dev)
    opt = Adam(ae.parameters(), lr=args.ae_lr)
    N = embs.size(1)
    for e in range(args.ae_epochs):
        ae.train()
        tot = 0
        for nodes in torch.randperm(N).split(args.ae_bs):
            x = embs[:n_train, nodes].to(dev)
            opt.zero_grad()
            loss = ae.loss(x)
            loss.backward()
            opt.step()
            tot += loss.item() * nodes.numel()
        if e % 10 == 0 or e == args.ae_epochs - 1:
            print(f'  ae [{e}] mse {tot / N:.5f}', flush=True)
    ae.eval()
    with torch.no_grad():
        return torch.cat([ae.enc(embs[:, nodes].to(dev)).cpu()
                          for nodes in torch.arange(N).split(args.ae_bs)], 1)


# ---- step 3: anomaly detector ------------------------------------------------------------------
class Detector(nn.Module):
    def __init__(self, latent, num_nodes, s):
        super().__init__()
        self.lin = nn.Linear(latent, num_nodes, bias=False)
        self.s = s

    def aggregate(self, z, rowptr, col):
        '''(z_u + sum of s neighbour samples) / (s + 1); a node with no neighbours uses itself.'''
        n = z.size(0)
        deg = (rowptr[1:] - rowptr[:-1]).unsqueeze(1)
        off = (torch.rand(n, self.s, device=z.device) * deg).long()
        idx = (rowptr[:-1].unsqueeze(1) + off).clamp(max=max(col.numel() - 1, 0))
        self_ = torch.arange(n, device=z.device).unsqueeze(1).expand(-1, self.s)
        nb = torch.where(deg > 0, col[idx] if col.numel() else self_, self_)
        return (z[nb].sum(1) + z) / (self.s + 1)

    def logits(self, a, nodes):
        return self.lin(a[nodes])                         # softmax lives in the loss / score

    def score(self, a, ei, bs=4096):
        out = []
        for e in ei.split(bs, dim=1):
            pu = torch.softmax(self.logits(a, e[0]), 1).gather(1, e[1:2].T).squeeze(1)
            pv = torch.softmax(self.logits(a, e[1]), 1).gather(1, e[0:1].T).squeeze(1)
            out.append(((1 - pu) + (1 - pv)) / 2)
        return torch.cat(out) if out else torch.zeros(0, device=a.device)


def fit(args, tl, num_nodes, days):
    dev = args.device
    t0 = time.time()
    embs = torch.stack([embed_snapshot(tl.graph[t], num_nodes, args).cpu()
                        for t in range(len(tl))])
    print(f'skip-gram embeddings: {len(tl)} snapshots in {time.time() - t0:.0f}s', flush=True)
    z = temporal_embeddings(embs, tl.n_train, args)
    del embs

    graphs = []
    for t in range(len(tl)):
        und = to_undirected(tl.graph[t])          # unique pairs: the detector's targets
        rowptr, col = csr(und, num_nodes)
        graphs.append((rowptr.to(dev), col.to(dev), und))
    gen = torch.Generator().manual_seed(0)
    va_neg = [random_pairs(e.size(1), num_nodes, gen) for e in tl.va]
    te_y = torch.cat(tl.y)

    det = Detector(args.ae_latent, num_nodes, args.s).to(dev)
    opt = Adam(det.parameters(), lr=args.anom_lr)
    ce = nn.CrossEntropyLoss()
    sel = Selector(args.patience)
    train_t = [t for t in range(tl.n_train) if tl.graph[t].size(1)]

    for e in range(args.anom_epochs):
        t0 = time.time()
        det.train()
        opt.zero_grad()
        losses = []
        for k, t in enumerate(torch.randperm(len(train_t)).tolist()):
            t = train_t[t]
            rowptr, col, und = graphs[t]
            a = det.aggregate(z[t].to(dev), rowptr, col)
            und = und.to(dev)
            loss = sum(ce(det.logits(a, p[0]), p[1]) * p.size(1)
                       for p in und.split(args.edge_bs, dim=1)) / und.size(1)
            (loss / args.accum).backward()
            losses.append(loss.item())
            if (k + 1) % args.accum == 0:
                opt.step()
                opt.zero_grad()
        if len(train_t) % args.accum:                     # the leftover partial group
            opt.step()

        det.eval()
        vp, vn, ts = [], [], []
        with torch.no_grad():
            for t in range(len(tl)):
                if not (tl.va[t].size(1) or tl.te[t].size(1)):
                    continue
                rowptr, col, _ = graphs[t]
                a = det.aggregate(z[t].to(dev), rowptr, col)
                if tl.va[t].size(1):
                    vp.append(det.score(a, tl.va[t].to(dev)).cpu())
                    vn.append(det.score(a, va_neg[t].to(dev)).cpu())
                ts.append(det.score(a, tl.te[t].to(dev)).cpu())
        va_m = val_auc(torch.cat(vp), torch.cat(vn))
        test = test_metrics(te_y, torch.cat(ts), days)
        improved = sel.update(e, va_m, test)
        log_epoch(e, sum(losses) / max(1, len(losses)), va_m, test, improved, t0)
        if sel.stop:
            break
    return sel.result()


if __name__ == '__main__':
    ap = arguments(__doc__.split('\n\n')[0])
    g = ap.add_argument_group('skip-gram (their defaults)')
    g.add_argument('--w2v-dim', type=int, default=128)
    g.add_argument('--walk-len', type=int, default=500)
    g.add_argument('--ctx', type=int, default=5)
    g.add_argument('--w2v-epochs', type=int, default=5)
    g.add_argument('--w2v-lr', type=float, default=0.025)
    g.add_argument('--w2v-bs', type=int, default=2048, help='walk start nodes per step')
    g = ap.add_argument_group('autoencoder')
    g.add_argument('--ae-hidden', type=int, default=128)
    g.add_argument('--ae-latent', type=int, default=64)
    g.add_argument('--ae-epochs', type=int, default=50)
    g.add_argument('--ae-lr', type=float, default=1e-3)
    g.add_argument('--ae-bs', type=int, default=2048, help='nodes per step')
    g = ap.add_argument_group('anomaly detector')
    g.add_argument('--s', type=int, default=10, help='neighbour samples per node')
    g.add_argument('--anom-epochs', type=int, default=10)
    g.add_argument('--anom-lr', type=float, default=1e-3)
    g.add_argument('--accum', type=int, default=5, help='snapshots per optimizer step')
    g.add_argument('--edge-bs', type=int, default=4096)
    g.add_argument('--patience', type=int, default=0, help='0 = run all --anom-epochs')
    args = finish(ap.parse_args())

    tr, va, te, n, temporal = load_splits(args.dataset, args.data)
    tl = Timeline(tr, va, te, args.ts_div, temporal)
    days = test_days(te)
    run(args, 'pikachu', lambda: fit(args, tl, n, days),
        ['w2v_dim', 'walk_len', 'ctx', 'w2v_epochs', 'w2v_lr', 'w2v_bs', 'ae_hidden', 'ae_latent',
         'ae_epochs', 'ae_lr', 's', 'anom_epochs', 'anom_lr', 'accum', 'ts_div'])
