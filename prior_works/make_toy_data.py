'''
Write two tiny datasets in CyberGFM's split format for smoke-testing the baselines:
toy (random edge split) and toy-ts (temporal split, attack after the training window).

    python make_toy_data.py --out tmp/toydata
    python gnn_test.py --dataset toy-ts --data tmp/toydata --epochs 2 --device -1
'''
import argparse
import os

import torch
from torch_geometric.data import Data


def save(path, src, dst, ts, n, label=None):
    order = torch.argsort(src * n + dst)
    src, dst, ts = src[order], dst[order], ts[order]
    idxptr = torch.zeros(n + 1, dtype=torch.long)
    idxptr[1:] = torch.bincount(src, minlength=n).cumsum(0)
    g = Data(x=torch.zeros(n, 1), idxptr=idxptr, col=dst, src=src, ts=ts)
    if label is not None:
        g.label = label[order].float()
    torch.save(g, path)


def main(out, n=60, hours=48, per_hour=150, seed=0):
    torch.manual_seed(seed)
    os.makedirs(out, exist_ok=True)
    # a community structure, so real edges are predictable and random pairs are not
    comm = torch.randint(0, 4, (n,))
    src = torch.randint(0, n, (hours * per_hour,))
    dst = torch.stack([torch.nonzero(comm == comm[s]).squeeze(1)[torch.randint(0, int((comm == comm[s]).sum()), (1,))]
                       for s in src]).squeeze(1)
    ts = torch.arange(hours).repeat_interleave(per_hour) * 3600 + torch.randint(0, 3600, (src.numel(),))
    # the attack: cross-community edges from two nodes in the last quarter
    late = hours * 3 // 4
    m = 40
    a_src = torch.randint(0, 2, (m,))
    a_dst = torch.nonzero(comm != comm[0]).squeeze(1)[torch.randint(0, int((comm != comm[0]).sum()), (m,))]
    a_ts = torch.randint(late * 3600, hours * 3600, (m,))
    mal = torch.cat([torch.zeros(src.numel()), torch.ones(m)]).bool()
    src, dst, ts = torch.cat([src, a_src]), torch.cat([dst, a_dst]), torch.cat([ts, a_ts])

    perm = torch.randperm(src.numel())
    tr = torch.zeros(src.numel(), dtype=torch.bool)
    va = torch.zeros_like(tr)
    tr[perm[:int(0.8 * len(perm))]] = True
    va[perm[int(0.8 * len(perm)):int(0.9 * len(perm))]] = True
    tr &= ~mal
    va &= ~mal
    te = ~(tr | va)
    for name, (a, b, c) in {'toy': (tr, va, te),
                            'toy-ts': ((ts < late * 3600) & tr | (ts < late * 3600) & te & ~mal,
                                       (ts < late * 3600) & va,
                                       ts >= late * 3600)}.items():
        for s, mask in zip(('tr', 'va', 'te'), (a, b, c)):
            save(f'{out}/{name}_tgraph_{s}.pt', src[mask], dst[mask], ts[mask], n,
                 mal[mask] if s == 'te' else None)
        print(f'{out}/{name}_tgraph_*.pt: {int(a.sum())} / {int(b.sum())} / {int(c.sum())} edges')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='tmp/toydata')
    main(ap.parse_args().out)
