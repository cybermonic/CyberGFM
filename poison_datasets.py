import torch
from torch_geometric.data import Data

EDGE = 0
TS = 1
EF = 2

def merge(ptr,col,ts, src2,col2,ts2, ef1=None, ef2=None):
    '''
    Merge two graphs and return the resulting csr object
    Assume g1 is much larger than g2
    '''

    # Split existing csr back into its component parts
    split = dict()
    for i in range(ptr.size(0)-1):
        st = ptr[i]; en = ptr[i+1]

        if ef1 is not None:
            split[i] = [
                [col[st:en]],
                [ts[st:en]],
                [ef1[st:en]]
            ]
        else:
            split[i] = [
                [col[st:en]],
                [ts[st:en]]
            ]

    # Append new edges to split
    edited = set()
    for i in range(src2.size(0)):
        src = src2[i].item()
        dst = col2[i:i+1]
        t = ts2[i:i+1]

        shard = split[src]

        if ef2 is not None:
            ef = ef2[i:i+1]
            shard[EF].append(ef)

        shard[EDGE].append(dst)
        shard[TS].append(t)
        edited.add(src)

    # Go back and sort any edited shards
    for src in split:
        if src in edited:
            shard = split[src]

            # Merge into single vec
            shard[EDGE] = torch.cat(shard[EDGE])
            shard[TS] = torch.cat(shard[TS])

            if ef1 is not None:
                shard[EF] = torch.cat(shard[EF])

            new_ts,idx = shard[TS].sort()

            shard[TS] = new_ts
            shard[EDGE] = shard[EDGE][idx]

            if ef1 is not None:
                shard[EF] = shard[EF][idx]
        else:
            split[src][EDGE] = split[src][EDGE][0]
            split[src][TS] = split[src][TS][0]
            if ef1 is not None:
                split[src][EF] = split[src][EF][0]

    # Merge everything back into a single CSR
    idx = [0]
    col = []
    ts = []
    ef = []
    for i in range(ptr.size(0)-1):
        shard = split[i]
        col.append(shard[EDGE])
        ts.append(shard[TS])

        if ef1 is not None:
            ef.append(shard[EF])

        idx.append(idx[-1] + col[-1].size(0))

    # Turn into tensors
    idx = torch.tensor(idx)
    col = torch.cat(col)
    ts = torch.cat(ts)
    if ef1 is not None:
        ef = torch.cat(ef)

    # Generate source tensor
    cnt = idx[1:] - idx[:-1]
    src = torch.arange(cnt.size(0))
    src = src.repeat_interleave(cnt)

    return Data(
        x=torch.arange(idx.size(0)-1).unsqueeze(-1),
        idxptr=idx,
        src=src,
        col=col,
        ts=ts,
        edge_attr=None if ef1 is None else ef
    )


def poison_attr_tgraph(tr,te, percent, fname):
    is_mal = te.label.nonzero().flatten()
    fname = f'data/{fname}_poison.pt'

    # Take some anomalous edges from te and put them in tr
    # Need new edges to be consistant across pretrain and
    # finetune stages, so save permutation
    try:
        poison = torch.load(fname)
    except FileNotFoundError:
        poison = torch.randperm(is_mal.size(0))
        torch.save(poison, fname)

    poison = is_mal[poison[:int(poison.size(0) * percent)]]

    new_col = te.col[poison]
    new_src = te.src[poison]
    new_ef = te.edge_attr[poison]
    new_ts = te.ts[poison]

    return merge(
        tr.idxptr,tr.col,tr.ts,
        new_src,new_col,new_ts,
        ef1=tr.edge_attr, ef2=new_ef
    )

def poison_nonattr_tgraph(tr,te, percent, fname):
    is_mal = te.label.nonzero().flatten()
    fname = f'data/{fname}_poison.pt'

    # Take some anomalous edges from te and put them in tr
    # Need new edges to be consistant across pretrain and
    # finetune stages, so save permutation
    try:
        poison = torch.load(fname)
    except FileNotFoundError:
        poison = torch.randperm(is_mal.size(0))
        torch.save(poison, fname)

    poison = is_mal[poison[:int(poison.size(0) * percent)]]

    new_col = te.col[poison]
    new_src = te.src[poison]
    new_ts = te.ts[poison]

    return merge(
        tr.idxptr,tr.col,tr.ts,
        new_src,new_col,new_ts
    )

def poison(tr,te,has_edge_attrs,percent,fname):
    '''
    Take malicious edges from the test set, and inject them into the training set
    Following the procedure used by SIGL (Han et al., 2021), keep the injected
    edges in the test set even if they've been added to the training set.
    '''
    if has_edge_attrs:
        return poison_attr_tgraph(tr,te, percent, fname)
    else:
        return poison_nonattr_tgraph(tr,te, percent, fname)

if __name__ == '__main__':
    # Test
    tr = torch.load('data/unsw_tgraph_tr.pt', weights_only=False)
    te = torch.load('data/unsw_tgraph_te.pt', weights_only=False)
    g = poison(tr,te, True, 0.25)
    print(tr.col.size(), '->', g.col.size())

    tr = torch.load('data/optc_tgraph_tr.pt', weights_only=False)
    te = torch.load('data/optc_tgraph_te.pt', weights_only=False)
    g = poison(tr,te, False, 0.25)