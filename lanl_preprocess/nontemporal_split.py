'''
Random (non-temporal) tr/va/te split of the LANL Argus CSR graphs.

Pulled from the old repo's partition_attr_tgraph(argus=True / raw_argus=True) and reindex();
the counterpart of temporal_split.py. Run from the repo root, like temporal_split.py:

    python random_split.py              # both: lanl14argus_tgraph_* and lanl14argus_tgraph_raw_*
    python random_split.py --only tok   # just the tokenized one
    python random_split.py --only raw   # just the raw one

Reads   data/lanl14argus_tgraph_csr.pt        -> data/lanl14argus_tgraph_{tr,va,te}.pt
        data/lanl14argus_tgraph_raw_csr.pt    -> data/lanl14argus_tgraph_raw_{tr,va,te}.pt

Split, unchanged from the original: torch.manual_seed(0), one randperm over all edges, 80/10/10,
every malicious edge forced into test. The raw files store the attributes as `raw_edge_attr`
(what the baselines read); the tokenized ones as `edge_attr`. `edge_weight` is carried through
whenever the CSR has it (graphs built by the current argus_to_csr.py), and skipped otherwise.
'''
from argparse import ArgumentParser

import torch
from torch_geometric.data import Data


def reindex(idxptr, subset_mask):
    '''
    Index pointer for the subset of `col` that `subset_mask` selects.

    new_ptr[i] = number of selected edges before row i = cumsum(mask)[idxptr[i]]. Same output as
    the old per-row Python loop, without the loop.
    '''
    cs = torch.cat([torch.zeros(1, dtype=torch.long), subset_mask.long().cumsum(0)])
    return cs[idxptr]


def partition_attr_tgraph(raw=False, data_dir='../data'):
    fname = f'{data_dir}/lanl14argus_tgraph{"_raw" if raw else ""}'
    g = torch.load(f'{fname}_csr.pt', weights_only=False)
    has_ew = getattr(g, 'edge_weight', None) is not None

    torch.manual_seed(0)
    idxs = torch.randperm(g.col.size(0))
    tr_end = int(idxs.size(0) * 0.8)
    va_end = int(idxs.size(0) * 0.9)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    tr[idxs[:tr_end]] = True
    va[idxs[tr_end:va_end]] = True
    te[idxs[va_end:]] = True

    # Mask out anomalies so they're all in test set
    tr[g.is_mal] = False
    va[g.is_mal] = False
    te[g.is_mal] = True

    for mask, name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)

        row = torch.arange(g.x.size(0))
        deg = new_ptr[1:] - new_ptr[:-1]
        row = row.repeat_interleave(deg)

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            src = row,
            col = g.col[mask],
            ts = g.ts[mask],
        )
        if raw:
            data.raw_edge_attr = g.edge_attr[mask]
        else:
            data.edge_attr = g.edge_attr[mask]

        if has_ew:
            data.edge_weight = g.edge_weight[mask]

        if name == 'te':
            label = torch.zeros(mask.size(0))
            label[g.is_mal] = 1
            label = label[mask]
            data.label = label

        torch.save(data, f'{fname}_{name}.pt')
        print(f'{fname}_{name}.pt: {int(mask.sum())} edges'
              f'{"" if has_ew else " (no edge_weight in the CSR)"}')


if __name__ == '__main__':
    ap = ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--only', choices=['tok', 'raw'], help='build just one of the two')
    ap.add_argument('--data-dir', default='../data')
    args = ap.parse_args()

    if args.only != 'raw':
        partition_attr_tgraph(raw=False, data_dir=args.data_dir)
    if args.only != 'tok':
        partition_attr_tgraph(raw=True, data_dir=args.data_dir)