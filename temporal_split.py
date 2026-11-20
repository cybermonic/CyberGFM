import torch
from torch_geometric.data import Data

FIRST_RED = 147600
def temporally_partition_lanl_tgraph():
    torch.manual_seed(0)

    g = torch.load('data/lanl14argus_tgraph_csr.pt', weights_only=False)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    pre_attack = g.ts < FIRST_RED
    potential = pre_attack.nonzero().squeeze(-1)
    tr_idx = int(potential.size(0) * 0.9)
    idx = torch.randperm(potential.size(0))

    # Train/Val on pre-attack
    tr[potential[idx[:tr_idx]]] = True
    va[potential[idx[tr_idx:]]] = True

    # Test on everything post attack
    te[~pre_attack] = True

    # Mask out anomalies so they're all in test set
    tr[g.is_mal] = False
    va[g.is_mal] = False
    te[g.is_mal] = True

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)

        row = torch.arange(g.x.size(0))
        deg = new_ptr[1:] - new_ptr[:-1]
        row = row.repeat_interleave(deg)

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            src = row,
            ts = g.ts[mask],
            edge_attr = g.edge_attr[mask]
        )
        if getattr(g, 'edge_weight', None) is not None:
            data.edge_weight = g.edge_weight[mask]

        if name == 'te':
            label = torch.zeros(mask.size(0))
            label[g.is_mal] = 1
            label = label[mask]
            data.label = label

        fn = f'data/lanl14argus-ts_tgraph_{name}.pt'
        torch.save(data, fn)

        print(f'{fn}: {data.col.size(0)} edges')

def temporally_split_dirty_lanl_tgraph(tr_days=7):
    torch.manual_seed(0)
    non_te = 3600 * 24 * tr_days
    tr_ts = non_te * 0.9

    g = torch.load('data/lanl14argus_tgraph_csr.pt', weights_only=False)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    tr_idx = g.ts < tr_ts
    va_idx = (g.ts < non_te).logical_and(g.ts >= tr_ts)

    # Train/Val on earlier temporal split
    # (will contain unlabeled malicious edges!)
    tr[tr_idx] = True
    va[va_idx] = True

    # Test on everything post temporal split
    te[g.ts >= non_te] = True

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)

        row = torch.arange(g.x.size(0))
        deg = new_ptr[1:] - new_ptr[:-1]
        row = row.repeat_interleave(deg)

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            src = row,
            ts = g.ts[mask],
            edge_attr = g.edge_attr[mask]
        )
        if getattr(g, 'edge_weight', None) is not None:
            data.edge_weight = g.edge_weight[mask]

        if name == 'te':
            label = torch.zeros(mask.size(0))
            label[g.is_mal] = 1
            label = label[mask]
            data.label = label

        fn = f'data/lanl14argus-dirtyts_tgraph_{name}.pt'
        torch.save(data, fn)

        print(f'{fn}: {data.col.size(0)} edges')

# Generate new index pointer for subset of column that was selected
def reindex(idxptr, subset_mask):
    cum = torch.cat([torch.zeros(1, dtype=torch.long),
                     subset_mask.long().cumsum(0)])
    return cum[idxptr]

OPTC_FIRST_RED = 579776
def temporally_partition_optc_tgraph():
    torch.manual_seed(1)

    g = torch.load('data/optc_tgraph_csr.pt', weights_only=False)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    pre_attack = g.ts < OPTC_FIRST_RED
    potential = pre_attack.nonzero().squeeze(-1)
    tr_idx = int(potential.size(0) * 0.9)
    idx = torch.randperm(potential.size(0))

    # Train/Val on pre-attack
    tr[potential[idx[:tr_idx]]] = True
    va[potential[idx[tr_idx:]]] = True

    # Test on everything post attack
    te[~pre_attack] = True

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)

        row = torch.arange(g.x.size(0))
        deg = new_ptr[1:] - new_ptr[:-1]
        row = row.repeat_interleave(deg)

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            src = row,
            ts = g.ts[mask],
            edge_attr = g.edge_attr[mask],
            label = g.label[mask]
        )

        torch.save(data, f'data/optc-ts_tgraph_{name}.pt')

OPTC_FIRST_RED = 573290
def temporally_partition_argus_optc_tgraph():
    torch.manual_seed(1)

    g = torch.load('data/optc-argus_tgraph_csr.pt', weights_only=False)

    tr = torch.zeros(g.col.size(0), dtype=torch.bool)
    va = torch.zeros_like(tr)
    te = torch.zeros_like(tr)

    pre_attack = g.ts < OPTC_FIRST_RED
    potential = pre_attack.nonzero().squeeze(-1)
    tr_idx = int(potential.size(0) * 0.9)
    idx = torch.randperm(potential.size(0))

    # Train/Val on pre-attack
    tr[potential[idx[:tr_idx]]] = True
    va[potential[idx[tr_idx:]]] = True

    # Test on everything post attack
    te[~pre_attack] = True

    for mask,name in [(tr, 'tr'), (va, 'va'), (te, 'te')]:
        new_ptr = reindex(g.idxptr, mask)

        row = torch.arange(g.x.size(0))
        deg = new_ptr[1:] - new_ptr[:-1]
        row = row.repeat_interleave(deg)

        data = Data(
            x = g.x,
            idxptr = new_ptr,
            col = g.col[mask],
            src = row,
            ts = g.ts[mask],
            edge_attr = g.edge_attr[mask],
            label = g.label[mask]
        )

        torch.save(data, f'data/optc-argus_tgraph_{name}.pt')


if __name__ == '__main__':
    temporally_split_dirty_lanl_tgraph(tr_days=7)
    temporally_partition_lanl_tgraph()