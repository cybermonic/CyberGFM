'''
Build the control split for the poisoning comparison: identical to lanl14argus-dirtyts except the
malicious edges inside the training window are removed from train and validation.

dirty vs. scrubbed then differ ONLY by those edges — same window, same training-set size, same
test set — so any score difference is attributable to training on unlabeled attack traffic, not to
how much data the model saw.

The dirty split saves labels only on its test file, so malicious positions come from `is_mal` on
data/lanl14argus_tgraph_csr.pt. The train/val window is inferred from the existing dirty files and
checked against them before anything is written.

    python scrub_dirty_split.py
    python scrub_dirty_split.py --dirty lanl14argus-dirtyts --out lanl14argus-cleants
'''
from argparse import ArgumentParser

import torch
from torch_geometric.data import Data


def reindex(idxptr, subset_mask):
    '''New index pointer for a subset of the columns (same helper as temporal_split.py).'''
    cum = torch.cat([torch.zeros(1, dtype=torch.long), subset_mask.long().cumsum(0)])
    return cum[idxptr]


def build(g, mask, label=None):
    new_ptr = reindex(g.idxptr, mask)
    deg = new_ptr[1:] - new_ptr[:-1]
    row = torch.arange(g.x.size(0)).repeat_interleave(deg)
    data = Data(x=g.x, idxptr=new_ptr, col=g.col[mask], src=row, ts=g.ts[mask],
                edge_attr=g.edge_attr[mask])
    if getattr(g, 'edge_weight', None) is not None:
        data.edge_weight = g.edge_weight[mask]
    if label is not None:
        data.label = label[mask]
    return data


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--csr', default='../data/lanl14argus_tgraph_csr.pt')
    ap.add_argument('--dirty', default='lanl14argus-dirtyts', help='existing dirty split to mirror')
    ap.add_argument('--out', default='lanl14argus-cleants')
    args = ap.parse_args()

    g = torch.load(args.csr, weights_only=False)
    dirty = {s: torch.load(f'../data/{args.dirty}_tgraph_{s}.pt', weights_only=False) for s in ('tr', 'va', 'te')}

    # temporal_split.py: tr = ts < 0.9 * non_te, va = [0.9 * non_te, non_te), te = ts >= non_te
    non_te = int(dirty['te'].ts.min())
    tr_ts = non_te * 0.9
    print(f'inferred window: train/val < {non_te} ({non_te / 86400:.2f} days), '
          f'train < {tr_ts:.0f} ({tr_ts / 86400:.2f} days)')

    tr_mask = g.ts < tr_ts
    va_mask = (g.ts >= tr_ts) & (g.ts < non_te)
    te_mask = g.ts >= non_te

    for name, mask in (('tr', tr_mask), ('va', va_mask), ('te', te_mask)):
        want, got = dirty[name].col.size(0), int(mask.sum())
        assert want == got, f'{name}: reconstructed {got} edges, {args.dirty} has {want} — window mismatch'
    print('reconstructed the dirty split exactly; scrubbing now')

    is_mal = torch.zeros(g.col.size(0), dtype=torch.bool)
    is_mal[g.is_mal] = True
    label = is_mal.float()

    n_tr, n_va = int((tr_mask & is_mal).sum()), int((va_mask & is_mal).sum())
    tr_mask &= ~is_mal
    va_mask &= ~is_mal
    print(f'removed {n_tr} malicious edges from train and {n_va} from validation '
          f'({n_tr + n_va} of {int(is_mal.sum())} total)')

    for name, mask in (('tr', tr_mask), ('va', va_mask), ('te', te_mask)):
        data = build(g, mask, label if name == 'te' else None)
        fn = f'../data/{args.out}_tgraph_{name}.pt'
        torch.save(data, fn)
        print(f'wrote {fn}: {data.col.size(0):,} edges'
              + (f', {int(data.label.sum())} malicious' if name == 'te' else ''))

    print(f'\ntest sets are identical to {args.dirty}; train differs by {n_tr} edges '
          f'({n_tr / int(tr_mask.sum()) * 100:.4f}% of training data)')