import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, remove_self_loops
from graph_walks import temporal_rw, rw as random_walk
from joblib import Parallel, delayed

from models.gnn_bert import GNNEmbedding

import os
LEGACY_EF_OFFSET = os.environ.get('LEGACY_EF_OFFSET') == '1'

class TRWSampler():
    def __init__(self, data: Data, walk_len=64, n_walks=1, batch_size=64, device='cpu', edge_features=False):
        self.x = data.x
        self.rowptr = data.idxptr.to(device)
        self.col = data.col.to(device)
        self.ts = data.ts.to(device)
        self.data = data

        self.num_nodes = data.x.size(0)

        ea = data.get('edge_attr', None)
        self.edge_attr = ea.to(device) if ea is not None else None

        if 'num_tokens' in data.keys():
            self.num_tokens = data.num_tokens

        else:
            self.num_tokens = data.x.size(0)
            if edge_features:
                self.num_tokens += self.edge_attr.max() + 1

        # Used for KG LP
        if 'filter_ptr' in data.keys():
            self.filter_ptr = data.filter_ptr
            self.filter_col = data.filter_col
            self.filter_rel = data.filter_rel

        self.edge_features = edge_features
        self.walk_len = walk_len
        self.n_walks = n_walks
        self.batch_size = batch_size
        self.device = device
        self.trim_missing = False
        self._inv_cache = None

        self.min_ts = None
        self.max_ts = None

    def to(self, device):
        self.rowptr = self.rowptr.to(device)
        self.col = self.col.to(device)
        self.ts = self.ts.to(device)

        # Will get pulled from the underlying data object
        # and moved to the active device next time it's used
        self._inv_cache = None

        if self.edge_features:
            self.edge_attr = self.edge_attr.to(device)

        self.device = device
        torch.cuda.empty_cache()

    def buggy_rw(self, batch, n_walks=1, min_ts=None, max_ts=None, reverse=False, trim_missing=True, walk_len=None):
        if walk_len is not None:
            wl = walk_len
        else:
            wl = self.walk_len

        batch = batch.repeat(n_walks)

        walks,eids = temporal_rw(
            self.rowptr, self.col, self.ts, batch.to(self.device),
            wl, min_ts=min_ts, max_ts=max_ts, reverse=reverse, return_edge_indices=True
        )

        if reverse:
            walks = walks.flip(1)
            eids = eids.flip(1)

        pad = eids == -1

        if not reverse:
            walks[:, 1:][pad] = GNNEmbedding.PAD
        else:
            walks[:, :-1][pad] = GNNEmbedding.PAD

        if self.edge_features:
            edge_feats = self.edge_attr[eids] + self.num_nodes
            edge_feats[pad] = GNNEmbedding.PAD
            edge_feats = torch.cat([
                edge_feats,
                torch.full((edge_feats.size(0),1,edge_feats.size(2)), GNNEmbedding.PAD, device=edge_feats.device)
            ], dim=1)

            # Interleave nids and eids
            walks = walks.unsqueeze(-1)
            walks = torch.cat([walks, edge_feats], dim=-1).view(walks.size(0), -1)

        pad = walks == GNNEmbedding.PAD
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_row = ~torch.prod(pad, dim=1, dtype=torch.bool)

        # If no walks went to full walk_len, trim them down to save mem
        if trim_missing:
            walks = walks[:, whole_col]
            walks = walks[whole_row]

        return walks

    def rw(self, batch, n_walks=1, min_ts=None, max_ts=None, reverse=False, trim_missing=True, walk_len=None):
        '''
        I realized that the reverse argument passed to temporal_rw just made the time stamps reversed, and
        didn't reverse the direction of the walks, so I fixed that here. Kept the old method bc it performs
        better on lanl (0.29 vs 0.20) but otherwise, nothing else improves. Weird.
        '''
        if walk_len is not None:
            wl = walk_len
        else:
            wl = self.walk_len

        batch = batch.repeat(n_walks)


        if reverse:
            inv = self._get_inv()
            rowptr, col, ts = inv['idxptr'], inv['col'], inv['ts']
        else:
            rowptr, col, ts = self.rowptr, self.col, self.ts

        walks,eids = temporal_rw(
            rowptr, col, ts, batch.to(self.device),
            wl, min_ts=min_ts, max_ts=max_ts, reverse=reverse, return_edge_indices=True
        )

        if reverse:
            eids = torch.where(eids >= 0, inv['perm'][eids.clamp(min=0)], eids)  # -> original edge ids
            walks = walks.flip(1)
            eids = eids.flip(1)

        pad = eids == -1

        if not reverse:
            walks[:, 1:][pad] = GNNEmbedding.PAD
        else:
            walks[:, :-1][pad] = GNNEmbedding.PAD

        if self.edge_features:
            edge_feats = self.edge_attr[eids] + self.num_nodes
            edge_feats[pad] = GNNEmbedding.PAD
            edge_feats = torch.cat([
                edge_feats,
                torch.full((edge_feats.size(0),1,edge_feats.size(2)), GNNEmbedding.PAD, device=edge_feats.device)
            ], dim=1)

            # Interleave nids and eids
            walks = walks.unsqueeze(-1)
            walks = torch.cat([walks, edge_feats], dim=-1).view(walks.size(0), -1)

        pad = walks == GNNEmbedding.PAD
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_row = ~torch.prod(pad, dim=1, dtype=torch.bool)

        # If no walks went to full walk_len, trim them down to save mem
        if trim_missing:
            walks = walks[:, whole_col]
            walks = walks[whole_row]

        return walks

    def __iter__(self):
        batches = torch.randperm(self.num_nodes).split(self.batch_size)
        for b in batches:
            yield self.rw(b, min_ts=self.min_ts, max_ts=self.max_ts, n_walks=self.n_walks)

    def _single_iter(self, b, shuffled=True):
        src = self.data.src[b.to(self.data.src.device)].to(self.device)
        dst = self.col[b]
        ts = self.ts[b]

        if self.edge_features:
            offset = 0 if LEGACY_EF_OFFSET else self.num_nodes
            return src, dst, ts, self.edge_attr[b] + offset

        return src, dst, ts

    def _get_inv(self):
        '''
        In-edge CSR for walking against edge direction, stored on the Data object as
        self.data.inv and built on first use:
            inv['col'][inv['idxptr'][v]:inv['idxptr'][v+1]]  sources of edges into v, sorted by ts
            inv['ts']                                        their timestamps
            inv['perm']                                      position in this order -> original edge id
        Rebuilt if the graph's edges changed since it was built (e.g. subsetting).
        Returns a copy on this sampler's device, cached until to() is called.
        '''
        if getattr(self, '_inv_cache', None) is not None:
            return self._inv_cache

        data = self.data
        inv = data.inv if 'inv' in data.keys() else None
        if inv is None or inv['col'].numel() != data.col.numel():
            idxptr, col, ts = data.idxptr.cpu(), data.col.cpu(), data.ts.cpu()
            n = idxptr.size(0) - 1
            if 'src' in data.keys():
                src = data.src.cpu()
            else:
                src = torch.arange(n).repeat_interleave(idxptr[1:] - idxptr[:-1])

            perm = torch.argsort(ts, stable=True)
            perm = perm[torch.argsort(col[perm], stable=True)]  # primary key dst, then ts
            inv_ptr = torch.zeros(n + 1, dtype=torch.long)
            inv_ptr[1:] = torch.bincount(col, minlength=n).cumsum(0)
            inv = dict(idxptr=inv_ptr, col=src[perm], ts=ts[perm], perm=perm)
            data.inv = inv

        self._inv_cache = {k: v.to(self.device) for k, v in inv.items()}
        return self._inv_cache

    def edge_iter(self, shuffle=True, return_index=False):
        if shuffle:
            batches = torch.randperm(self.col.size(0), device=self.device).split(self.batch_size)
        else:
            batches = torch.arange(self.col.size(0), device=self.device).split(self.batch_size)

        for b in batches:
            samp = self._single_iter(b, shuffled=shuffle)

            if return_index:
                yield *samp,b
            else:
                yield samp

    def parallel_edge_iter(self, n_jobs=32, shuffle=True, return_index=False):
        if shuffle:
            batches = torch.randperm(self.col.size(0), device=self.device).split(self.batch_size)
        else:
            batches = torch.arange(self.col.size(0), device=self.device).split(self.batch_size)

        def thread_job(b):
            if shuffle:
                b = b.sort().values

            dst = self.col[b]
            ts = self.ts[b]

            # Worst case O(|V| + |b|) -> O(n)
            src = []
            cur_src = find_src(b[0], self.rowptr)
            cur_max = self.rowptr[cur_src+1]

            for b_ in b:
                while cur_max < b_:
                    cur_src += 1
                    cur_max  = self.rowptr[cur_src+1]
                src.append(cur_src)

            src = torch.tensor(src, device=self.device)
            return src,dst,ts

        for b in batches:
            # Single thread: 13-14 it/s
            # 32: 10, 16: 8, 8: 7    :(
            outs = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(thread_job)(b_) for b_ in b.split(n_jobs)
            )
            src,dst,ts = zip(*outs)
            src = torch.cat(src); dst = torch.cat(dst); ts = torch.cat(ts)

            if return_index:
                yield src,dst,ts,b
            else:
                yield src,dst,ts

    def add_edge_index(self):
        if not hasattr(self.data, 'edge_index'):
            src = torch.arange(self.rowptr.size(0)-1, device=self.col.device)
            deg = self.rowptr[1:] - self.rowptr[:-1]
            src = src.repeat_interleave(deg)

            ei = torch.stack([src, self.col])
            ei,ew = ei.unique(dim=1, return_counts=True)
            self.edge_index = ei
        else:
            self.edge_index = self.data.edge_index.to(self.col.device)

def find_src(col_idx, idxptr):
    st = 0
    en = idxptr.size(0)-1
    while (en-st > 1):
        mid = st + ((en-st) // 2)
        if idxptr[mid] > col_idx:
            en = mid
        else:
            st = mid
    if idxptr[st] > col_idx:
        return st-1
    else:
        return st


class RWSampler(TRWSampler):
    def buggy_rw(self, batch, n_walks=1, trim_missing=True, walk_len=None, reverse=False, **kwargs):
        if walk_len is not None:
            wl = walk_len
        else:
            wl = self.walk_len

        batch = batch.repeat(n_walks)

        walks,eids = random_walk(
            self.rowptr, self.col, batch.to(self.device),
            wl, 1, 1
        )

        if reverse:
            walks = walks.flip(1)
            eids = eids.flip(1)

        pad = eids == -1

        if not reverse:
            walks[:, 1:][pad] = GNNEmbedding.PAD
        else:
            walks[:, :-1][pad] = GNNEmbedding.PAD

        if self.edge_features:
            edge_feats = self.edge_attr[eids] + self.num_nodes
            edge_feats[pad] = GNNEmbedding.PAD
            edge_feats = torch.cat([
                edge_feats,
                torch.full((edge_feats.size(0),1,edge_feats.size(2)), GNNEmbedding.PAD, device=edge_feats.device)
            ], dim=1)

            # Interleave nids and eids
            walks = walks.unsqueeze(-1)
            walks = torch.cat([walks, edge_feats], dim=-1).view(walks.size(0), -1)

        pad = walks == GNNEmbedding.PAD
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_row = ~torch.prod(pad, dim=1, dtype=torch.bool)

        # If no walks went to full walk_len, trim them down to save mem
        if trim_missing:
            walks = walks[:, whole_col]
            walks = walks[whole_row]

        return walks

    def rw(self, batch, n_walks=1, trim_missing=True, walk_len=None, reverse=False, **kwargs):
        if walk_len is not None:
            wl = walk_len
        else:
            wl = self.walk_len

        batch = batch.repeat(n_walks)

        if reverse:
            inv = self._get_inv()
            rowptr, col = inv['idxptr'], inv['col']
        else:
            rowptr, col = self.rowptr, self.col

        walks,eids = random_walk(
            rowptr, col, batch.to(self.device),
            wl, 1, 1
        )

        if reverse:
            eids = torch.where(eids >= 0, inv['perm'][eids.clamp(min=0)], eids)
            walks = walks.flip(1)
            eids = eids.flip(1)

        pad = eids == -1

        if not reverse:
            walks[:, 1:][pad] = GNNEmbedding.PAD
        else:
            walks[:, :-1][pad] = GNNEmbedding.PAD

        if self.edge_features:
            edge_feats = self.edge_attr[eids] + self.num_nodes
            edge_feats[pad] = GNNEmbedding.PAD
            edge_feats = torch.cat([
                edge_feats,
                torch.full((edge_feats.size(0),1,edge_feats.size(2)), GNNEmbedding.PAD, device=edge_feats.device)
            ], dim=1)

            # Interleave nids and eids
            walks = walks.unsqueeze(-1)
            walks = torch.cat([walks, edge_feats], dim=-1).view(walks.size(0), -1)

        pad = walks == GNNEmbedding.PAD
        whole_col = ~torch.prod(pad, dim=0, dtype=torch.bool)
        whole_row = ~torch.prod(pad, dim=1, dtype=torch.bool)

        # If no walks went to full walk_len, trim them down to save mem
        if trim_missing:
            walks = walks[:, whole_col]
            walks = walks[whole_row]

        return walks
