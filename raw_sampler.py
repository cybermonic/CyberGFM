'''
Samplers producing walks with raw (continuous) edge features, for models/raw_feat_bert.py.

`rw_feats()` returns two aligned tensors:
    walks: B x S   node ids, FEAT (-3) at feature slots, PAD (-2) where a walk ended
    feats: B x S x F   raw features of the edge in that slot (zeros elsewhere)

One feature slot per hop, so a walk of k nodes is 2k-1 tokens instead of k + (k-1)F.

Backward compatible: these classes subclass RWSampler / TRWSampler and leave rw(), __iter__ and
_single_iter's tokenized behaviour alone unless raw features are requested. Parent rw() still
returns node-only walks (they are constructed with edge_features=False), so anything expecting
the old interface keeps working.
'''
import os

import torch

from models.gnn_bert import GNNEmbedding
from models.raw_feat_bert import FEAT
from sampler import RWSampler, TRWSampler

try:
    from graph_walks import temporal_rw
except ImportError:
    temporal_rw = None

PAD, MASK = GNNEmbedding.PAD, GNNEmbedding.MASK


def load_raw_graph(dataset, split, path=None):
    '''
    Load a split whose edge_attr holds raw feature vectors.
    Tries the usual names for the *_raw.pt files; pass `path` to be explicit.
    '''
    candidates = [path] if path else [
        f'data/{dataset}_tgraph_{split}_raw.pt',
        f'data/{dataset}_{split}_raw.pt',
        f'data/{dataset}_raw_tgraph_{split}.pt',
        f'data/{dataset}-raw_tgraph_{split}.pt',
    ]
    for fn in candidates:
        if fn and os.path.exists(fn):
            return torch.load(fn, weights_only=False)
    raise FileNotFoundError(f'No raw-feature file for {dataset}/{split}; tried: {candidates}')


def raw_edge_attr(data):
    '''Raw per-edge feature matrix (E x F float) from a Data object.'''
    for key in ('edge_attr_raw', 'edge_attr_cont', 'edge_attr'):
        if key in data.keys() and getattr(data, key) is not None:
            x = getattr(data, key)
            if x.dim() == 1:
                x = x.unsqueeze(-1)
            if not x.is_floating_point():
                raise TypeError(f'{key} is {x.dtype}; raw features must be float (use the *_raw.pt files)')
            return x.float()
    raise ValueError('No raw edge features found on this Data object')


class _RawFeatMixin:
    '''Shared logic; see RawFeatRWSampler / RawFeatTRWSampler below.'''

    def __init__(self, data, *args, raw_feats=None, **kwargs):
        kwargs['edge_features'] = False  # parent stays node-only; features come from rw_feats()
        super().__init__(data, *args, **kwargs)

        self.raw = (raw_feats if raw_feats is not None else raw_edge_attr(data)).to(self.device)
        self.feat_dim = self.raw.size(1)
        self.raw_features = True
        self.num_tokens = self.num_nodes  # vocabulary is nodes only
        self._ptr = data.idxptr.to(self.device)
        self._col = data.col.to(self.device)
        self._ts = data.ts.to(self.device)

    def _raw_walk(self, start, walk_len, **kwargs):
        raise NotImplementedError

    def rw_feats(self, start, walk_len=None, reverse=False, trim_missing=True, **kwargs):
        '''Interleaved walk of node ids and FEAT slots, plus the aligned raw feature tensor.'''
        wl = walk_len or self.walk_len
        start = start.to(self.device)
        nodes, eids = self._raw_walk(start, wl, reverse=reverse, **kwargs)
        nodes, eids = nodes.clone(), eids.clone()

        if reverse:
            nodes, eids = nodes.flip(1), eids.flip(1)
        pad = eids == -1
        if reverse:
            nodes[:, :-1][pad] = PAD
        else:
            nodes[:, 1:][pad] = PAD

        B, L = nodes.shape                      # L = wl + 1 nodes, wl edges
        S = 2 * L - 1                           # n0 FEAT n1 FEAT ... n_{L-1}
        walks = torch.full((B, S), PAD, dtype=nodes.dtype, device=nodes.device)
        feats = torch.zeros(B, S, self.feat_dim, device=nodes.device)

        walks[:, 0::2] = nodes
        # eids[:, k] joins nodes k and k+1, so it sits at sequence position 2k+1
        has_edge = ~pad & (nodes[:, :-1] != PAD) & (nodes[:, 1:] != PAD)
        slots = walks[:, 1::2]
        slots[has_edge] = FEAT
        walks[:, 1::2] = slots
        feat_rows = feats[:, 1::2]
        feat_rows[has_edge] = self.raw[eids[has_edge]]
        feats[:, 1::2] = feat_rows

        if trim_missing:
            keep = (walks != PAD).any(dim=0)
            walks, feats = walks[:, keep], feats[:, keep]
        return walks, feats

    def raw_edges(self, idx):
        '''(src, dst, ts, raw features) for the given edge indices — the raw analogue of _single_iter.'''
        idx = idx.to(self.device)
        return (self.data.src[idx.to(self.data.src.device)].to(self.device),
                self._col[idx], self._ts[idx], self.raw[idx])

    def raw_edge_iter(self, batch_size=None, shuffle=True):
        n = self._col.size(0)
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for b in order.split(batch_size or self.batch_size):
            yield (*self.raw_edges(b), b)

    def raw_iter(self, batch_size=None):
        '''Walks for pretraining: one batch of start nodes at a time.'''
        for b in torch.randperm(self.num_nodes).split(batch_size or self.batch_size):
            yield self.rw_feats(b.repeat(self.n_walks))


class RawFeatRWSampler(_RawFeatMixin, RWSampler):
    def _raw_walk(self, start, walk_len, **kwargs):
        return torch.ops.torch_cluster.random_walk(self._ptr, self._col, start, walk_len, 1, 1)


class RawFeatTRWSampler(_RawFeatMixin, TRWSampler):
    def _raw_walk(self, start, walk_len, min_ts=None, max_ts=None, reverse=False, **kwargs):
        if temporal_rw is None:
            raise ImportError('temporal_walks extension not found')
        return temporal_rw(self._ptr, self._col, self._ts, start, walk_len,
                           min_ts=min_ts, max_ts=max_ts, reverse=reverse, return_edge_indices=True)


class RawFeatTokenizer:
    '''
    BERT-style corruption for raw-feature walks: only node positions are masked, since FEAT slots
    carry vectors rather than predictable tokens. Same 80/10/10 split and schedule as RWTokenizer.
    '''

    def __init__(self, num_nodes, fixed_rate=0.7, min_rate=0.15):
        self.num_nodes = int(num_nodes)
        self.fixed_rate = fixed_rate
        self.min_rate = min_rate
        self.mask_rate = fixed_rate

    def set_mask_rate(self, percent_done):
        self.mask_rate = max((1 - min(1.0, percent_done) ** 2) * self.fixed_rate, self.min_rate)
        return self.mask_rate

    def mask(self, walks):
        walks = walks.clone()
        nodes = walks >= 0
        sel = (torch.rand(walks.shape, device=walks.device) < self.mask_rate) & nodes
        targets = walks[sel]

        idx = sel.nonzero()
        perm = torch.randperm(idx.size(0), device=walks.device)
        n = idx.size(0)
        to_mask = idx[perm[:int(n * 0.8)]]
        to_swap = idx[perm[int(n * 0.9):]]
        walks[to_mask[:, 0], to_mask[:, 1]] = MASK
        walks[to_swap[:, 0], to_swap[:, 1]] = torch.randint(
            0, self.num_nodes, (to_swap.size(0),), device=walks.device)
        return walks, sel, targets, walks != PAD


# ---------------------------------------------------------------------------
# Input builders (raw-feature analogues of common.context / common.lp_inputs)
# ---------------------------------------------------------------------------
def raw_context(tr, src, ts, delta, use_walk):
    '''Walk ending at src, clamped to the training graph's end like common.context.'''
    if not use_walk:
        B = src.size(0)
        return src.unsqueeze(-1), torch.zeros(B, 1, tr.feat_dim, device=src.device)

    if not hasattr(tr, '_ts_max'):
        tr._ts_max = int(tr._ts.max())
    max_ts = ts.long().clamp(max=tr._ts_max)
    min_ts = (max_ts - delta).clamp(min=0)
    return tr.rw_feats(src, reverse=True, trim_missing=False, min_ts=min_ts, max_ts=max_ts)


def left_align_raw(walks, feats):
    '''Move PAD to the end of each row, keeping features aligned (see common.left_align).'''
    order = torch.argsort((walks == PAD).int(), dim=1, stable=True)
    return walks.gather(1, order), feats.gather(1, order.unsqueeze(-1).expand_as(feats))


def raw_lp_inputs(tr, src, dst, ts, x_uv, walk_len, delta):
    '''[walk ... src, FEAT(x_uv), MASK] -> (walks, mask, targets, attn, feats).'''
    rw, rf = raw_context(tr, src, ts, delta, use_walk=walk_len > 1)
    B = rw.size(0)
    tail = torch.tensor([FEAT, MASK], dtype=rw.dtype, device=rw.device).repeat(B, 1)
    tail_f = torch.zeros(B, 2, tr.feat_dim, device=rw.device)
    tail_f[:, 0] = x_uv

    walks = torch.cat([rw, tail], dim=1)
    feats = torch.cat([rf, tail_f], dim=1)
    walks, feats = left_align_raw(walks, feats)
    return walks, walks == MASK, dst, walks != PAD, feats


def make_raw_lp_scorer(model, tr, walk_len, delta, score='sigmoid'):
    '''Anomaly score for edges (higher = more anomalous), matching common.make_lp_scorer.'''
    def score_fn(src, dst, ts, x_uv):
        walks, mask, tgt, attn, feats = raw_lp_inputs(tr, src, dst, ts, x_uv, walk_len, delta)
        logits = model.modified_fwd(walks, mask, tgt, attn, feats=feats, return_loss=False).logits[mask]
        idx = torch.arange(dst.size(0), device=dst.device)
        if score == 'nll':
            return -torch.log_softmax(logits.float(), dim=-1)[idx, dst]
        return 1 - torch.sigmoid(logits[idx, dst])
    return score_fn


# ---------------------------------------------------------------------------
# Evaluation (raw-feature analogues of common.score_split / common.evaluate)
# ---------------------------------------------------------------------------
def raw_random_edges(tr, n, generator=None):
    '''Random node pairs with timestamps and raw features drawn from real training edges.'''
    src = torch.randint(0, tr.num_nodes, (n,), generator=generator)
    dst = torch.randint(0, tr.num_nodes, (n,), generator=generator)
    pick = torch.randint(0, tr._col.size(0), (n,), generator=generator).to(tr._ts.device)
    d = tr.device
    return src.to(d), dst.to(d), tr._ts[pick].to(d), tr.raw[pick].to(d)


@torch.no_grad()
def raw_score_split(score_fn, split, tr, bs, desc=''):
    from tqdm import tqdm
    n = split._col.size(0)
    out = torch.empty(n)
    for b in tqdm(torch.arange(n).split(bs), desc=desc, leave=False):
        src, dst, ts, x = [t.to(tr.device) for t in split.raw_edges(b)]
        out[b] = score_fn(src, dst, ts, x).float().cpu()
    return out


@torch.no_grad()
def raw_evaluate(score_fn, tr, va, te, bs, seed=0, recall=0.5):
    '''Same metrics as common.evaluate, on raw-feature samplers.'''
    import numpy as np
    from fast_auc import fast_auc, fast_ap
    from metrics import detection_metrics

    te_pred = raw_score_split(score_fn, te, tr, bs, 'test').numpy()
    te_y = te.label.cpu().numpy()

    tn = raw_score_split(score_fn, va, tr, bs, 'val')
    g = torch.Generator().manual_seed(seed)
    n = va._col.size(0)
    tp = torch.empty(n)
    for b in torch.arange(n).split(bs):
        tp[b] = score_fn(*raw_random_edges(tr, b.numel(), g)).float().cpu()

    va_pred = torch.cat([tp, tn]).numpy()
    va_y = np.concatenate([np.ones(n), np.zeros(n)])

    te_ap = fast_ap(te_y, te_pred)
    out = dict(te_auc=fast_auc(te_y, te_pred), te_ap=te_ap)
    out.update(detection_metrics(te_y, te_pred, te_ap, ts=te._ts.cpu().numpy(), recall=recall))
    out.update(va_auc=fast_auc(va_y, va_pred), va_ap=fast_ap(va_y, va_pred))
    return out


def load_raw_samplers(dataset, temporal, walk_len, batch_size, eval_bs, device='cpu'):
    '''(tr, va, te) raw-feature samplers; va/te stay on CPU like common.load_eval_samplers.'''
    cls = RawFeatTRWSampler if temporal else RawFeatRWSampler
    tr = cls(load_raw_graph(dataset, 'tr'), walk_len=walk_len, batch_size=batch_size, device=device)

    va_data = load_raw_graph(dataset, 'va')
    va = cls(va_data, walk_len=walk_len, batch_size=eval_bs)
    va.label = torch.zeros_like(va._col)

    te_data = load_raw_graph(dataset, 'te')
    te = cls(te_data, walk_len=walk_len, batch_size=eval_bs)
    te.label = te_data.label
    return tr, va, te