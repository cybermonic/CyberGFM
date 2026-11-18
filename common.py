'''
Shared pieces for pretrain.py, lp_finetune.py and cls_finetune.py:
dataset defaults, training-graph loading (poisoning / subsets), random-walk
context construction, random negative edges, evaluation and result tracking.
'''
import atexit
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertConfig

from fast_auc import fast_auc, fast_ap
from metrics import detection_metrics
from models.gnn_bert import GNNEmbedding
from poison_datasets import poison
from sampler import TRWSampler, RWSampler
from torch_geometric.data import Data
from utils import reindex

PAD, MASK, OFFSET = GNNEmbedding.PAD, GNNEmbedding.MASK, GNNEmbedding.OFFSET
HOUR, DAY = 60 * 60, 60 * 60 * 24

SIZES = {
    'tiny': SimpleNamespace(H=128, L=2),
    'mini': SimpleNamespace(H=256, L=4),
    'small': SimpleNamespace(H=512, L=4),
    'med': SimpleNamespace(H=512, L=8),
    'baseline': SimpleNamespace(H=768, L=12),
}


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
def dataset_kind(name):
    '''lanl14argus, lanl14argus-dirtyts, ... -> lanl; unsw -> unsw; optc, optc-ts -> optc'''
    for kind in ('lanl', 'unsw', 'optc'):
        if name.startswith(kind):
            return kind
    raise ValueError(f'Unrecognized dataset "{name}" (expected a lanl*, unsw*, or optc* prefix)')


def uses_edge_features(name, ignore=False):
    return dataset_kind(name) in ('lanl', 'unsw') and not ignore


def load_split(name, split):
    return torch.load(f'data/{name}_tgraph_{split}.pt', weights_only=False)


def run_tag(poison_pct=0, tr_size=1.0):
    tag = ''
    if poison_pct:
        tag += f'_{poison_pct}poisoned'
    if tr_size != 1:
        tag += f'_{tr_size:g}data'
    return tag


def pretrained_path(name, size, temporal, poison_pct=0, tr_size=1.0, best=False, tag=''):
    '''tag distinguishes pretraining variants of the same dataset/size, e.g. "_wl32".'''
    kind = 'temporal' if temporal else 'static'
    prefix = 'trw_bert' if temporal else 'rw_bert'
    return (f'pretrained/{kind}/{name}/{prefix}_{name}{run_tag(poison_pct, tr_size)}{tag}_{size}'
            f'{"-best" if best else ""}.pt')


def load_train_graph(name, edge_features, poison_pct=0, tr_size=1.0):
    '''Training graph with optional poisoning (Sec. 4.7) and subsetting (Fig. 7).'''
    tr = load_split(name, 'tr')

    if poison_pct:
        if not 0 < poison_pct <= 100:
            raise ValueError('--poison must be in (0, 100]')
        num_tokens = tr.get('num_tokens', None)
        te = load_split(name, 'te')
        tr = poison(tr.cpu(), te.cpu(), edge_features, poison_pct / 100, name)
        if num_tokens is not None:
            tr.num_tokens = num_tokens
        print(f'Injected {poison_pct}% of malicious test edges into training')

    if tr_size != 1:
        # Keyed by edge count so different poisoning levels don't share an incompatible permutation
        os.makedirs('subsets', exist_ok=True)
        fn = f'subsets/{name}_{tr.col.size(0)}.pt'
        if os.path.exists(fn):
            perm = torch.load(fn, weights_only=True)
        else:
            perm = torch.randperm(tr.col.size(0))
            torch.save(perm, fn)

        keep = torch.zeros(tr.col.size(0), dtype=torch.bool)
        keep[perm[:int(perm.size(0) * tr_size)]] = True
        tr.col, tr.src, tr.ts = tr.col[keep], tr.src[keep], tr.ts[keep]
        if tr.get('edge_attr', None) is not None:
            tr.edge_attr = tr.edge_attr[keep]
        tr.idxptr = reindex(tr.src, tr.x.size(0))
        print(f'Using {tr_size:.1%} of training edges ({tr.col.size(0)})')

    return tr


def undirected_walk_graph(data):
    '''
    The graph random walks move on under --undirected-walks: every edge (u, v, t, features) can
    also be walked as (v, u, t, features), so a walk can leave a node along its in-edges as well
    as its out-edges. A self-loop is its own reverse and is not added twice.

    Returns a new CSR graph (rows sorted by timestamp, which the temporal walk requires) and
    leaves `data` untouched. Only walks use it: the edges that are fine-tuned on and scored
    still come from the directed graphs.
    '''
    n = data.x.size(0)
    idxptr, col, ts = data.idxptr.cpu(), data.col.cpu(), data.ts.cpu()
    src = torch.arange(n).repeat_interleave(idxptr[1:] - idxptr[:-1])
    rev = src != col

    s, d, t = torch.cat([src, col[rev]]), torch.cat([col, src[rev]]), torch.cat([ts, ts[rev]])
    order = torch.argsort(t, stable=True)
    order = order[torch.argsort(s[order], stable=True)]       # by source, then by time

    ptr = torch.zeros(n + 1, dtype=torch.long)
    ptr[1:] = torch.bincount(s, minlength=n).cumsum(0)
    out = Data(x=data.x, idxptr=ptr, col=d[order], src=s[order], ts=t[order])
    ea = data.get('edge_attr', None)
    if ea is not None:
        ea = ea.cpu()
        out.edge_attr = torch.cat([ea, ea[rev]])[order]
    if 'num_tokens' in data.keys():
        out.num_tokens = data.num_tokens
    return out


def make_sampler(data, temporal, edge_features, walk_len, batch_size, device='cpu', undirected_walks=False):
    '''
    undirected_walks: sample every random walk (pretraining walks and fine-tuning / evaluation
    context) on undirected_walk_graph(data), held as sampler.walker. The sampler's own edges --
    what edge_iter() fine-tunes on and what random_edges() draws from -- stay the directed ones.
    '''
    cls = TRWSampler if temporal else RWSampler
    sampler = cls(data, walk_len=walk_len, batch_size=batch_size, device=device, edge_features=edge_features)
    if undirected_walks:
        und = undirected_walk_graph(data)
        sampler.walker = cls(und, walk_len=walk_len, batch_size=batch_size, device=device,
                             edge_features=edge_features)
        deg, und_deg = data.idxptr[1:] - data.idxptr[:-1], und.idxptr[1:] - und.idxptr[:-1]
        print(f'Undirected walks: {data.col.size(0):,} directed edges -> {und.col.size(0):,} walkable; '
              f'nodes a walk can leave: {int((deg > 0).sum()):,} -> {int((und_deg > 0).sum()):,}')
    return sampler


def walk_sampler(sampler):
    '''The sampler that random walks are drawn from (see make_sampler).'''
    return getattr(sampler, 'walker', sampler)


def load_eval_samplers(name, temporal, edge_features, walk_len, eval_bs):
    va = make_sampler(load_split(name, 'va'), temporal, edge_features, walk_len, eval_bs)
    va.label = torch.zeros_like(va.col)

    te_data = load_split(name, 'te')
    te = make_sampler(te_data, temporal, edge_features, walk_len, eval_bs)
    te.label = te_data.label
    return va, te


def bert_config(name, tr, size):
    num_tokens = int(tr.num_tokens)
    p = SIZES[size]
    return BertConfig(
        vocab_size=num_tokens + OFFSET,
        hidden_size=p.H,
        num_hidden_layers=p.L,
        num_attention_heads=p.H // 64,
        intermediate_size=p.H * 4,
        num_nodes=num_tokens,
        max_position_embeddings=1024 if dataset_kind(name) == 'lanl' else 512,
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def left_align(walk):
    '''
    Move PAD tokens to the end of each row, keeping the order of real tokens.

    Context walks are reversed so they end at the source node, which leaves
    short walks padded at the front. Pretraining walks start at position 0, so
    left-aligning puts fine-tuning tokens on the positions pretraining trained.
    '''
    order = torch.argsort((walk == PAD).int(), dim=1, stable=True)
    return walk.gather(1, order)


def last_real(x, walk):
    '''x[:, i] at each row's last non-PAD position (for left-aligned sequences).'''
    idx = (walk != PAD).sum(dim=1) - 1
    return x[torch.arange(walk.size(0), device=walk.device), idx]


def context(tr, src, ts, delta, use_walk):
    '''
    Random walk ending at src, used as context for scoring an edge out of src.

    Temporal walks look in [ts - delta, ts], clamped to the end of the training
    graph so edges after the training cutoff still get context.
    Under --undirected-walks the walk is drawn on the undirected training graph.
    With edge features, the all-PAD feature block trailing src is removed so the
    scored edge's features sit directly after src (as in pretraining).
    '''
    if not use_walk:
        return src.unsqueeze(-1)

    if not hasattr(tr, '_ts_max'):
        tr._ts_max = int(tr.ts.max())
    max_ts = ts.long().clamp(max=tr._ts_max)
    min_ts = (max_ts - delta).clamp(min=0)

    rw = walk_sampler(tr).rw(src, max_ts=max_ts, min_ts=min_ts, reverse=True, trim_missing=False)
    if tr.edge_features:
        rw = rw[:, :-tr.edge_attr.size(1)]
    return rw

def random_edges_harder(tr, n, generator=None):
    '''
    Generate random edges, but make it a little harder than a pure uniform
    distro. Use real source nodes and random dst nodes. Hopefully will
    improve validation so it doesn't saturate at 0.99 AUC every time.
    '''
    d = tr.device
    pick = torch.randint(0, tr.col.size(0), (n,), generator=generator)

    if 'src' in tr.data.keys():
        src = tr.data.src[pick.to(tr.data.src.device)]
    else:
        src = torch.searchsorted(tr.rowptr, pick.to(tr.rowptr.device), right=True) - 1

    dst = torch.randint(0, tr.num_nodes, (n,), generator=generator)

    pick_ts = pick.to(tr.ts.device)
    ts = tr.ts[pick_ts].to(d)
    ef = (tr.edge_attr[pick_ts] + tr.num_nodes).to(d) if tr.edge_features else None
    return src.to(d), dst.to(d), ts, ef

def random_edges(tr, n, generator=None):
    '''Random node pairs with timestamps and (offset) edge features drawn from real training edges.'''
    src = torch.randint(0, tr.num_nodes, (n,), generator=generator)
    dst = torch.randint(0, tr.num_nodes, (n,), generator=generator)
    pick = torch.randint(0, tr.col.size(0), (n,), generator=generator).to(tr.ts.device)

    d = tr.device
    ts = tr.ts[pick].to(d)
    ef = (tr.edge_attr[pick] + tr.num_nodes).to(d) if tr.edge_features else None
    return src.to(d), dst.to(d), ts, ef


VAL_NEG = ('uniform', 'degree', 'src')


def _edge_keys(tr):
    '''Sorted src * N + dst of every training edge (cached on the sampler), for rejecting real pairs.'''
    if getattr(tr, '_edge_keys', None) is None:
        src = tr.data.src.cpu().long() if 'src' in tr.data.keys() else \
            torch.arange(tr.num_nodes).repeat_interleave(tr.data.idxptr[1:] - tr.data.idxptr[:-1])
        tr._edge_keys = (src * tr.num_nodes + tr.data.col.cpu().long()).unique()
    return tr._edge_keys


def random_edges_degree(tr, n, generator=None, tries=20):
    '''
    Validation negatives that look like plausible traffic: the source, timestamp and edge features
    of a random real training edge, and a destination drawn in proportion to in-degree (the
    destination of another random training edge). Pairs that are training edges, and self-loops,
    are redrawn, so every negative is a (src, dst) the training graph has never seen.
    '''
    E = tr.col.size(0)
    pick = torch.randint(0, E, (n,), generator=generator)
    if 'src' in tr.data.keys():
        src = tr.data.src.cpu()[pick].long()
    else:
        src = torch.searchsorted(tr.rowptr.cpu(), pick, right=True) - 1
    col = tr.data.col.cpu().long()
    keys = _edge_keys(tr)
    dst = col[torch.randint(0, E, (n,), generator=generator)]
    for _ in range(tries):
        k = src * tr.num_nodes + dst
        pos = torch.searchsorted(keys, k).clamp(max=keys.numel() - 1)
        bad = (keys[pos] == k) | (src == dst)
        if not bad.any():
            break
        dst[bad] = col[torch.randint(0, E, (int(bad.sum()),), generator=generator)]

    d = tr.device
    pick_ts = pick.to(tr.ts.device)
    ts = tr.ts[pick_ts].to(d)
    ef = (tr.edge_attr[pick_ts] + tr.num_nodes).to(d) if tr.edge_features else None
    return src.to(d), dst.to(d), ts, ef


def val_negatives(kind):
    return {'degree': random_edges_degree, 'src': random_edges_harder, 'uniform': random_edges}[kind]


def lp_inputs(tr, src, dst, ts, ef, walk_len, delta):
    '''[walk ... src, (src->dst edge features), MASK] with dst as the target.'''
    rw = context(tr, src, ts, delta, use_walk=walk_len > 1)
    parts = [rw] + ([ef] if ef is not None else [])
    parts.append(torch.full((rw.size(0), 1), MASK, device=rw.device, dtype=rw.dtype))
    walk = left_align(torch.cat(parts, dim=1))
    return walk, walk == MASK, dst, walk != PAD


def make_lp_scorer(model, tr, walk_len, delta, score='sigmoid'):
    '''Anomaly score for edges (higher = more anomalous).'''
    def score_fn(src, dst, ts, ef):
        walk, mask, tgt, attn = lp_inputs(tr, src, dst, ts, ef, walk_len, delta)
        logits = model.modified_fwd(walk, mask, tgt, attn, return_loss=False).logits[mask]
        idx = torch.arange(dst.size(0), device=dst.device)
        if score == 'nll':
            return -torch.log_softmax(logits.float(), dim=-1)[idx, dst]
        return 1 - torch.sigmoid(logits[idx, dst])
    return score_fn


def causal_lp_inputs(tr, src, ts, ef, walk_len, delta):
    '''[walk ... src, (src->dst edge features)]; a causal model predicts dst as the next token.'''
    rw = context(tr, src, ts, delta, use_walk=walk_len > 1)
    return left_align(torch.cat([rw, ef], dim=1)) if ef is not None else left_align(rw)


def make_causal_lp_scorer(model, tr, walk_len, delta, score='sigmoid'):
    '''Anomaly score for edges from a GPT model (higher = more anomalous).'''
    def score_fn(src, dst, ts, ef):
        walk = causal_lp_inputs(tr, src, ts, ef, walk_len, delta)
        logits = last_real(model.logits(walk), walk)
        idx = torch.arange(dst.size(0), device=dst.device)
        if score == 'nll':
            return -torch.log_softmax(logits.float(), dim=-1)[idx, dst]
        return 1 - torch.sigmoid(logits[idx, dst])
    return score_fn


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_split(score_fn, split, tr, bs, desc):
    n = split.col.size(0)
    out = torch.empty(n)
    for b in tqdm(torch.arange(n).split(bs), desc=desc, leave=False, disable=None):
        samp = [s.to(tr.device) for s in split._single_iter(b, shuffled=False)]
        src, dst, ts = samp[:3]
        ef = samp[3] if len(samp) == 4 else None
        out[b] = score_fn(src, dst, ts, ef).float().cpu()
    return out


@torch.no_grad()
def evaluate(score_fn, tr, va, te, bs, seed=0, recall=0.5, val_neg='uniform', nll_fn=None):
    '''
    Test: labeled test edges. Besides AUC and AP, reports metrics that stay comparable across
    splits with different attack rates (see metrics.py): AP lift over random ranking, and
    precision and benign edges flagged per day at `recall`.
    Validation: held-out real edges (label 0) vs. an equal number of random edges (label 1),
    drawn by val_negatives(val_neg):
        degree   real source / time / features, destination by in-degree, never a training pair
        src      real source / time / features, uniform destination (random_edges_harder)
        uniform  uniform source and destination (the default; the original rule)
    nll_fn: an LP scorer built with score='nll'. If given, the mean NLL of the true destination
    over the validation edges is printed (not added to the results, so no new CSV column).
    '''
    te_pred = score_split(score_fn, te, tr, bs, 'test')
    te_y = te.label.cpu().numpy()

    tn = score_split(score_fn, va, tr, bs, 'val')
    g = torch.Generator().manual_seed(seed)
    neg = val_negatives(val_neg)
    n = va.col.size(0)
    tp = torch.empty(n)
    for b in torch.arange(n).split(bs):
        tp[b] = score_fn(*neg(tr, b.numel(), g)).float().cpu()

    va_pred = torch.cat([tp, tn]).numpy()
    va_y = np.concatenate([np.ones(n), np.zeros(n)])

    te_ap = fast_ap(te_y, te_pred.numpy())
    out = dict(te_auc=fast_auc(te_y, te_pred.numpy()), te_ap=te_ap)
    out.update(detection_metrics(te_y, te_pred.numpy(), te_ap, ts=te.ts.cpu().numpy(), recall=recall))
    out.update(va_auc=fast_auc(va_y, va_pred), va_ap=fast_ap(va_y, va_pred))
    if nll_fn is not None:
        nll = score_split(nll_fn, va, tr, bs, 'val nll')
        print(f'  val ({val_neg} negatives) AUC {out["va_auc"]:.4f} AP {out["va_ap"]:.4f} | '
              f'held-out NLL of the true destination {nll.mean():.4f} (median {nll.median():.4f})')
    return out


# ---------------------------------------------------------------------------
# Validation probe (--val-probe): which validation metric tracks test AP?
# ---------------------------------------------------------------------------
def _triple_keys(src, dst, ef, num_nodes):
    '''One int64 per (src, dst, edge features) -- or per (src, dst) without features. Wraps; used for set membership only.'''
    h = src.long() * num_nodes + dst.long()
    if ef is not None:
        for j in range(ef.size(1)):
            h = h * 1000003 + ef[:, j].long() + 1
    return h


def _train_keys(tr):
    '''(sorted pair keys, sorted pair+feature keys) of the training edges, cached on the sampler.'''
    if getattr(tr, '_probe_keys', None) is None:
        data = tr.data
        src = data.src.cpu().long() if 'src' in data.keys() else \
            torch.arange(tr.num_nodes).repeat_interleave(data.idxptr[1:] - data.idxptr[:-1])
        col = data.col.cpu().long()
        ef = data.edge_attr.cpu() if tr.edge_features else None
        tr._probe_keys = (_triple_keys(src, col, None, tr.num_nodes).unique(),
                          _triple_keys(src, col, ef, tr.num_nodes).unique(), src, col)
    return tr._probe_keys


def _member(keys, k):
    pos = torch.searchsorted(keys, k).clamp(max=keys.numel() - 1)
    return keys[pos] == k


def random_edges_featswap(tr, n, generator=None, tries=20):
    '''
    Attribute negatives: the source, destination and timestamp of a random real training edge with
    the edge features of ANOTHER random training edge, redrawn until the (src, dst, features)
    combination is one the training graph has never seen. The node pair is real; only its
    attributes are wrong. None without edge features.
    '''
    if not tr.edge_features:
        return None
    _, triples, src_all, col_all = _train_keys(tr)
    ea = tr.data.edge_attr.cpu()
    E = col_all.size(0)
    pick = torch.randint(0, E, (n,), generator=generator)
    src, dst = src_all[pick], col_all[pick]
    feat = torch.randint(0, E, (n,), generator=generator)
    for _ in range(tries):
        bad = _member(triples, _triple_keys(src, dst, ea[feat], tr.num_nodes))
        if not bad.any():
            break
        feat[bad] = torch.randint(0, E, (int(bad.sum()),), generator=generator)
    d = tr.device
    ts = tr.ts[pick.to(tr.ts.device)].to(d)
    return src.to(d), dst.to(d), ts, (ea[feat] + tr.num_nodes).to(d)


def _auc_ap(pos, neg):
    '''AUC / AP of negatives (label 1, should score higher) against positives (label 0); nan if either is empty.'''
    if pos.numel() == 0 or neg.numel() == 0:
        return float('nan'), float('nan')
    pred = torch.cat([neg, pos]).numpy()
    y = np.concatenate([np.ones(neg.numel()), np.zeros(pos.numel())])
    return fast_auc(y, pred), fast_ap(y, pred)


@torch.no_grad()
def validation_probe(score_fn, tr, va, bs, seed=0, nll_fn=None, max_neg=200_000):
    '''
    Every candidate validation metric at once, so one run shows which of them moves with test AP.

    Held-out (validation) edges, as three sets of positives:
        all      every validation edge                                   (what evaluate() uses)
        newpair  only those whose (src, dst) is NOT an edge of the training graph
        newedge  only those whose (src, dst, edge features) combination is not in training
    against three kinds of negatives (the same draws every epoch):
        uniform   uniform source and destination                         (what evaluate() uses)
        degree    real source / time / features, destination by in-degree, never a training pair
        featswap  a real training pair with another edge's features, never a training combination
    Returns {va_<positives>_<negatives>_auc / _ap, n_<positives>, nll_<positives>}.
    '''
    N = tr.num_nodes
    pairs, triples, _, _ = _train_keys(tr)
    v_src = va.data.src.cpu().long()
    v_dst = va.data.col.cpu().long()
    v_ef = va.data.edge_attr.cpu() if tr.edge_features else None
    sets = {'all': torch.ones(v_dst.size(0), dtype=torch.bool),
            'newpair': ~_member(pairs, _triple_keys(v_src, v_dst, None, N)),
            'newedge': ~_member(triples, _triple_keys(v_src, v_dst, v_ef, N))}

    pos = score_split(score_fn, va, tr, bs, 'probe val')
    n = min(v_dst.size(0), max_neg)
    out = {}
    for kind, draw in (('uniform', random_edges), ('degree', random_edges_degree), ('featswap', random_edges_featswap)):
        g = torch.Generator().manual_seed(seed)
        neg, ok = torch.empty(n), True
        for b in torch.arange(n).split(bs):
            e = draw(tr, b.numel(), g)
            if e is None:
                ok = False
                break
            neg[b] = score_fn(*e).float().cpu()
        for name, mask in sets.items():
            auc, ap = _auc_ap(pos[mask], neg) if ok else (float('nan'), float('nan'))
            out[f'va_{name}_{kind}_auc'], out[f'va_{name}_{kind}_ap'] = auc, ap
    for name, mask in sets.items():
        out[f'n_{name}'] = int(mask.sum())
    if nll_fn is not None:
        nll = score_split(nll_fn, va, tr, bs, 'probe nll')
        for name, mask in sets.items():
            out[f'nll_{name}'] = float(nll[mask].mean()) if mask.any() else float('nan')
    return out


def log_probe(path, epoch, m, probe):
    '''Append one row (epoch, test metrics, every probe metric) to the probe CSV and print a summary.'''
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = dict(epoch=epoch, te_auc=m['te_auc'], te_ap=m['te_ap'], **probe)
    new = not os.path.exists(path)
    with open(path, 'w' if new else 'a') as f:
        if new:
            f.write(','.join(row) + '\n')
        f.write(','.join(str(v) for v in row.values()) + '\n')
    print(f'  probe [epoch {epoch}] test AP {m["te_ap"]:.4f} | validation AUC by positives (rows) and negatives (cols):')
    for name in ('all', 'newpair', 'newedge'):
        cells = '  '.join(f'{k} {probe[f"va_{name}_{k}_auc"]:.4f}' for k in ('uniform', 'degree', 'featswap'))
        nll = f'  nll {probe[f"nll_{name}"]:.4f}' if f'nll_{name}' in probe else ''
        print(f'    {name:<8} (n={probe[f"n_{name}"]:>8,})  {cells}{nll}')


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # exists, owned by another user
        return True
    return True


class ResultTracker:
    '''
    Writes one CSV row per evaluation and remembers the best by validation AUC (or AP).

    Holds a lock file ({path}.lock, containing the owner's PID) for the whole run, so a second
    process with the same settings fails immediately instead of truncating the CSV and
    interleaving its rows. Locks left by processes that no longer exist are taken over.
    '''

    def __init__(self, path, select_by='auc'):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.path = path
        self.key = f'va_{select_by}'
        self.best = None
        self.lock = f'{path}.lock'
        self._acquire_lock()
        atexit.register(self._release_lock)
        self.columns = None  # taken from the first metrics dict, so new metrics need no changes here
        open(path, 'w').close()

    def _acquire_lock(self):
        for _ in range(2):
            try:
                fd = os.open(self.lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    with open(self.lock) as f:
                        owner = int(f.read().strip() or -1)
                except (OSError, ValueError):
                    owner = -1
                if owner > 0 and _pid_alive(owner):
                    raise RuntimeError(
                        f'{self.path} is being written by another running process (PID {owner}). '
                        f'Stop it, or use a different --tag / --out-dir.')
                os.remove(self.lock)  # stale lock from a dead process
                continue
            with os.fdopen(fd, 'w') as f:
                f.write(str(os.getpid()))
            return
        raise RuntimeError(f'Could not acquire {self.lock}')

    def _release_lock(self):
        try:
            with open(self.lock) as f:
                if f.read().strip() == str(os.getpid()):
                    os.remove(self.lock)
        except OSError:
            pass

    @staticmethod
    def _describe(m):
        extra = ''
        if 'te_ap_lift' in m:
            r = next((k[len('te_prec_r'):] for k in m if k.startswith('te_prec_r')), None)
            extra = f' lift {m["te_ap_lift"]:.1f}x'
            if r is not None:
                extra += f', @{r}% recall: prec {m[f"te_prec_r{r}"]:.4f}, FP/day {m[f"te_fp_day_r{r}"]:.1f}'
        return (f'TEST AUC {m["te_auc"]:.4f} AP {m["te_ap"]:.4f}{extra} | '
                f'VAL AUC {m["va_auc"]:.4f} AP {m["va_ap"]:.4f}')

    def log(self, epoch, updates, m):
        with open(self.path, 'a') as f:
            if self.columns is None:
                self.columns = list(m)
                f.write(','.join(['epoch', 'updates'] + self.columns) + '\n')
            f.write(','.join([str(epoch), str(updates)] + [str(m.get(c, '')) for c in self.columns]) + '\n')
        print(f'[epoch {epoch}] {self._describe(m)}')

        improved = self.best is None or m[self.key] > self.best[2][self.key]
        if improved:
            self.best = (epoch, updates, m)
        return improved

    def close(self):
        if self.best is None:
            print('No evaluations were run.')
            return
        e, _, m = self.best
        msg = f'BEST (by {self.key}, epoch {e}): {self._describe(m)}'
        print('#' * 20 + '\n' + msg)
        with open(self.path, 'a') as f:
            f.write('# ' + msg + '\n')
        self._release_lock()


def warmup_linear(opt, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(1e-8, 1 - (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def save_checkpoint(state_dict, path):
    '''
    Write to a temporary file in the same directory, then rename over the target.
    The rename is atomic, so a concurrent torch.load never sees a partially written file.
    '''
    tmp = f'{path}.tmp.{os.getpid()}'
    torch.save(state_dict, tmp)
    os.replace(tmp, path)


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


class SpeedTimer:
    '''
    Per-micro-batch timings for --speedtest, split into sampling, forward,
    backward and optimizer step. Synchronizes CUDA before each reading so GPU
    work is attributed to the stage that queued it. Disabled timers are no-ops.
    '''

    STAGES = ('samp', 'fwd', 'bwd', 'step')

    def __init__(self, enabled, device):
        self.enabled = enabled
        self.cuda = enabled and device != 'cpu' and torch.cuda.is_available()
        self.times = {k: [] for k in self.STAGES}
        self.start = time.time()
        self._t = self.start

    def _now(self):
        if self.cuda:
            torch.cuda.synchronize()
        return time.time()

    def mark(self):
        if self.enabled:
            self._t = self._now()

    def lap(self, stage):
        if self.enabled:
            now = self._now()
            self.times[stage].append(now - self._t)
            self._t = now

    def dump(self, path, **meta):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        total = self._now() - self.start
        summary = {k: dict(n=len(v), total=sum(v), mean=(sum(v) / len(v) if v else 0.0))
                   for k, v in self.times.items()}
        with open(path, 'w') as f:
            json.dump(dict(meta=meta, total_seconds=total, summary=summary, times=self.times), f, indent=1)
        print(f'Speedtest: {total:.1f}s total -> {path}')
        for k, v in summary.items():
            print(f'  {k:>4}: {v["total"]:.1f}s over {v["n"]} calls ({v["mean"] * 1000:.1f} ms each)')