'''
Shared plumbing for the non-Argus baselines: GraphSAGE / GIN (gnn_test.py), Euler
(euler_test.py), PIKACHU (pikachu_test.py) and node2vec (n2v_test.py).

DATA. Every script reads CyberGFM's own split files, <data>/<dataset>_tgraph_{tr,va,te}.pt, so
each baseline scores exactly the validation and test edges CyberGFM scores, one row per event.
None of these models uses edge features, so the tokenized (non-raw) files are fine.

SPLITS. A split is temporal when every test edge is at or after the last training edge
(lanl14argus-ts, -dirtyts, -cleants, optc-argus); otherwise it is a random edge split
(lanl14argus, optc, unsw). The static models (GNN, node2vec) always score held-out edges
against the embedding of the training graph, as CyberGFM does. The snapshot models (Euler,
PIKACHU) need a graph for every snapshot they score, so `Timeline` gives them one:

  * inside the training window, the snapshot graph is the TRAINING edges of that bucket, and
    validation / test edges in the window are held out and scored against it;
  * beyond the training window (temporal splits only) the snapshot graph is the edges observed
    there -- validation edges for -dirtyts / -cleants, then test edges -- which is how Euler and
    Argus run detection: the recurrent state carries on from the training window, and the
    edges of snapshot t are scored with the embedding of t. No loss is ever taken on them.

METRICS. Model selection is on validation AUC: held-out validation edges against the same
number of uniformly random node pairs, drawn once per run from a fixed generator so every
epoch is scored against the same negatives. Test metrics are `test_metrics` (AUC, AP, AP lift,
precision and benign alarms per day at 50% recall), the same function as the Argus scripts.
Scores are always oriented HIGH = anomalous.

RESULTS. One row per run appended to <out-dir>/<model>/<dataset><tag>.csv (file-locked, so
seed jobs can share it), with the selected-epoch test metrics, the last-epoch and test-snooped
values for reference, and the hyperparameters.
'''
import argparse
import csv
import fcntl
import math
import os
import random
import socket
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

DAY = 86400
DATASETS = ['lanl14argus', 'lanl14argus-ts', 'lanl14argus-dirtyts', 'lanl14argus-cleants',
            'optc', 'optc-argus', 'unsw']
TS_DIV = {'lanl': 3600, 'optc': 3600, 'unsw': 1, 'toy': 3600}   # seconds per snapshot


def dataset_kind(name):
    for kind in TS_DIV:
        if name.startswith(kind):
            return kind
    raise ValueError(f'unrecognized dataset {name!r}')


# =================================================================================================
# Data
# =================================================================================================
@dataclass
class Split:
    src: torch.Tensor
    dst: torch.Tensor
    ts: torch.Tensor                 # seconds
    y: Optional[torch.Tensor]        # 0/1, test only

    @property
    def ei(self):
        return torch.stack([self.src, self.dst])

    def __len__(self):
        return self.src.numel()


def _split(g, labelled):
    if 'src' in g:
        src = g.src
    else:
        src = torch.arange(g.x.size(0)).repeat_interleave(g.idxptr[1:] - g.idxptr[:-1])
    y = g.label.long().clamp(0, 1) if labelled else None
    return Split(src.long(), g.col.long(), g.ts.long(), y)


def load_splits(dataset, data_dir):
    '''Val and test stay one row per event: every edge, duplicates included, is scored.'''
    gs = {s: torch.load(f'{data_dir}/{dataset}_tgraph_{s}.pt', weights_only=False)
          for s in ('tr', 'va', 'te')}
    num_nodes = gs['tr'].x.size(0)
    tr, va, te = _split(gs['tr'], False), _split(gs['va'], False), _split(gs['te'], True)
    for s in (tr, va, te):
        assert int(s.ei.max()) < num_nodes, 'node id outside x; are the split files from one build?'
    temporal = int(te.ts.min()) >= int(tr.ts.max())
    print(f'{dataset}: {num_nodes:,} nodes | train {len(tr):,}, val {len(va):,}, test '
          f'{len(te):,} edges ({int(te.y.sum()):,} malicious) | '
          f'{"temporal" if temporal else "random"} split', flush=True)
    return tr, va, te, num_nodes, temporal


def test_days(te):
    '''Span of the test edges in days: the denominator of benign alarms per day.'''
    return max(1, int(te.ts.max() - te.ts.min())) / DAY


def to_undirected(ei):
    if ei.size(1) == 0:
        return ei
    ei = torch.cat([ei, ei.flip(0)], dim=1)
    return ei.unique(dim=1)


def csr(ei, num_nodes):
    '''(rowptr, col) of ei sorted by source.'''
    order = torch.argsort(ei[0] * num_nodes + ei[1])
    src, col = ei[0][order], ei[1][order]
    rowptr = torch.zeros(num_nodes + 1, dtype=torch.long)
    rowptr[1:] = torch.bincount(src, minlength=num_nodes).cumsum(0)
    return rowptr, col


class Timeline:
    '''
    The splits cut into snapshots of `ts_div` seconds. Every list is indexed by snapshot, on one
    axis shared by all three splits (the union of their non-empty buckets, in order).

      graph[t]   edges message passing / walks see at t (see the module docstring)
      n_train    snapshots 0 .. n_train-1 are the training window; losses only ever use these
      tr[t]      training edges at t (the loss targets)
      va[t]      validation edges scored at t
      te[t], y[t]  test edges scored at t and their labels
    '''

    def __init__(self, tr, va, te, ts_div, temporal, axis_from='all'):
        b = {k: s.ts // ts_div for k, s in (('tr', tr), ('va', va), ('te', te))}
        if axis_from == 'train':
            # only buckets that hold a TRAINING edge; val/test edges elsewhere are dropped (the
            # original per-dataset scripts did this with `ts = tr.ts.unique()`)
            self.axis = b['tr'].unique()
            for k, s in (('va', va), ('te', te)):
                keep = torch.isin(b[k], self.axis)
                if not keep.all():
                    extra = f' ({int(s.y[~keep].sum()):,} malicious)' if s.y is not None else ''
                    print(f'  axis=train: dropping {int((~keep).sum()):,} of {keep.numel():,} '
                          f'{k} edges in buckets with no training edge{extra}', flush=True)
                s.src, s.dst, s.ts = s.src[keep], s.dst[keep], s.ts[keep]
                if s.y is not None:
                    s.y = s.y[keep]
                b[k] = b[k][keep]
        else:
            self.axis = torch.cat(list(b.values())).unique()
        self.ts_div = ts_div
        # a random split's window is the whole axis: a last bucket that happens to hold only
        # held-out edges must not become an "observed" snapshot
        last_tr = int(b['tr'].max()) if temporal else int(self.axis.max())
        self.n_train = int((self.axis <= last_tr).sum())

        self.tr = self._bucket(tr.ei, b['tr'])
        self.va = self._bucket(va.ei, b['va'])
        self.te = self._bucket(te.ei, b['te'])
        self.y = self._bucket(te.y.unsqueeze(0), b['te'])
        self.y = [y.squeeze(0) for y in self.y]

        # beyond the training window, the graph is whatever was observed there
        obs = torch.cat([va.ei[:, b['va'] > last_tr], te.ei[:, b['te'] > last_tr]], dim=1)
        obs_b = torch.cat([b['va'][b['va'] > last_tr], b['te'][b['te'] > last_tr]])
        beyond = self._bucket(obs, obs_b)
        self.graph = [self.tr[t] if t < self.n_train else beyond[t] for t in range(len(self))]

        n_beyond = sum(e.size(1) for e in self.graph[self.n_train:])
        print(f'timeline: {len(self)} snapshots of {ts_div}s, {self.n_train} in the training '
              f'window' + (f'; {len(self) - self.n_train} after it, embedded from {n_beyond:,} '
                           f'observed (val/test) edges' if n_beyond else ''), flush=True)

    def _bucket(self, ei, bucket):
        pos = torch.searchsorted(self.axis, bucket)
        order = torch.argsort(pos, stable=True)
        sizes = torch.bincount(pos, minlength=len(self.axis)).tolist()
        return list(torch.split(ei[:, order], sizes, dim=1))

    def __len__(self):
        return len(self.axis)


def random_pairs(n, num_nodes, generator=None, device='cpu'):
    return torch.randint(0, num_nodes, (2, n), generator=generator).to(device)


def local_pairs(ei, device='cpu'):
    '''The original scripts' negatives: src uniform over [min, max] of the snapshot's sources,
    dst likewise over its destinations (inclusive, so a single-source snapshot does not fail).'''
    n = ei.size(1)
    src = torch.randint(int(ei[0].min()), int(ei[0].max()) + 1, (n,))
    dst = torch.randint(int(ei[1].min()), int(ei[1].max()) + 1, (n,))
    return torch.stack([src, dst]).to(device)


# =================================================================================================
# Skip-gram on uniform random walks (node2vec with p = q = 1), for n2v_test.py and PIKACHU.
# Self-contained, so it needs neither pyg-lib nor torch_cluster.
# =================================================================================================
EPS = 1e-15


def walks(rowptr, col, start, length):
    '''Uniform random walks of `length` nodes; a node with no out-neighbours stays put.
    Repeated edges in the CSR are kept, so multigraph edges weight the transitions.'''
    out = [start]
    cur = start
    deg = rowptr[1:] - rowptr[:-1]
    for _ in range(length - 1):
        d = deg[cur]
        off = (torch.rand(cur.numel(), device=cur.device) * d).long()
        nxt = col[(rowptr[cur] + off).clamp(max=max(col.numel() - 1, 0))] if col.numel() else cur
        cur = torch.where(d > 0, nxt, cur)
        out.append(cur)
    return torch.stack(out, 1)


def windows(rw, ctx):
    return torch.cat([rw[:, j:j + ctx] for j in range(rw.size(1) - ctx + 1)])


def skipgram_loss(emb, pos, neg):
    '''PyG Node2Vec.loss: the first node of each window against the rest; one negative walk
    of uniformly random nodes per positive walk.'''
    def side(w, sign):
        h0 = emb[w[:, 0]].unsqueeze(1)
        out = (h0 * emb[w[:, 1:]]).sum(-1).view(-1)
        p = torch.sigmoid(out)
        return -torch.log((p if sign else 1 - p) + EPS).mean()
    return side(pos, True) + side(neg, False)


def skipgram(ei, num_nodes, *, dim, walk_len, ctx, epochs, lr, bs, device, name=None):
    '''Embeddings (num_nodes, dim) from walks on the graph ei, as given (directed, repeats kept).
    `name` prints the mean loss after every epoch.'''
    rowptr, col = csr(ei, num_nodes)
    rowptr, col = rowptr.to(device), col.to(device)
    emb = torch.nn.Parameter(torch.empty(num_nodes, dim, device=device)
                             .uniform_(-0.5 / dim, 0.5 / dim))          # word2vec's init
    opt = torch.optim.Adam([emb], lr=lr)
    t0 = time.time()
    for e in range(epochs):
        tot, nb = 0.0, 0
        for batch in torch.randperm(num_nodes, device=device).split(bs):
            pos = windows(walks(rowptr, col, batch, walk_len), ctx)
            neg = torch.cat([batch.view(-1, 1), torch.randint(num_nodes, (batch.numel(),
                             walk_len - 1), device=device)], 1)
            neg = windows(neg, ctx)
            opt.zero_grad()
            loss = skipgram_loss(emb, pos, neg)
            loss.backward()
            opt.step()
            tot, nb = tot + loss.item(), nb + 1
        if name:
            print(f'  {name} [{e}] loss {tot / nb:.4f}  {time.time() - t0:.0f}s', flush=True)
    return emb.detach()


# =================================================================================================
# Metrics and model selection
# =================================================================================================
def val_auc(pos_score, neg_score):
    '''Anomaly scores (high = anomalous) of real and random pairs; random pairs are the positives.'''
    s = torch.cat([pos_score, neg_score]).double().cpu().numpy()
    y = np.r_[np.zeros(pos_score.numel()), np.ones(neg_score.numel())]
    return roc_auc_score(y, s), average_precision_score(y, s)


def test_metrics(y, score, days, recall=0.5):
    '''AUC, AP, AP lift over a random ranking, and precision / benign alarms per day at `recall`,
    over every test event (one row per edge, duplicates included).'''
    y, s = y.cpu().numpy().astype(bool), score.double().cpu().numpy()
    out = dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s))
    out['ap_lift'] = out['ap'] / y.mean()
    order = np.argsort(-s, kind='stable')
    k = int(np.searchsorted(np.cumsum(y[order]), math.ceil(recall * y.sum())))
    flagged = s >= s[order[min(k, len(s) - 1)]]
    tp, fp = (flagged & y).sum(), (flagged & ~y).sum()
    r = int(round(recall * 100))
    out[f'prec_r{r}'] = tp / (tp + fp)
    out[f'fp_day_r{r}'] = fp / days
    return out


class Selector:
    '''Best epoch by validation AUC with early stopping, plus the last and test-snooped values.'''

    def __init__(self, patience):
        self.patience, self.misses = patience, 0
        self.best = self.snooped = self.last = None

    def update(self, epoch, va, test):
        self.last = (epoch, va, test)
        improved = self.best is None or va[0] > self.best[1][0]
        if improved:
            self.best, self.misses = (epoch, va, test), 0
        else:
            self.misses += 1
        if self.snooped is None or test['ap'] > self.snooped[1]['ap']:
            self.snooped = (epoch, test)
        return improved

    @property
    def stop(self):
        return self.patience > 0 and self.misses >= self.patience

    def result(self):
        epoch, (va_auc, va_ap), test = self.best
        return {**test, 'va_auc': va_auc, 'va_ap': va_ap, 'epoch': epoch,
                'epochs_ran': self.last[0] + 1,
                'auc_last': self.last[2]['auc'], 'ap_last': self.last[2]['ap'],
                # selected on TEST AP: an upper bound, never the headline
                'ap_snooped': self.snooped[1]['ap'], 'epoch_snooped': self.snooped[0]}


def log_epoch(e, loss, va, test, improved, t0):
    print(f'[{e}] {time.time() - t0:.0f}s loss {loss:.4f} | VAL AUC {va[0]:.4f} AP {va[1]:.4f} | '
          f'TEST AUC {test["auc"]:.4f} AP {test["ap"]:.4f} P@r50 {test["prec_r50"]:.4f} '
          f'FP/day {test["fp_day_r50"]:.0f}{" *" if improved else ""}', flush=True)


# =================================================================================================
# Runs
# =================================================================================================
def arguments(description, model_flags=None):
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument('--dataset', required=True, help=f'one of {", ".join(DATASETS)}')
    ap.add_argument('--data', default='../data')
    ap.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--runs', type=int, default=1, help='runs in this process, seeds seed..seed+runs-1')
    ap.add_argument('--out-dir', default='../results/baselines',
                    help='results go to <out-dir>/<model>/<dataset><tag>.csv, one row per run')
    ap.add_argument('--tag', default='')
    ap.add_argument('--ts-div', type=int, help='seconds per snapshot (default: 3600 LANL/OpTC, 1 UNSW)')
    return ap


def finish(args):
    args.device = f'cuda:{args.device}' if args.device >= 0 and torch.cuda.is_available() else 'cpu'
    if args.ts_div is None:
        args.ts_div = TS_DIV[dataset_kind(args.dataset)]
    return args


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run(args, model_name, fit, hyper):
    '''Call fit() args.runs times; append one row per run to <out-dir>/<model>/<dataset><tag>.csv.'''
    out = f'{args.out_dir}/{model_name}/{args.dataset}{args.tag}.csv'
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        if seed is not None:
            seed_everything(seed)
        print(f'\n===== {model_name} on {args.dataset}, seed {seed} =====', flush=True)
        t0 = time.time()
        res = fit()
        row = dict(seed='' if seed is None else seed, model=model_name, dataset=args.dataset,
                   **{k: getattr(args, k) for k in hyper}, host=socket.gethostname(),
                   finished=datetime.now().isoformat(timespec='seconds'),
                   minutes=round((time.time() - t0) / 60, 2), **res)
        append_result(out, row)
        print(f'TEST AUC {res["auc"]:.4f} AP {res["ap"]:.4f} (epoch {res["epoch"]}) -> {out}',
              flush=True)


def append_result(path, row):
    '''Append one row to a CSV the seed jobs share; the lock covers read-header-then-write.'''
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'a+', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0)
            first = f.readline()
            fields = next(csv.reader([first])) if first.strip() else list(row)
            missing = [k for k in row if k not in fields]
            if missing:
                print(f'[append_result] {path} has no column(s) {missing}; not recorded. '
                      f'Use a new --tag or --out-dir for runs with new columns.', flush=True)
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', restval='')
            if not first.strip():
                w.writeheader()
            w.writerow(row)
            f.flush()
            os.fsync(f.fileno())
        finally:
            try:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
