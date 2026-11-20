'''
Everything the four Argus scripts share, so each script only has to load its data:

  Snaps / make_snapshots   per-snapshot edge lists built from flat edge tensors
  arguments                the flags every script takes, with per-dataset defaults
  run                      seeds -> train -> select by validation -> one CSV row per run
  train_epoch / evaluate   their training loop and their protocol, in one place
  append_result / ...      results and caches, safe for many LSF jobs on NFS

Protocol, as in their classification.py:
  * ONE optimizer step per epoch over the whole training window. --bs only decides how many
    snapshots are resident at once; gradients accumulate across pieces and the GRU state is
    carried between them, so a small --bs shortens BPTT, nothing else.
  * Model selection on validation AUC + AP, where validation is held-out real edges against
    uniform random pairs. Early stopping counts misses from epoch 1 and stops at `patience`.
  * Test edges are scored with the embedding of the snapshot that contains them (their
    decode_all), or of the previous snapshot with --score-mode predictive. On a temporal split
    the GRU runs on from the training window into the test snapshots.
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
from typing import List, Optional

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from argus import Argus

try:
    from argus_opt import SOAP              # their optimizer, vendored
except ImportError:                          # pragma: no cover
    from libauc.optimizers import SOAP

DAY = 24 * 3600


# =================================================================================================
# Snapshots
# =================================================================================================
@dataclass
class Snaps:
    '''One split as a sequence of snapshots. Lists are per snapshot; ea is (E, F).'''
    ei: List[torch.Tensor]
    ea: List[torch.Tensor]
    ew: Optional[List[torch.Tensor]]        # GCN edge weights, or None
    y: Optional[List[torch.Tensor]]         # 0/1 labels (test only)
    w: Optional[List[torch.Tensor]]         # events behind each row, when rows were deduplicated
    axis: torch.Tensor                      # the bucket id of each snapshot
    seconds: int                            # length of one bucket
    num_nodes: int

    def __len__(self):
        return len(self.ei)

    def piece(self, a, b):
        return self.ei[a:b], self.ea[a:b], (self.ew[a:b] if self.ew is not None else None)

    def days(self):
        return (int(self.axis.max() - self.axis.min()) + 1) * self.seconds / DAY


def std_edge_w(cnt):
    '''
    Their load_utils.std_edge_w for one snapshot: sigmoid(floor(c / std(c))).
    Their version divides by NaN or 0 on a one-edge or all-equal snapshot; that gives 0.5 here.
    '''
    c = cnt.float()
    std = c.std() if c.numel() > 1 else c.new_tensor(float('nan'))
    if not torch.isfinite(std) or std == 0:
        return torch.full_like(c, 0.5)
    return torch.sigmoid((c.long() / std).long().float())


def make_snapshots(src, dst, bucket, axis, feats, *, seconds, num_nodes, ew=None, y=None,
                   dedup=False):
    '''
    Bucket flat edges onto `axis` (sorted bucket ids; empty buckets give empty snapshots, so the
    GRU sees real elapsed time and splits on the same axis stay aligned index for index).

    dedup=True collapses repeated (src, dst) in a snapshot to one row, as their loaders do: the
    features are averaged, the label is the max, `w` holds the count and `ew` becomes
    std_edge_w(count). Edges whose bucket is not on `axis` are dropped, and reported.
    '''
    pos = torch.searchsorted(axis, bucket)
    on = (pos < len(axis)) & (axis[pos.clamp(max=len(axis) - 1)] == bucket)
    if not on.all():
        print(f'  make_snapshots: {int((~on).sum()):,} of {on.numel():,} edges fall outside '
              f'the axis and are dropped')
    order = torch.argsort(pos[on], stable=True)
    take = lambda t: None if t is None else t[on][order]
    src, dst, feats, ew, y = take(src), take(dst), take(feats).float(), take(ew), take(y)
    sizes = torch.bincount(pos[on][order], minlength=len(axis)).tolist()
    split = lambda t: None if t is None else list(torch.split(t, sizes))

    eis = list(torch.split(torch.stack([src, dst]), sizes, dim=1))
    eas, ews, ys, ws = split(feats), split(ew), split(y), None
    if dedup:
        ws, ews = [], []
        for i, ei in enumerate(eis):
            if ei.size(1) == 0:
                ws.append(torch.zeros(0, dtype=torch.long))
                ews.append(torch.zeros(0))
                continue
            uniq, inv, cnt = ei.unique(dim=1, return_inverse=True, return_counts=True)
            ea = torch.zeros(uniq.size(1), feats.size(1)).index_add_(0, inv, eas[i])
            eis[i], eas[i] = uniq, ea / cnt.clamp(min=1).unsqueeze(1)
            if ys is not None:
                ys[i] = torch.zeros(uniq.size(1), dtype=ys[i].dtype).scatter_reduce_(
                    0, inv, ys[i], reduce='amax', include_self=False)
            ws.append(cnt)
            ews.append(std_edge_w(cnt))
    if ys is not None:
        ys = [t.clamp(0, 1).long() for t in ys]
    return Snaps(eis, eas, ews, ys, ws, axis, seconds, num_nodes)


# =================================================================================================
# Arguments
# =================================================================================================
def arguments(description, *, beta, gamma, bs, undirected, arch='lanl'):
    '''
    The flags every Argus script takes. Defaults follow the PAPER where paper and code disagree
    (two GRU layers, beta > 0) and train to convergence; run_argus.sh PROFILE=code switches to
    their released recipe (--gru-layers 1 --beta 0 --epochs 100 --patience 3).
    '''
    ap = argparse.ArgumentParser(description=description)
    g = ap.add_argument_group('runtime')
    g.add_argument('--device', type=int, default=0, help='CUDA index, or -1 for CPU')
    g.add_argument('--seed', type=int, default=None)
    g.add_argument('--runs', type=int, default=1, help='runs in this process, seeds seed..seed+runs-1')
    g.add_argument('--out-dir', default='../results/argus',
                   help='results go to <out-dir>/<experiment><tag>.csv, one row per run')
    g.add_argument('--tag', default='', help='variant suffix for the results file')
    g.add_argument('--cache', default='tmp', help='where built snapshots are cached')
    g.add_argument('--threads', type=int, default=0, help='torch.set_num_threads; 0 leaves it')
    g.add_argument('--save-scores', action='store_true',
                   help='save the selected epoch\'s per-edge test scores to '
                        '<out-dir>/scores/<experiment><tag>_s<seed>.pt')
    g.add_argument('--bs', '--chunk', dest='bs', type=int, default=bs,
                   help='snapshots resident on the GPU at once (training and inference). One '
                        'optimizer step per epoch whatever this is; 0 = all of them')

    g = ap.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=1000, help='their main.py: 100')
    g.add_argument('--patience', type=int, default=100, help='their main.py: 3')
    g.add_argument('--lr', type=float, default=0.01, help='paper and main.py: 0.01')

    g = ap.add_argument_group('model')
    g.add_argument('--arch', choices=['lanl', 'optc'], default=arch,
                   help='their Argus_LANL (GCN x3 + NNConv) or Argus_OPTC (GCN x4 + DropEdge)')
    g.add_argument('--h-dim', type=int, default=32, help='main.py -H 32')
    g.add_argument('--z-dim', type=int, default=16, help='main.py -z 16; paper: 16')
    g.add_argument('--gru-layers', type=int, default=2, help='paper: 2; their code: 1')
    g.add_argument('--margin', type=float, default=0.8, help='APLoss margin; paper: 0.8')
    g.add_argument('--gamma', type=float, default=gamma, help='APLoss moving average; paper: '
                   '0.01 LANL, 0.1 OpTC')
    g.add_argument('--beta', type=float, default=beta,
                   help='weight on L_DEC; paper: 0.01 LANL, 0.5 OpTC. Their code has no L_DEC '
                        'term, so only --beta 0 is their code')
    g.add_argument('--max-pos', type=int, default=0,
                   help='cap positives per snapshot in the loss (0 = none); APLoss memory is '
                        'quadratic in it. Not in their code')
    g.add_argument('--no-edge-weight', action='store_true',
                   help='do not pass edge weights to the GCN layers (their code does)')
    g.add_argument('--undirected', dest='undirected', action='store_true',
                   help='propagate over each snapshot\'s undirected view; loss and metrics stay '
                        'on the directed edges. Not in their code')
    g.add_argument('--directed', dest='undirected', action='store_false')
    g.add_argument('--score-mode', choices=['contemporaneous', 'predictive'],
                   default='contemporaneous',
                   help='score snapshot t with the embedding of t (theirs) or of t-1')
    ap.set_defaults(undirected=undirected)
    return ap


def finish(args):
    '''Post-parse fixups shared by every script.'''
    args.device = f'cuda:{args.device}' if args.device >= 0 else 'cpu'
    args.bs = args.bs if args.bs > 0 else 1 << 30
    if args.threads:
        torch.set_num_threads(args.threads)
    return args


# =================================================================================================
# Training and evaluation
# =================================================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model, opt, tr, bs):
    '''One optimizer step over the whole window, accumulated in pieces of `bs` snapshots.'''
    model.train()
    opt.zero_grad()
    h, total, n = None, 0.0, len(tr)
    for a in range(0, n, bs):
        b = min(a + bs, n)
        zs, h = model(*tr.piece(a, b), h0=h)
        loss = model.loss(zs, tr.ei[a:b]) * ((b - a) / n)
        if loss.requires_grad:             # a piece of nothing but empty snapshots has no loss
            loss.backward()
        total += loss.item()
        h = h.detach()                     # the state crosses the piece boundary, the gradient not
    opt.step()
    return total


@torch.no_grad()
def embed(model, snaps, bs, h=None):
    '''(T, N, z) embeddings on the CPU, and the final GRU state.'''
    zs = []
    for a in range(0, len(snaps), bs):
        z, h = model(*snaps.piece(a, a + bs), h0=h)
        zs.append(z.cpu())
    return torch.cat(zs), h


def _scoring_embeddings(zs, prev, mode):
    '''The embedding that scores snapshot i: zs[i], or zs[i-1] (prev for i = 0) if predictive.'''
    if mode == 'contemporaneous':
        return list(zs)
    return [prev] + list(zs[:-1])


@torch.no_grad()
def evaluate(model, tr, va, te, *, temporal, mode, bs, keep=None):
    '''Validation (AUC, AP) for model selection, and the test metrics. keep: a dict that receives
    the scored test rows (src, dst, label, score, weight) in scoring order.'''
    model.eval()
    dev = model.device
    zs_tr, h = embed(model, tr, bs)

    pos, neg = [], []
    for z, ei in zip(_scoring_embeddings(zs_tr, None, mode), va.ei):
        if z is None or ei.size(1) == 0:
            continue
        z, ei = z.to(dev), ei.to(dev)
        pos.append(model.decode(ei, z).cpu())
        neg.append(model.decode(model.negatives(ei), z).cpu())
    va_auc, va_ap = val_metrics(torch.cat(pos), torch.cat(neg))

    if temporal:                           # the GRU runs on into the test window
        zs_te, _ = embed(model, te, bs, h)
        prev = zs_tr[-1]
    else:                                  # test snapshots share the training axis
        zs_te, prev = zs_tr, None
    score, y, w, eis = [], [], [], []
    for i, z in enumerate(_scoring_embeddings(zs_te, prev, mode)):
        ei = te.ei[i]
        if z is None or ei.size(1) == 0:
            continue
        score.append(-model.decode(ei.to(dev), z.to(dev)).cpu())    # low probability = anomalous
        y.append(te.y[i])
        eis.append(ei)
        if te.w is not None:
            w.append(te.w[i])
    if keep is not None:
        ei = torch.cat(eis, dim=1)
        keep.update(src=ei[0], dst=ei[1], label=torch.cat(y), score=torch.cat(score),
                    weight=torch.cat(w) if w else None)
    return va_auc, va_ap, test_metrics(torch.cat(y), torch.cat(score),
                                       torch.cat(w) if w else None, te.days())


def val_metrics(pos, neg):
    '''
    Their model-selection metric exactly. classification.py calls get_score(p, n) on a function
    declared get_score(nscore, pscore), so the RANDOM pairs are the positive class and the score
    is 1 - p. AUC is the same either way round; AP is not.
    '''
    score = (1 - torch.cat([neg, pos])).numpy()
    y = np.r_[np.ones(neg.numel()), np.zeros(pos.numel())]
    return roc_auc_score(y, score), average_precision_score(y, score)


def test_metrics(y, score, w, days, recall=0.5):
    '''
    AUC, AP, AP lift over a random ranking, and precision / benign alarms per day at `recall`.
    With w (deduplicated rows) every metric counts EVENTS, the denominator CyberGFM's per-event
    test rows use; auc_uniq / ap_uniq count rows, which is what their score_stats computes.
    '''
    y, s = y.numpy().astype(bool), score.double().numpy()
    wt = np.ones_like(s) if w is None else w.double().numpy()
    out = dict(auc=roc_auc_score(y, s, sample_weight=wt),
               ap=average_precision_score(y, s, sample_weight=wt))
    if w is not None:
        out.update(auc_uniq=roc_auc_score(y, s), ap_uniq=average_precision_score(y, s))

    out['ap_lift'] = out['ap'] / (wt[y].sum() / wt.sum())
    order = np.argsort(-s, kind='stable')                  # ties flagged together, as metrics.py
    k = int(np.searchsorted(np.cumsum((wt * y)[order]), math.ceil(recall * wt[y].sum())))
    flagged = s >= s[order[min(k, len(s) - 1)]]
    tp, fp = wt[flagged & y].sum(), wt[flagged & ~y].sum()
    r = int(round(recall * 100))
    out[f'prec_r{r}'] = tp / (tp + fp)
    out[f'fp_day_r{r}'] = fp / days
    return out


class Selector:
    '''Best epoch by validation AUC + AP, their early-stopping rule, and the test-snooped best.'''

    def __init__(self, patience):
        self.patience, self.misses = patience, 0
        self.best = self.snooped = None

    def update(self, epoch, key, test):
        improved = self.best is None or key > self.best[0]
        if improved:
            self.best, self.misses = (key, epoch, test), 0
        elif epoch >= 1:                   # their loop only counts misses after the first epoch
            self.misses += 1
        if self.snooped is None or test['ap'] > self.snooped[1]['ap']:
            self.snooped = (epoch, test)
        return improved

    @property
    def stop(self):
        return self.misses >= self.patience

    def result(self, last_epoch, last_test):
        _, epoch, test = self.best
        return {**test, 'epoch': epoch, 'epochs_ran': last_epoch + 1,
                'auc_last': last_test['auc'], 'ap_last': last_test['ap'],
                # selected on TEST AP: an upper bound, never the headline
                'ap_snooped': self.snooped[1]['ap'], 'epoch_snooped': self.snooped[0]}


def fit(args, tr, va, te, temporal, scores_path=None):
    pos_len = max(ei.size(1) for ei in tr.ei)
    model = Argus(tr.num_nodes, tr.ea[0].size(1), args.h_dim, args.z_dim, args.gru_layers,
                  args.arch, args.margin, args.gamma, args.beta, args.undirected, args.max_pos,
                  min(pos_len, args.max_pos) if args.max_pos else pos_len, args.device)
    opt = SOAP(model.parameters(), lr=args.lr, mode='adam', weight_decay=0.0)
    sel = Selector(args.patience)
    for e in range(args.epochs):
        t0 = time.time()
        loss = train_epoch(model, opt, tr, args.bs)
        keep = {} if scores_path else None
        va_auc, va_ap, test = evaluate(model, tr, va, te, temporal=temporal,
                                       mode=args.score_mode, bs=args.bs, keep=keep)
        improved = sel.update(e, va_auc + va_ap, test)
        if improved and keep is not None:   # ends as the selected epoch's scores
            os.makedirs(os.path.dirname(scores_path), exist_ok=True)
            torch.save({**keep, 'epoch': e}, scores_path)
        star = ' *' if improved else ''
        print(f'[{e}] {time.time() - t0:.0f}s loss {loss:.4f} | VAL AUC {va_auc:.4f} AP '
              f'{va_ap:.4f} | TEST AUC {test["auc"]:.4f} AP {test["ap"]:.4f} '
              f'P@r50 {test["prec_r50"]:.4f} FP/day {test["fp_day_r50"]:.0f}{star}', flush=True)
        if sel.stop:
            break
    return sel.result(e, test)


def run(args, tr, va, te, *, temporal, experiment, **record):
    '''Train args.runs times and append one row per run to <out-dir>/<experiment><tag>.csv.'''
    if args.no_edge_weight:
        for s in (tr, va, te):
            s.ew = None
    edge_weight = tr.ew is not None
    print(f'{experiment}: {len(tr)} train / {len(te)} test snapshots, '
          f'{sum(e.size(1) for e in tr.ei):,} training edges, edge weights '
          f'{"on" if edge_weight else "off"}, {"undirected" if args.undirected else "directed"} '
          f'propagation, {"temporal" if temporal else "random"} split', flush=True)

    out = f'{args.out_dir}/{experiment}{args.tag}.csv'
    for r in range(args.runs):
        seed = None if args.seed is None else args.seed + r
        if seed is not None:
            seed_everything(seed)
        print(f'\n===== {experiment}, seed {seed} =====', flush=True)
        t0 = time.time()
        scores = (f'{args.out_dir}/scores/{experiment}{args.tag}_s{seed}.pt'
                  if getattr(args, 'save_scores', False) else None)
        res = fit(args, tr, va, te, temporal, scores_path=scores)
        row = dict(seed='' if seed is None else seed, dataset=experiment, **record,
                   split='temporal' if temporal else 'random',
                   **{k: getattr(args, k) for k in HYPER_ARGS},
                   edge_weight=int(edge_weight), undirected=int(args.undirected),
                   host=socket.gethostname(), finished=datetime.now().isoformat(timespec='seconds'),
                   minutes=round((time.time() - t0) / 60, 2), **res)
        append_result(out, row)
        print(f'appended to {out}', flush=True)


# recorded in every row; show_argus_results.py refuses to pool rows that differ in any of them
HYPER_ARGS = ['arch', 'lr', 'h_dim', 'z_dim', 'gru_layers', 'margin', 'gamma', 'beta', 'max_pos',
              'epochs', 'patience', 'bs', 'score_mode']


# =================================================================================================
# Files
# =================================================================================================
def append_result(path, row):
    '''
    Append one row to a CSV every seed job of the experiment shares. The lock is on the CSV
    itself (nothing left behind) and covers read-header-then-write. Columns the header lacks are
    reported and dropped rather than misaligning later rows.
    '''
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
            except OSError:                # NFS emulates flock; closing releases it anyway
                pass


def build_once(path, build):
    '''
    Load a cached object, or build and save it. No locks: concurrent jobs each build and
    os.replace() a private temp file into place, which is atomic on NFS within a directory.
    '''
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return torch.load(path, weights_only=False)
    obj = build()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = f'{path}.tmp.{socket.gethostname()}.{os.getpid()}'
    torch.save(obj, tmp)
    os.replace(tmp, path)
    print(f'built {path}', flush=True)
    return obj