'''
Pretrain CyberGFM with raw (continuous) edge features: one projected feature slot per hop
instead of F discretized tokens (see models/raw_feat_bert.py, raw_sampler.py).

    python pretrain_raw.py --dataset lanl14argus --device 0
    python pretrain_raw.py --dataset lanl14argus-dirtyts --trw --device 0 --walk-len 32

Reads data/{dataset}_tgraph_{split}_raw.pt (or the other names load_raw_graph tries).
Checkpoints go to pretrained/{static,temporal}/{dataset}/...rw_rawbert_...pt, which is where
lp_finetune_raw.py looks. They are NOT interchangeable with tokenized checkpoints: the
vocabulary has no feature tokens and the model has a feature projection.
'''
from argparse import ArgumentParser
import math
import os
import time

import torch
from torch.optim import AdamW

from common import DAY, HOUR, SIZES, SpeedTimer, dataset_kind, pretrained_path, run_tag, save_checkpoint, seed_everything
from models.raw_feat_bert import FEAT, RawFeatRWBert, raw_feat_config
from raw_sampler import PAD, RawFeatTokenizer, load_raw_samplers, make_raw_lp_scorer, raw_evaluate

BS = 1024
LR = 3e-4
LOG_EVERY = 50


def defaults(kind, size):
    '''Raw sequences are 2k-1 tokens for a k-node walk, so walks can be much longer than the tokenized ones.'''
    if kind == 'lanl':
        d = dict(total_tokens=1e9, walk_len=32, eval_every=14, mini_bs=128,
                 eval_bs={'mini': 256, 'med': 128, 'baseline': 128}.get(size, 512))
    elif kind == 'unsw':
        d = dict(total_tokens=1e8, walk_len=64, eval_every=500, mini_bs=1024, eval_bs=2048)
    else:  # optc
        d = dict(total_tokens=1e8, walk_len=64, eval_every=100, mini_bs=1035, eval_bs=4096)
    if size == 'baseline':
        d['mini_bs'] = min(d['mini_bs'], 256)
    return d


def lr_at(tokens, total, warmup):
    if tokens < warmup:
        return LR * tokens / warmup
    return LR * max(1e-8, 1 - (tokens - warmup) / (total - warmup))


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--size', default='tiny', choices=list(SIZES))
    ap.add_argument('--device', type=int, default=0, help='-1 for CPU')
    ap.add_argument('--trw', action='store_true')
    ap.add_argument('--walk-len', type=int)
    ap.add_argument('--n-tokens', type=float, help='training length in units of 1e8 tokens')
    ap.add_argument('--mini-bs', type=int)
    ap.add_argument('--n-walks', type=int, default=1)
    ap.add_argument('--eval-every', type=int, help='epochs between evaluations; 0 disables')
    ap.add_argument('--eval-walk-len', type=int, default=4, help='context walk length for the periodic evaluation')
    ap.add_argument('--eval-delta', type=int, help='temporal window for eval walks (default: 1h LANL, 1d OpTC, 0 UNSW)')
    ap.add_argument('--snapshot-every', type=float, help='also keep a checkpoint every N x 1e8 tokens')
    ap.add_argument('--log-out')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--speedtest', action='store_true')
    args = ap.parse_args()
    print(args)
    seed_everything(args.seed)

    name, kind = args.dataset, dataset_kind(args.dataset)
    device = args.device if args.device >= 0 else 'cpu'
    cfg = defaults(kind, args.size)
    walk_len = args.walk_len or cfg['walk_len']
    mini_bs = args.mini_bs or cfg['mini_bs']
    eval_every = cfg['eval_every'] if args.eval_every is None else args.eval_every
    eval_delta = args.eval_delta if args.eval_delta is not None else {'lanl': HOUR, 'unsw': 0, 'optc': DAY}[kind]
    total = int(args.n_tokens * 1e8) if args.n_tokens else int(cfg['total_tokens'])
    warmup = total // 10
    accum = math.ceil(BS / mini_bs)
    snapshot_every = int(args.snapshot_every * 1e8) if args.snapshot_every else 0

    raw_path = lambda p: p.replace('_bert_', '_rawbert_')
    ckpt = raw_path(pretrained_path(name, args.size, args.trw, tag=args.tag))
    best_ckpt = raw_path(pretrained_path(name, args.size, args.trw, best=True, tag=args.tag))
    log_dir = args.log_out or os.path.dirname(ckpt)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    stem = f'{"t" if args.trw else ""}rw_rawbert_{name}{run_tag()}{args.tag}_{args.size}'

    tr, va, te = load_raw_samplers(name, args.trw, walk_len, mini_bs, cfg['eval_bs'], device)
    tr.n_walks = args.n_walks
    print(f'{name}: raw features dim={tr.feat_dim}, walk_len={walk_len}, mini_bs={mini_bs} x{accum}, '
          f'tokens={total:.1e}, vocab={tr.num_nodes} nodes')

    p = SIZES[args.size]
    config = raw_feat_config(tr.num_nodes, tr.feat_dim, p.H, p.L, 1024 if kind == 'lanl' else 512)
    model = RawFeatRWBert(config).to(device)
    model.fit_normalizer(tr.raw)
    tok = RawFeatTokenizer(tr.num_nodes)
    tok.set_mask_rate(0)
    opt = AdamW(model.parameters(), lr=LR, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scorer = make_raw_lp_scorer(model, tr, args.eval_walk_len, eval_delta)

    train_log = open(f'{log_dir}/{stem}_log.csv', 'w')
    train_log.write('updates,tokens,loss,lr,mask_rate,seq_len,walk_len,seconds\n')
    eval_log = open(f'{log_dir}/{stem}_eval.csv', 'w')
    eval_log.write('epoch,updates,tokens,te_auc,te_ap,va_auc,va_ap\n')

    tokens = updates = micro = epoch = 0
    next_snapshot = snapshot_every
    win_tokens = win_walks = 0
    best = -float('inf')
    start = time.time()
    timer = SpeedTimer(args.speedtest, device)
    opt.zero_grad()

    while tokens < total:
        model.train()
        timer.mark()
        for walks, feats in tr.raw_iter():
            if walks.size(0) == 0:
                timer.mark()
                continue

            walks, sel, targets, attn = tok.mask(walks)
            if targets.numel() == 0:
                timer.mark()
                continue
            n_tok = int((walks != PAD).sum())
            tokens += n_tok
            win_tokens += n_tok
            win_walks += walks.size(0)
            timer.lap('samp')

            loss = model.modified_fwd(walks, sel, targets, attn, feats=feats)
            timer.lap('fwd')
            # Sum (don't average) gradients over micro-batches, as in the original code. With clipping
            # at 5 this matters: averaging produced markedly worse checkpoints for downstream LP.
            loss.backward()
            timer.lap('bwd')

            micro += 1
            if micro < accum:
                timer.mark()
                continue
            micro = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            lr = lr_at(tokens, total, warmup)
            for g in opt.param_groups:
                g['lr'] = lr
            opt.step()
            opt.zero_grad()
            timer.lap('step')
            updates += 1
            tok.set_mask_rate(min(1, tokens / warmup))

            if updates % LOG_EVERY == 0:
                el = time.time() - start
                seq_len = win_tokens / max(1, win_walks)
                nodes = (seq_len + 1) / 2  # n0 FEAT n1 ... -> 2k-1 tokens for k nodes
                win_tokens = win_walks = 0
                print(f'[{updates}-{epoch}] loss {loss.item():.4f} lr {lr:.2e} mask {tok.mask_rate:.3f} '
                      f'tokens {tokens:.2e} seq len {seq_len:.1f} (walk len {nodes:.2f}) {el:.0f}s')
                train_log.write(f'{updates},{tokens},{loss.item()},{lr},{tok.mask_rate},{seq_len},{nodes},{el}\n')
                train_log.flush()

            if snapshot_every and tokens >= next_snapshot:
                snap = ckpt.replace('.pt', f'-snap{round(tokens / 1e6)}Mtok.pt')
                save_checkpoint(model.state_dict(), snap)
                print(f'snapshot -> {snap}')
                while next_snapshot <= tokens:
                    next_snapshot += snapshot_every

            if tokens >= total:
                break
            timer.mark()

        if args.speedtest:
            timer.dump(f'latency/latency_{stem}_pt.json', dataset=name, size=args.size, temporal=args.trw,
                       walk_len=walk_len, mini_bs=mini_bs, accum=accum, tokens=tokens, updates=updates,
                       device=str(device))
            raise SystemExit

        epoch += 1
        if eval_every and epoch % eval_every == 0:
            save_checkpoint(model.state_dict(), ckpt)
            model.eval()
            torch.cuda.empty_cache()
            m = raw_evaluate(scorer, tr, va, te, cfg['eval_bs'], seed=args.seed)
            print(f'[epoch {epoch}] TEST AUC {m["te_auc"]:.4f} AP {m["te_ap"]:.4f} | '
                  f'VAL AUC {m["va_auc"]:.4f} AP {m["va_ap"]:.4f}')
            eval_log.write(f'{epoch},{updates},{tokens},{m["te_auc"]},{m["te_ap"]},{m["va_auc"]},{m["va_ap"]}\n')
            eval_log.flush()
            if m['va_auc'] > best:
                best = m['va_auc']
                save_checkpoint(model.state_dict(), best_ckpt)

    save_checkpoint(model.state_dict(), ckpt)
    print(f'Done: {tokens:.2e} tokens, {updates} updates, {time.time() - start:.0f}s -> {ckpt}')