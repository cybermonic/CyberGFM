'''
Pretrain the causal (GPT) CyberGFM variant with next-token prediction on random walks.
Same data, walks, token budget and schedule as pretrain.py so the two are comparable.

    python pretrain_gpt.py --dataset lanl14argus --device 0
    python pretrain_gpt.py --dataset optc-ts --trw --device 0
    python pretrain_gpt.py --dataset lanl14argus --device 0 --speedtest   # one epoch, timings only

    python pretrain_gpt.py --dataset lanl14argus-ts --undirected-walks --tag _undir --device 0

Checkpoints: pretrained/{static,temporal}/{dataset}/{t}rw_gpt_{dataset}..._{size}.pt
'''
from argparse import ArgumentParser
import math
import os
import time

import torch
from torch.optim import AdamW

from common import (DAY, HOUR, PAD, SIZES, SpeedTimer, dataset_kind, evaluate, load_eval_samplers,
                    load_train_graph, make_causal_lp_scorer, make_sampler, pretrained_path,
                    run_tag, save_checkpoint, seed_everything, uses_edge_features, walk_sampler)
from models.hugging_gpt import GPT, gpt_config
from pretrain import BS, LOG_EVERY, defaults, lr_at


def gpt_pretrained_path(*args, **kwargs):
    return pretrained_path(*args, **kwargs).replace('_bert_', '_gpt_')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--size', default='tiny', choices=list(SIZES))
    ap.add_argument('--device', type=int, default=0, help='-1 for CPU')
    ap.add_argument('--trw', action='store_true')
    ap.add_argument('--ignore-edge-feats', action='store_true')
    ap.add_argument('--undirected-walks', action='store_true',
                    help='draw every random walk on the undirected training graph (as the BERT '
                         'scripts); the edges fine-tuned on and scored stay directed. Give the run '
                         'its own --tag, e.g. _undir')
    ap.add_argument('--walk-len', type=int)
    ap.add_argument('--n-tokens', type=float, help='training length in units of 1e8 tokens')
    ap.add_argument('--mini-bs', type=int)
    ap.add_argument('--eval-every', type=int, help='epochs between evaluations; 0 disables')
    ap.add_argument('--eval-walk-len', type=int, default=1,
                    help='context walk length for the periodic evaluation. The default (1) scores '
                         '[src, features, MASK] with no walk, which the model never sees in training; '
                         'match the fine-tuning walk length for a more meaningful proxy.')
    ap.add_argument('--eval-delta', type=int, help='temporal window for eval walks (default: 1h LANL, 1d OpTC, 0 UNSW)')
    ap.add_argument('--snapshot-every', type=float,
                    help='also keep a checkpoint every N x 1e8 tokens, for plotting fine-tuned '
                         'performance against pretraining length')
    ap.add_argument('--poison', type=int, default=0)
    ap.add_argument('--tr-size', type=float, default=1.0)
    ap.add_argument('--log-out')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--tag', default='', help='checkpoint suffix for pretraining variants, e.g. _wl32')
    ap.add_argument('--speedtest', action='store_true',
                    help='train one epoch without evaluation or checkpoints, write timings to latency/, exit')
    args = ap.parse_args()
    print(args)
    seed_everything(args.seed)

    name, kind = args.dataset, dataset_kind(args.dataset)
    device = args.device if args.device >= 0 else 'cpu'
    edge_features = uses_edge_features(name, args.ignore_edge_feats)
    cfg = defaults(kind, args.size, edge_features, args.trw)
    walk_len = args.walk_len or cfg['walk_len']
    mini_bs = args.mini_bs or cfg['mini_bs']
    eval_every = cfg['eval_every'] if args.eval_every is None else args.eval_every
    eval_delta = args.eval_delta if args.eval_delta is not None else {'lanl': HOUR, 'unsw': 0, 'optc': DAY}[kind]
    snapshot_every = int(args.snapshot_every * 1e8) if args.snapshot_every else 0
    total = int(args.n_tokens * 1e8) if args.n_tokens else int(cfg['total_tokens'])
    warmup = total // 10
    accum = math.ceil(BS / mini_bs)

    ckpt = gpt_pretrained_path(name, args.size, args.trw, args.poison, args.tr_size, tag=args.tag)
    best_ckpt = gpt_pretrained_path(name, args.size, args.trw, args.poison, args.tr_size, best=True, tag=args.tag)
    log_dir = args.log_out or os.path.dirname(ckpt)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    stem = f'{"t" if args.trw else ""}rw_gpt_{name}{run_tag(args.poison, args.tr_size)}{args.tag}_{args.size}'

    tr_data = load_train_graph(name, edge_features, args.poison, args.tr_size)
    tr = make_sampler(tr_data, args.trw, edge_features, walk_len, mini_bs, device,
                      undirected_walks=args.undirected_walks)
    walk_src = walk_sampler(tr)  # tr itself unless --undirected-walks
    tr.n_walks = walk_src.n_walks = cfg['n_walks']
    va, te = load_eval_samplers(name, args.trw, edge_features, walk_len, cfg['eval_bs'])

    print(f'{name}: edge features={edge_features}, walk_len={walk_len}, mini_bs={mini_bs} x{accum}, '
          f'tokens={total:.1e}, vocab={int(tr.num_tokens)}, '
          f'walks={"undirected" if args.undirected_walks else "directed"}')

    p = SIZES[args.size]
    config = gpt_config(tr.num_tokens, p.H, p.L, 1024 if kind == 'lanl' else 512)
    model = GPT(config).to(device)
    opt = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    scorer = make_causal_lp_scorer(model, tr, walk_len=args.eval_walk_len, delta=eval_delta)

    train_log = open(f'{log_dir}/{stem}_log.csv', 'w')
    train_log.write('updates,tokens,loss,lr,seq_len,walk_len,seconds\n')
    eval_log = open(f'{log_dir}/{stem}_eval.csv', 'w')
    eval_log.write('epoch,updates,tokens,te_auc,te_ap,va_auc,va_ap\n')

    # Walks interleave F feature tokens per hop: n0 f.. n1 f.. n2 -> tokens = nodes + (nodes - 1) * F
    feat_dim = tr.edge_attr.size(1) if edge_features else 0
    tokens = updates = micro = epoch = 0
    next_snapshot = snapshot_every
    win_tokens = win_walks = 0  # for average sequence / walk length since the last log line
    best = -float('inf')
    start = time.time()
    timer = SpeedTimer(args.speedtest, device)
    opt.zero_grad()

    while tokens < total:
        model.train()
        timer.mark()
        for walks in walk_src:  # walk sampling happens inside the iterator
            if walks.size(0) == 0:
                timer.mark()
                continue
            n_tok = int((walks != PAD).sum())
            tokens += n_tok
            win_tokens += n_tok
            win_walks += walks.size(0)
            timer.lap('samp')

            loss = model(walks)
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

            if updates % LOG_EVERY == 0:
                el = time.time() - start
                seq_len = win_tokens / max(1, win_walks)
                walk_len = (seq_len + feat_dim) / (1 + feat_dim)
                win_tokens = win_walks = 0
                print(f'[{updates}-{epoch}] loss {loss.item():.4f} lr {lr:.2e} tokens {tokens:.2e} '
                      f'seq len {seq_len:.1f} (walk len {walk_len:.2f}) {el:.0f}s')
                train_log.write(f'{updates},{tokens},{loss.item()},{lr},{seq_len},{walk_len},{el}\n')
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
                       edge_features=edge_features, walk_len=walk_len, mini_bs=mini_bs,
                       accum=accum, tokens=tokens, updates=updates, device=str(device))
            raise SystemExit

        epoch += 1

        if eval_every and epoch % eval_every == 0:
            save_checkpoint(model.state_dict(), ckpt)  # latest; saved with each evaluation, not every epoch
            model.eval()
            torch.cuda.empty_cache()
            m = evaluate(scorer, tr, va, te, cfg['eval_bs'], seed=args.seed)
            print(f'[epoch {epoch}] TEST AUC {m["te_auc"]:.4f} AP {m["te_ap"]:.4f} | '
                  f'VAL AUC {m["va_auc"]:.4f} AP {m["va_ap"]:.4f}')
            eval_log.write(f'{epoch},{updates},{tokens},{m["te_auc"]},{m["te_ap"]},{m["va_auc"]},{m["va_ap"]}\n')
            eval_log.flush()
            if m['va_auc'] > best:
                best = m['va_auc']
                save_checkpoint(model.state_dict(), best_ckpt)

    save_checkpoint(model.state_dict(), ckpt)
    print(f'Done: {tokens:.2e} tokens, {updates} updates, {time.time() - start:.0f}s -> {ckpt}')