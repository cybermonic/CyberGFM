'''
Classification fine-tuning: [walk ... u, (u->v edge features), v, CLS] -> is this a real edge?
Trained on real training edges vs. random edges. Anomaly score = 1 - sigmoid(logit).

    python cls_finetune.py --dataset optc-ts --device 0
    python cls_finetune.py --dataset optc-ts --trw --device 0
    python cls_finetune.py --dataset optc-ts --device 0 --speedtest   # one epoch, timings only
'''
from argparse import ArgumentParser
import math

import torch
from torch.optim import AdamW

from common import (VAL_NEG, DAY, MASK, PAD, SIZES, ResultTracker, SpeedTimer, bert_config, context, dataset_kind, evaluate, left_align,
                    load_eval_samplers, load_train_graph, make_sampler, pretrained_path,
                    random_edges, run_tag, seed_everything, uses_edge_features, warmup_linear)
from models.gnn_bert import RWBertFT

BS = 1024


def cls_inputs(tr, src, dst, ts, ef, walk_len, delta):
    rw = context(tr, src, ts, delta, use_walk=walk_len > 0)
    parts = [rw] + ([ef] if ef is not None else [])
    parts += [dst.unsqueeze(-1), torch.full((rw.size(0), 1), MASK, device=rw.device, dtype=rw.dtype)]
    walk = left_align(torch.cat(parts, dim=1))
    return walk, walk != PAD, walk == MASK


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--size', default='tiny', choices=SIZES.keys())
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--trw', action='store_true')
    ap.add_argument('--ignore-edge-feats', action='store_true')
    ap.add_argument('--undirected-walks', action='store_true',
                    help='draw the context walks on the undirected training graph. The edges that are '
                         'fine-tuned on and the validation / test edges that are scored stay directed')
    ap.add_argument('--walk-len', type=int, default=4)
    ap.add_argument('--delta', type=int, help='temporal walk window (seconds); default 1 day (0 for UNSW)')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--warmup-epochs', type=float, help='default: epochs / 3.33')
    ap.add_argument('--eval-every', type=int, default=1, help='epochs')
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--mini-bs', type=int)
    ap.add_argument('--eval-bs', type=int)
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--val-neg', choices=VAL_NEG, default='uniform',
                    help='validation negatives (see common.evaluate);')
    ap.add_argument('--from-random', action='store_true')
    ap.add_argument('--model-fname')
    ap.add_argument('--best-pretrained', action='store_true')
    ap.add_argument('--poison', type=int, default=0)
    ap.add_argument('--tr-size', type=float, default=1.0)
    ap.add_argument('--out-dir')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pretrain-tag', default='', help='--tag used when pretraining, e.g. _wl32')
    ap.add_argument('--speedtest', action='store_true',
                    help='train one epoch without evaluation, write timings to latency/, exit')
    args = ap.parse_args()
    print(args)
    seed_everything(args.seed)

    name, kind = args.dataset, dataset_kind(args.dataset)
    device = args.device if args.device >= 0 else 'cpu'
    edge_features = uses_edge_features(name, args.ignore_edge_feats)
    delta = args.delta if args.delta is not None else (0 if kind == 'unsw' else DAY)
    mini_bs = args.mini_bs or (512 if args.size == 'baseline' else 1024)
    eval_bs = args.eval_bs or (2048 if kind == 'optc' else 1024)
    warmup_epochs = args.epochs / 3.33 if args.warmup_epochs is None else args.warmup_epochs
    accum = math.ceil(BS / mini_bs)

    tr_data = load_train_graph(name, edge_features, args.poison, args.tr_size)
    tr = make_sampler(tr_data, args.trw, edge_features, args.walk_len, mini_bs, device,
                      undirected_walks=args.undirected_walks)
    va, te = load_eval_samplers(name, args.trw, edge_features, args.walk_len, eval_bs)

    sd = None
    if not args.from_random:
        path = args.model_fname or pretrained_path(name, args.size, args.trw, args.poison,
                                                   args.tr_size, best=args.best_pretrained, tag=args.pretrain_tag)
        print('Loading', path)
        sd = torch.load(path, weights_only=True, map_location='cpu')
    model = RWBertFT(bert_config(name, tr, args.size), sd, device=device, from_random=args.from_random)

    updates_per_epoch = math.ceil(tr.col.size(0) / (mini_bs * accum))
    total_steps = updates_per_epoch * args.epochs
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=args.wd)
    sched = warmup_linear(opt, int(updates_per_epoch * warmup_epochs), total_steps)

    @torch.no_grad()
    def scorer(src, dst, ts, ef):
        out = model.predict(*cls_inputs(tr, src, dst, ts, ef, args.walk_len, delta))
        return 1 - torch.sigmoid(out).squeeze(-1)

    out_dir = args.out_dir or f'results/cls-{"temporal" if args.trw else "static"}/{name}'
    stem = (f'{"rand_init_" if args.from_random else ""}cls{run_tag(args.poison, args.tr_size)}{args.pretrain_tag}'
            f'_{args.size}_wl{args.walk_len}{args.tag}')
    tracker = ResultTracker(f'{out_dir}/{stem}.csv', args.select_by)
    print(f'{name}: edge features={edge_features}, delta={delta}, mini_bs={mini_bs} x{accum}, '
          f'{total_steps} updates, walks={"undirected" if args.undirected_walks else "directed"}')

    def run_eval(epoch, updates):
        model.eval()
        torch.cuda.empty_cache()
        tracker.log(epoch, updates, evaluate(scorer, tr, va, te, eval_bs, seed=args.seed, val_neg=args.val_neg))

    if not (args.from_random or args.speedtest):
        run_eval(0, 0)

    timer = SpeedTimer(args.speedtest, device)
    updates = micro = 0
    opt.zero_grad()
    for epoch in range(1, args.epochs + 1):
        model.train()
        timer.mark()
        for samp in tr.edge_iter():
            src, dst, ts = samp[:3]
            ef = samp[3] if edge_features else None
            n = src.size(0)

            neg_src, neg_dst, neg_ts, neg_ef = random_edges(tr, n)
            src, dst = torch.cat([src, neg_src]), torch.cat([dst, neg_dst])
            ts = torch.cat([ts, neg_ts])
            if ef is not None:
                ef = torch.cat([ef, neg_ef])

            labels = torch.zeros(2 * n, 1, device=src.device)
            labels[:n] = 1  # real edges
            inputs = cls_inputs(tr, src, dst, ts, ef, args.walk_len, delta)
            timer.lap('samp')

            loss = model(*inputs, labels)
            timer.lap('fwd')
            (loss / accum).backward()
            timer.lap('bwd')

            micro += 1
            if micro < accum:
                timer.mark()
                continue
            micro = 0

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            opt.step()
            sched.step()
            opt.zero_grad()
            timer.lap('step')
            updates += 1
            if updates % 100 == 0:
                print(f'[{updates}-{epoch}] loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e}')
            timer.mark()

        if args.speedtest:
            timer.dump(f'latency/latency_{name}_cls{run_tag(args.poison, args.tr_size)}_{args.size}_wl{args.walk_len}.json',
                       dataset=name, size=args.size, temporal=args.trw, edge_features=edge_features,
                       walk_len=args.walk_len, mini_bs=mini_bs, accum=accum, updates=updates,
                       edges=tr.col.size(0), device=str(device))
            raise SystemExit

        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            run_eval(epoch, updates)

    tracker.close()