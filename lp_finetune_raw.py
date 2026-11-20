'''
LP fine-tuning for the raw-feature model: given [walk ... u, FEAT(x_uv), MASK], predict v.
Anomaly score = 1 - sigmoid(logit_v), as in lp_finetune.py.

    python lp_finetune_raw.py --dataset lanl14argus --device 0 --walk-len 4
    python lp_finetune_raw.py --dataset lanl14argus-dirtyts --trw --device 0 --walk-len 4
'''
from argparse import ArgumentParser
import math

import torch
from torch.optim import AdamW

from common import (DAY, HOUR, SIZES, ResultTracker, SpeedTimer, dataset_kind, pretrained_path,
                    run_tag, seed_everything, warmup_linear)
from models.raw_feat_bert import RawFeatRWBert, raw_feat_config
from raw_sampler import load_raw_samplers, make_raw_lp_scorer, raw_evaluate, raw_lp_inputs

BS = 1024


def defaults(kind, size, walk_len):
    delta = {'lanl': HOUR, 'unsw': 0, 'optc': DAY}[kind]
    mini_bs = 512 if size == 'baseline' else 1024
    eval_bs = {'lanl': 1024, 'unsw': 1024, 'optc': 4096}[kind]
    if kind == 'lanl':
        if walk_len > 8:
            mini_bs, eval_bs = 512, 512
        if walk_len > 16:
            mini_bs, eval_bs = 256, 256
    return delta, mini_bs, eval_bs


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--size', default='tiny', choices=list(SIZES))
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--trw', action='store_true')
    ap.add_argument('--walk-len', type=int, default=4)
    ap.add_argument('--delta', type=int)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--warmup-epochs', type=float, help='default: epochs / 3.75')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.02)
    ap.add_argument('--mini-bs', type=int)
    ap.add_argument('--eval-bs', type=int)
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--score', choices=['sigmoid', 'nll'], default='sigmoid')
    ap.add_argument('--from-random', action='store_true')
    ap.add_argument('--model-fname', help='default: the pretrained raw checkpoint for this dataset/size')
    ap.add_argument('--best-pretrained', action='store_true')
    ap.add_argument('--pretrain-tag', default='')
    ap.add_argument('--out-dir')
    ap.add_argument('--tag', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--speedtest', action='store_true')
    args = ap.parse_args()
    print(args)
    seed_everything(args.seed)

    name, kind = args.dataset, dataset_kind(args.dataset)
    device = args.device if args.device >= 0 else 'cpu'
    delta, mini_bs, eval_bs = defaults(kind, args.size, args.walk_len)
    delta = delta if args.delta is None else args.delta
    mini_bs = args.mini_bs or mini_bs
    eval_bs = args.eval_bs or eval_bs
    warmup_epochs = args.epochs / 3.75 if args.warmup_epochs is None else args.warmup_epochs
    accum = math.ceil(BS / mini_bs)

    tr, va, te = load_raw_samplers(name, args.trw, args.walk_len, mini_bs, eval_bs, device)

    p = SIZES[args.size]
    config = raw_feat_config(tr.num_nodes, tr.feat_dim, p.H, p.L, 1024 if kind == 'lanl' else 512)
    model = RawFeatRWBert(config)
    if not args.from_random:
        path = args.model_fname or pretrained_path(
            name, args.size, args.trw, best=args.best_pretrained, tag=args.pretrain_tag
        ).replace('_bert_', '_rawbert_')
        print('Loading', path)
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        model.fit_normalizer(tr.raw)
    model = model.to(device)

    updates_per_epoch = math.ceil(tr._col.size(0) / (mini_bs * accum))
    total_steps = updates_per_epoch * args.epochs
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=args.wd)
    sched = warmup_linear(opt, int(updates_per_epoch * warmup_epochs), total_steps)
    scorer = make_raw_lp_scorer(model, tr, args.walk_len, delta, args.score)

    out_dir = args.out_dir or f'results/lp-raw-{"temporal" if args.trw else "static"}/{name}'
    stem = (f'{"rand_init_" if args.from_random else ""}lp_raw_{"temporal" if args.trw else "static"}'
            f'{run_tag()}{args.pretrain_tag}_{args.size}_wl{args.walk_len}{args.tag}')
    tracker = ResultTracker(f'{out_dir}/{stem}.csv', args.select_by)
    print(f'{name}: raw features dim={tr.feat_dim}, delta={delta}, mini_bs={mini_bs} x{accum}, '
          f'{total_steps} updates')

    def run_eval(epoch, updates):
        model.eval()
        torch.cuda.empty_cache()
        tracker.log(epoch, updates, raw_evaluate(scorer, tr, va, te, eval_bs, seed=args.seed))

    if not (args.from_random or args.speedtest):
        run_eval(0, 0)

    timer = SpeedTimer(args.speedtest, device)
    updates = micro = 0
    opt.zero_grad()
    for epoch in range(1, args.epochs + 1):
        model.train()
        timer.mark()
        for src, dst, ts, x, _ in tr.raw_edge_iter():
            walks, mask, tgt, attn, feats = raw_lp_inputs(tr, src, dst, ts, x, args.walk_len, delta)
            timer.lap('samp')

            loss = model.modified_fwd(walks, mask, tgt, attn, feats=feats)
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
            timer.dump(f'latency/latency_{name}_lp-raw_{"temporal" if args.trw else "static"}'
                       f'_{args.size}_wl{args.walk_len}.json',
                       dataset=name, size=args.size, temporal=args.trw, walk_len=args.walk_len,
                       mini_bs=mini_bs, accum=accum, updates=updates, edges=tr._col.size(0), device=str(device))
            raise SystemExit

        run_eval(epoch, updates)

    tracker.close()