'''
Link-prediction fine-tuning: given a walk ending at u (+ the u->v edge features),
predict v at a [MASK] position. Anomaly score = 1 - sigmoid(logit_v).

    python lp_finetune.py --dataset lanl14argus --device 0
    python lp_finetune.py --dataset lanl14argus-dirtyts --trw --walk-len 8 --device 0
    python lp_finetune.py --dataset unsw --poison 10 --device 0
    python lp_finetune.py --dataset lanl14argus --device 0 --speedtest   # one epoch, timings only
'''
from argparse import ArgumentParser
import math

import torch
from torch.optim import AdamW

from common import (VAL_NEG, DAY, HOUR, ResultTracker, SpeedTimer, bert_config, dataset_kind, evaluate, lp_inputs,
                    load_eval_samplers, load_train_graph, make_lp_scorer, make_sampler,
                    pretrained_path, run_tag, seed_everything, uses_edge_features, warmup_linear)
from models.gnn_bert import RWBert

BS = 1024


def defaults(kind, size, walk_len):
    delta = {'lanl': HOUR, 'unsw': 0, 'optc': DAY}[kind]
    mini_bs = 512 if size == 'baseline' else 1024
    eval_bs = {'lanl': 1024, 'unsw': 1024, 'optc': 4096}[kind]
    if kind == 'lanl':
        if walk_len > 4:
            mini_bs = min(mini_bs, 512)
        if walk_len > 8:
            mini_bs, eval_bs = 256, 512
        if walk_len > 16:
            eval_bs = 256
        if walk_len > 32:
            mini_bs, eval_bs = 128, 128
    return delta, mini_bs, eval_bs


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--size', default='tiny', choices=['tiny', 'mini', 'med', 'baseline'])
    ap.add_argument('--device', type=int, default=0)
    ap.add_argument('--trw', action='store_true')
    ap.add_argument('--ignore-edge-feats', action='store_true')
    ap.add_argument('--undirected-walks', action='store_true',
                    help='draw the context walks on the undirected training graph. The edges that are '
                         'fine-tuned on and the validation / test edges that are scored stay directed')
    ap.add_argument('--walk-len', type=int, default=4)
    ap.add_argument('--delta', type=int, help='temporal walk window (seconds)')
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--warmup-epochs', type=float, help='default: epochs / 3.75')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=0.02)
    ap.add_argument('--mini-bs', type=int)
    ap.add_argument('--eval-bs', type=int)
    ap.add_argument('--select-by', choices=['auc', 'ap'], default='auc')
    ap.add_argument('--val-neg', choices=VAL_NEG, default='uniform',
                    help='validation negatives (see common.evaluate);')
    ap.add_argument('--score', choices=['sigmoid', 'nll'], default='sigmoid')
    ap.add_argument('--from-random', action='store_true')
    ap.add_argument('--model-fname', help='default: pretrained/{static,temporal}/{dataset}/...')
    ap.add_argument('--best-pretrained', action='store_true', help='load the -best pretraining checkpoint')
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
    delta, mini_bs, eval_bs = defaults(kind, args.size, args.walk_len)
    delta = delta if args.delta is None else args.delta
    mini_bs = args.mini_bs or mini_bs
    eval_bs = args.eval_bs or eval_bs
    warmup_epochs = args.epochs / 3.75 if args.warmup_epochs is None else args.warmup_epochs
    accum = math.ceil(BS / mini_bs)

    tr_data = load_train_graph(name, edge_features, args.poison, args.tr_size)
    tr = make_sampler(tr_data, args.trw, edge_features, args.walk_len, mini_bs, device,
                      undirected_walks=args.undirected_walks)
    va, te = load_eval_samplers(name, args.trw, edge_features, args.walk_len, eval_bs)

    model = RWBert(bert_config(name, tr, args.size))
    if not args.from_random:
        path = args.model_fname or pretrained_path(name, args.size, args.trw, args.poison,
                                                   args.tr_size, best=args.best_pretrained, tag=args.pretrain_tag)
        print('Loading', path)
        model.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
    model = model.to(device)

    updates_per_epoch = math.ceil(tr.col.size(0) / (mini_bs * accum))
    total_steps = updates_per_epoch * args.epochs
    opt = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=1e-10, weight_decay=args.wd)
    sched = warmup_linear(opt, int(updates_per_epoch * warmup_epochs), total_steps)
    scorer = make_lp_scorer(model, tr, args.walk_len, delta, args.score)
    nll_scorer = make_lp_scorer(model, tr, args.walk_len, delta, 'nll')

    out_dir = args.out_dir or f'results/lp-{"temporal" if args.trw else "static"}/{name}'
    stem = (f'{"rand_init_" if args.from_random else ""}lp{run_tag(args.poison, args.tr_size)}{args.pretrain_tag}'
            f'_{args.size}_wl{args.walk_len}{args.tag}')
    tracker = ResultTracker(f'{out_dir}/{stem}.csv', args.select_by)
    print(f'{name}: edge features={edge_features}, delta={delta}, mini_bs={mini_bs} x{accum}, '
          f'{total_steps} updates, walks={"undirected" if args.undirected_walks else "directed"}')

    def run_eval(epoch, updates):
        model.eval()
        torch.cuda.empty_cache()
        tracker.log(epoch, updates, evaluate(scorer, tr, va, te, eval_bs, seed=args.seed, val_neg=args.val_neg, nll_fn=nll_scorer))

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
            walk, mask, tgt, attn = lp_inputs(tr, src, dst, ts, ef, args.walk_len, delta)
            timer.lap('samp')

            loss = model.modified_fwd(walk, mask, tgt, attn)
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
            timer.dump(f'latency/latency_{name}_lp{run_tag(args.poison, args.tr_size)}_{args.size}_wl{args.walk_len}.json',
                       dataset=name, size=args.size, temporal=args.trw, edge_features=edge_features,
                       walk_len=args.walk_len, mini_bs=mini_bs, accum=accum, updates=updates,
                       edges=tr.col.size(0), device=str(device))
            raise SystemExit

        run_eval(epoch, updates)

    tracker.close()