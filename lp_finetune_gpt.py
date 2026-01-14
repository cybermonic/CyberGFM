from argparse import ArgumentParser
from copy import deepcopy
import os 
import time
from types import SimpleNamespace

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import (
    roc_auc_score as auc_score,
    average_precision_score as ap_score
)
import torch
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from transformers import OpenAIGPTConfig
from tqdm import tqdm

from eval_trw import CausalEvaluator
from fast_auc import fast_auc, fast_ap
from models.gnn_bert import GNNEmbedding
from models.hugging_gpt import GPT 
from sampler import TRWSampler as TRW, RWSampler as RW
from utils import reindex

SPEEDTEST = False
DEVICE = 0
EPOCHS = 36    # Epochs
WARMUP_E = EPOCHS / 3.75 # Originally 36 and 9.6
LR = 3e-4

WALK_LEN = 4
NUM_EVAL_ITERS = 1
MINI_BS = 512
BS = 1024
EVAL_BS = 1024
EVAL_EVERY = 250
T_MAX = 100_000 # From alibaba source code

class Scheduler(LRScheduler):
    def __init__(self, optimizer, warmup_stop, total_steps, last_epoch=-1, verbose="deprecated"):
        self.warmup_stop = warmup_stop
        self.total_steps = total_steps
        self.cosine = CosineAnnealingLR(
            optimizer, T_MAX, last_epoch=-1
        )

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Warmup period
        if self.last_epoch < self.warmup_stop:
            return [group['initial_lr'] * (self.last_epoch / self.warmup_stop)
                    for group in self.optimizer.param_groups]
        else:
            return [
                group['initial_lr'] * (
                max(
                    1e-8, 
                    1 - ((self.last_epoch-self.warmup_stop)/(self.total_steps-self.warmup_stop))
                ))
                for group in self.optimizer.param_groups
            ]

def train(tr,va,te, model: GPT):
    opt = AdamW(
        model.parameters(), lr=LR,
        betas=(0.9, 0.99), eps=1e-10, weight_decay=0.02
    )

    updates_per_epoch = tr.col.size(0) / BS
    warmup_stop = int(updates_per_epoch * WARMUP_E)
    total_steps = int(updates_per_epoch * EPOCHS)

    
    best = 0 #va_ap
    sched = Scheduler(opt, warmup_stop, total_steps)

    if not SPEEDTEST:
        with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'w+') as f:
            f.write(f'epoch,updates,auc,ap,val_auc,val_ap\n')
            
            if not (args.from_random or args.special): 
                f.write(f'0,0,0,0,0,0\n')

    updates = 0
    opt.zero_grad()
    st = time.time()
    steps = 0

    times = dict({
        'samp': [],
        'fwd': [],
        'bwd': [],
        'step': []
    })

    e = 0
    for e in range(5):
        for samp in tr.edge_iter():
            if tr.edge_features:
                src,dst,ts,ef = samp
            else:
                src,dst,ts = samp
                ef = None

            model.train()
            
            log_st = time.time()
            walk = evaluator.sample(tr, src,dst,ts, WALK_LEN, edge_features=ef)
            times['samp'].append(time.time() - log_st)

            log_st = time.time()
            loss = model.modified_fwd(walk)
            times['fwd'].append(time.time() - log_st)

            log_st = time.time()
            loss.backward()
            times['bwd'].append(time.time() - log_st)

            steps += 1
            if steps*MINI_BS == BS:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                
                log_st = time.time()
                opt.step()
                times['step'].append(time.time() - log_st)

                sched.step()
                en = time.time()

                lr = sched.get_last_lr()[0]
                print(f'[{updates}-{e}] {loss} (lr: {lr:0.2e}, {en-st:0.2f}s)')

                updates += 1
                steps = 0
                opt.zero_grad()
                st = time.time()

            '''
            Only consider results of full epochs 
            Speeds up eval time, and has better results 
            
            if updates and updates % EVAL_EVERY == 0:
                model.eval()
                te_auc, te_ap, va_auc, va_ap = get_metrics(tr,va,te, model)

                if va_ap > best:
                    best = va_ap
                    best_te = (te_auc, te_ap, va_auc, va_ap)

                with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
                    f.write(f'{e+1},{updates},{te_auc},{te_ap},{va_auc},{va_ap}\n')

                auc, ap, va_auc, va_ap = best_te
                print('#'*20)
                print(f'BEST SCORES')
                print('#'*20)
                print(f"VAL:  AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
                print(f"TEST: AUC: {auc:0.4f}, AP:  {ap:0.4f}")
            '''

        if SPEEDTEST: 
            import json 
            with open(f'latency/latency_{DATASET}_ft.json', 'w+') as f:
                f.write(
                    json.dumps(times, indent=1)
                )
            exit()

        model.eval()
        te_auc, te_ap, va_auc, va_ap = evaluator.get_metrics(tr,va,te, model)

        if va_ap > best:
            best = va_ap
            best_te = (te_auc, te_ap, va_auc, va_ap, e)

        with open(f'{HOME}/{DATASET}/snapshot-ft_results_{FNAME}_{SIZE}_wl{WALK_LEN}.txt', 'a') as f:
            f.write(f'{e+1},{updates},{te_auc},{te_ap},{va_auc},{va_ap}\n')

        auc, ap, va_auc, va_ap, _ = best_te
        print('#'*20)
        print(f'BEST SCORES')
        print('#'*20)
        print(f"VAL:  AUC: {va_auc:0.4f}, AP:  {va_ap:0.4f}")
        print(f"TEST: AUC: {auc:0.4f}, AP:  {ap:0.4f}")

    return best_te

def special(tr,va,te, model,sd,tag,out_dir):
    global EPOCHS, WARMUP_E

    with open(f'{out_dir}/epoch_ablation-wl{WALK_LEN}{tag}.txt', 'w+') as f:
        f.write('e,e_best,auc,ap,v_auc,v_ap\n')

    for e in [1,2,3,5]:
        EPOCHS = e 
        WARMUP_E = EPOCHS / 3.75 

        model.load_state_dict(sd)
        model.to(DEVICE)
        auc,ap,v_auc,v_ap,e_best = train(tr,va,te, model)

        with open(f'{out_dir}/epoch_ablation-wl{WALK_LEN}{tag}.txt', 'a') as f:
            f.write(f'{e},{e_best},{auc},{ap},{v_auc},{v_ap}\n')


if __name__ == '__main__':
    arg = ArgumentParser()
    arg.add_argument('--size', default='tiny')
    arg.add_argument('--device', type=int, default=0)
    arg.add_argument('--walk-len', type=int, default=4)
    arg.add_argument('--optc', action='store_true')
    arg.add_argument('--unsw', action='store_true')
    arg.add_argument('--argus', action='store_true')
    arg.add_argument('--trw', action='store_true')
    arg.add_argument('--best', action='store_true')
    arg.add_argument('--from-random', action='store_true')
    arg.add_argument('--tr-size', type=float, default=1.)
    arg.add_argument('--model-fname', default='')
    arg.add_argument('--tag', default='')
    arg.add_argument('--special', action='store_true')
    arg.add_argument('--out-dir', default='')
    args = arg.parse_args()
    print(args) 

    SIZE = args.size
    DEVICE = args.device if args.device >= 0 else 'cpu'
    WALK_LEN = args.walk_len
    DATASET = 'optc' if args.optc else 'unsw' if args.unsw else \
        'lanl14argus' if args.argus else 'unknown'
    WORKERS = 16
    EVAL_EVERY = 1000

    if args.out_dir: 
        HOME = args.out_dir
    else: 
        HOME = f'results/lp-{"temporal" if args.trw else "static"}/'

    edge_features = args.unsw or args.lanlflows or args.lanlcomp or args.argus

    params = {
        'tiny': SimpleNamespace(H=128, L=2, MINI_BS=1024),
        'mini': SimpleNamespace(H=256, L=4, MINI_BS=1024),
        'med': SimpleNamespace(H=512, L=8, MINI_BS=1024),
        'baseline': SimpleNamespace(H=768, L=12, MINI_BS=512)
    }[SIZE]
    MINI_BS = params.MINI_BS

    print(DATASET)

    if args.from_random: 
        sd = None 
    
    else: 
        # Option to provide explicit model name
        if args.model_fname: 
            sd = torch.load(args.model_fname,  weights_only=True)
        
        # Otherwise, it's inferred from args
        else: 
            if args.trw:
                sd = torch.load(f'pretrained/temporal/{DATASET}/trw_bert_{DATASET}_{SIZE}{"-best" if args.best else ""}.pt', weights_only=True)
            else:
                sd = torch.load(f'pretrained/static/{DATASET}/rw_bert_{DATASET}_{SIZE}{"-best" if args.best else ""}.pt', weights_only=True)

    FNAME = (
        f'{"rand_init_" if args.from_random else ""}' + 
        f'snapshot_bert{"_static" if not args.trw else ""}{"_best-val" if args.best else ""}' + args.tag
    )
    print(FNAME)

    TRWSampler = TRW if args.trw else RW

    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    
    # For training set size ablation study
    tr = torch.load(f'data/{DATASET}_tgraph_tr.pt', weights_only=False)
    if args.tr_size != 1: 
        HOME = 'results/training_data_ablation/'
        FNAME += f'_{args.tr_size:0.4f}pct'

        if not os.path.exists(f'subsets/{DATASET}.pt'): 
            perturb = torch.randperm(tr.col.size(0))
        else: 
            perturb = torch.load(f'subsets/{DATASET}.pt', weights_only=True)

        perturb = perturb[: int(perturb.size(0) * args.tr_size)]

        # Need to keep everything in same order, so use mask instead of index
        to_keep = torch.zeros(tr.col.size(0), dtype=torch.bool)
        to_keep[perturb] = 1

        tr.col = tr.col[to_keep]
        tr.src = tr.src[to_keep]
        tr.ts = tr.ts[to_keep]
        tr.idxptr = reindex(tr.src, tr.x.size(0))

        if 'edge_attr' in tr.keys(): 
            tr.edge_attr = tr.edge_attr[to_keep]

    tr = TRWSampler(tr, device=DEVICE, walk_len=WALK_LEN, batch_size=MINI_BS, edge_features=edge_features)

    va = torch.load(f'data/{DATASET}_tgraph_va.pt', weights_only=False)
    va = TRWSampler(va, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    va.label = torch.zeros_like(va.col)

    te = torch.load(f'data/{DATASET}_tgraph_te.pt', weights_only=False)
    label = te.label
    te = TRWSampler(te, walk_len=WALK_LEN, batch_size=EVAL_BS, edge_features=edge_features)
    te.label = label

    if DATASET == 'lanl14argus': 
        DELTA = 60*60
        SNAPSHOTS = tr.ts.unique().tolist()
        WORKERS = 1
        EVAL_EVERY = 1000

        if WALK_LEN > 8: 
            MINI_BS = 256
            tr.batch_size = MINI_BS
            EVAL_BS = 512
        if WALK_LEN > 16: 
            EVAL_BS = 256
        if WALK_LEN > 32: 
            EVAL_BS = 128
            MINI_BS = 128
            tr.batch_size = 128


    elif DATASET == 'unsw':
        WORKERS = 8
        DELTA = 0
        SNAPSHOTS = tr.ts.unique().tolist()
        EVAL_EVERY = 500

        # OOM
        if WALK_LEN > 16:
            WORKERS = 4

    elif DATASET == 'optc':
        DELTA = 60*60*24
        SNAPSHOTS = (tr.ts // DELTA).unique().tolist()[:5]
        WORKERS = 1
        EVAL_BS = 2048*2

    else:
        print(f"Unrecognized dataset: {DATASET}")

    config = OpenAIGPTConfig(
        tr.num_tokens + GNNEmbedding.OFFSET,
        hidden_size=         params.H,
        num_hidden_layers=   params.L,
        num_attention_heads= params.H // 64,
        intermediate_size=   params.H * 4,
        num_nodes = tr.num_tokens,
        max_position_embeddings = 1024 if args.argus else 512
    )
    model = GPT(config)

    evaluator = CausalEvaluator(
        args.walk_len,
        dataset=DATASET, device=DEVICE,
        delta=DELTA, workers=WORKERS,
        eval_bs=EVAL_BS
    )
    
    if not args.from_random:
        model.load_state_dict(sd)
    
    model = model.to(DEVICE)

    if args.special: 
        special(tr,va,te, model,sd,args.tag,args.out_dir)
    else: 
        train(tr,va,te, model)

