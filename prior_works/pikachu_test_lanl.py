from collections import defaultdict 
import json 
from math import ceil 
import os 
import time

import pandas as pd 
import torch 
from torch.optim import Adam, SGD
from torch_geometric.data import Data 
from torch_geometric.utils import to_undirected
from tqdm import tqdm 
from sklearn.metrics import roc_auc_score as auc_score, average_precision_score as ap_score

from pikachu_models import CTDNE, Autoencoder, AnomalyDetector 

DEVICE = 2

# Defaults from Word2Vec and Pikachu code 
W2V_EPOCHS = 5 
W2V_EMB = 128 
WALK_LEN = 500
CTXT_SIZE = 5 
AE_EPOCHS = 50  # Default 50
AE_LR = 0.001 # Default keras lr 
AE_HIDDEN = 128
AE_LATENT = 64
ANOM_LR = 0.001
ANOM_EPOCHS = 10
ANOM_BATCH_SIZE = 5 
TRAIN_WIN = 5 # How many snapshots per batch for anomaly detection

SPEEDTEST = True 

def preprocess(g, ts, undirected=False): 
    '''
    Slice input CSR graph into discrete time units
    '''
    snapshots = []

    for t in tqdm(ts, desc='Preprocessing'): 
        csr = defaultdict(list)    
        src = g.src[g.ts == t]
        dst = g.col[g.ts == t]

        if undirected: 
            src,dst = to_undirected(torch.stack([src,dst]))

        for i in range(src.size(0)): 
            csr[src[i].item()].append(dst[i].item())

        ptr = [0]
        col = []
        for i in range(g.x.size(0)): 
            col_n = csr[i] 
            ptr.append(ptr[-1] + len(col_n))
            col += col_n 

        if 'label' in g.keys(): 
            y = g.label[g.ts == t]
        else: 
            y = None 

        g_t = Data(
            x = g.x, 
            edge_index = torch.stack([src,dst]), 
            rowptr = torch.tensor(ptr), 
            col = torch.tensor(col),
            label = y 
        )
        snapshots.append(g_t)

    return snapshots 

def get_node_embeddings(tr): 
    embs = []

    for tr_t in tqdm(tr, desc='Running n2v'): 
        model = CTDNE(W2V_EMB, WALK_LEN, CTXT_SIZE, tr_t.x.size(0), device=DEVICE)
        opt = Adam(model.parameters(), lr=0.025)

        for e in range(1, W2V_EPOCHS+1): 
            BS = tr_t.x.size(0) // 4
            batches = torch.randperm(tr_t.x.size(0))
            for batch in range(ceil(tr_t.x.size(0) / BS)):
                st = batch*BS 
                en = (batch+1)*BS

                opt.zero_grad()
                pos, neg = model.sample(batches[st:en].to(DEVICE), tr_t.to(DEVICE))
                loss = model.loss(pos, neg)
                loss.backward()
                opt.step()
        
        with torch.no_grad():
            model.eval()
            embs.append(model.forward().detach())

    return embs 

def train_ae(emb_list): 
    embs = torch.stack(emb_list, dim=1).to(DEVICE)
    model = Autoencoder(embs.size(-1), AE_HIDDEN, AE_LATENT, device=DEVICE)
    opt = Adam(model.parameters(), lr=AE_LR)

    BS = 2048
    n_batches = ceil(embs.size(0) / BS)

    print("Training GRU Autoencoder")
    for e in range(AE_EPOCHS):  
        batches = torch.randperm(embs.size(0))
        
        model.train()
        opt.zero_grad() 
        for i in range(n_batches):
            emb = embs[i*BS : (i+1)*BS]
            loss = model.forward(emb)
            loss.backward()
        opt.step()

        print(f'\t[{e}] {loss.item()}')

    st = time.time()
    with torch.no_grad(): 
        model.eval()

        z = torch.empty((tr[0].x.size(0), len(tr), AE_LATENT), device=DEVICE)
        batches = torch.arange(embs.size(0))
        n_batches = ceil(batches.size(0) / BS)

        for i in range(n_batches):
            emb = embs[i*BS : (i+1)*BS]
            z[i*BS : min((i+1)*BS, z.size(0))] = model.enc(emb)
    
    eval_time = time.time() - st
    return z, eval_time

def train_anom(z, tr, va, te): 
    '''
    z: N x T x d matrix of node embs
    '''
    model = AnomalyDetector(AE_LATENT, z.size(0), device=DEVICE)
    
    # Pikachu src code does gradient descent by hand using SGD 
    # Experiments showed Adam does a much better job
    opt = Adam(model.parameters(), lr=ANOM_LR) 
    
    def evaluate(g): 
        model.eval()
        edges = []
        if 'label' not in g[0].keys(): 
            y = []
            for g_ in g:
                y_ = torch.zeros(g_.edge_index.size(1)*2)
                y_[g_.edge_index.size(1):] = 1
                
                edges.append(
                    torch.cat([
                        g_.edge_index, 
                        torch.randint(0, g_.x.size(0), g_.edge_index.size())
                    ], dim=1)
                )
                y.append(y_)
            
            y = torch.cat(y) 

        else: 
            y = torch.cat([g[i].label for i in range(len(g))])
            edges = [g[i].edge_index for i in range(len(g))]
        
        y_hat = []
        for i in range(z.size(1)): 
            y_hat.append(model.get_score(z[:, i], edges[i], tr[i].rowptr, tr[i].col).detach())
        y_hat = torch.cat(y_hat).cpu()

        auc = auc_score(y, y_hat)
        ap = ap_score(y, y_hat)
        
        return auc,ap 

    best = (0,0,0)
    snooped = (0,0)
    eval_time = 0 

    print("Training anomaly detector")
    for e in range(1,ANOM_EPOCHS): 
        opt.zero_grad()
        model.train()
        for mb,t in enumerate(torch.randperm(z.size(1))): 
            loss = model.forward(z[:, t], tr[t].edge_index, tr[t].rowptr, tr[t].col)
            loss.backward() 

            if mb and mb % ANOM_BATCH_SIZE == 0:
                opt.step()
                opt.zero_grad()

            print(f'\t[{e}-{mb}] {loss.item()}')
        opt.step()

        st = time.time()
        with torch.no_grad(): 
            model.eval()
            va_auc,va_ap = evaluate(va)
            print(f'\tVa: AUC {va_auc:0.4f}, AP {va_ap:0.4f}')
            te_auc,te_ap = evaluate(te) 
            print(f'\tTe: AUC {te_auc:0.4f}, AP {te_ap:0.4f}')
        
        if va_auc > best[0]: 
            best = (va_auc, te_auc, te_ap)
        if te_auc > snooped[0]: 
            snooped = (te_auc, te_ap)
        
        en = time.time()
        eval_time += en-st 

    return {
        'best-auc': best[1], 
        'best-ap': best[2], 
        'snooped-auc': snooped[0],
        'snooped-ap': snooped[1], 
        'last-auc': te_auc, 
        'last-ap': te_ap
    }, eval_time 

def train_full(tr,va,te): 
    st = time.time()
    embs = get_node_embeddings(tr) 
    en = time.time() 
    with open('pikachu_lanl_times.txt', 'a') as f:
       f.write(f'emb,{en-st},{W2V_EPOCHS}\n')

    embs = [torch.rand((tr[0].x.size(0), W2V_EMB)) for _ in range(len(tr))]
    st = time.time()
    embs, eval_t = train_ae(embs)
    en = time.time()
    
    with open('pikachu_lanl_times.txt', 'a') as f:
        f.write(f'ae,{en-st-eval_t},{AE_EPOCHS}\n')
        f.write(f'ae-eval,{eval_t}\n')

    st = time.time()
    stats, eval_t = train_anom(embs, tr,va,te)
    en = time.time()

    with open('pikachu_lanl_times.txt', 'a') as f:
        f.write(f'anom,{en-st-eval_t},{ANOM_EPOCHS}\n')
        f.write(f'anom-eval,{eval_t}\n')

    if SPEEDTEST: 
        exit()

    print(json.dumps(stats, indent=1))
    return stats 

if __name__ == '__main__': 
    if not os.path.exists('tmp/pika_lanl_tr.pt'):
        tr = torch.load('../data/lanl14argus_tgraph_tr.pt', weights_only=False)
        ts = tr.ts.unique()

        tr = preprocess(tr, ts, undirected=True)
        va = preprocess(torch.load('../data/lanl14argus_tgraph_va.pt', weights_only=False), ts)
        te = preprocess(torch.load('../data/lanl14argus_tgraph_te.pt', weights_only=False), ts)

        torch.save(tr, 'tmp/pika_lanl_tr.pt')
        torch.save(va, 'tmp/pika_lanl_va.pt')
        torch.save(te, 'tmp/pika_lanl_te.pt')
        
    else: 
        tr = torch.load('tmp/pika_lanl_tr.pt', weights_only=False)
        va = torch.load('tmp/pika_lanl_va.pt', weights_only=False)
        te = torch.load('tmp/pika_lanl_te.pt', weights_only=False)

    stats = [
        train_full(tr,va,te)
        for _ in range(10)
    ]
    df = pd.DataFrame(stats)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()
    
    df.to_csv('pikachu_results.csv')