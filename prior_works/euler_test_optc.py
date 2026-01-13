import os 
import pandas as pd
import time 
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

SPEEDTEST = True
EPOCHS = 15 # Validation is no help. Gets decent scores quickly, then overfits
DEVICE = 1

class Euler(nn.Module):
    def __init__(self, in_dim, hidden, emb_dim, device='cpu'):
        super().__init__()

        self.device = device
        self.gcn1 = GCNConv(in_dim, hidden).to(device)
        self.gcn2 = GCNConv(hidden, hidden).to(device)
        self.rnn = nn.GRU(hidden, emb_dim, batch_first=True, device=device)

    def forward(self, x, eis):
        x = x.to(self.device)
        eis = [add_remaining_self_loops(ei.to(self.device))[0] for ei in eis]

        zs = [torch.relu(self.gcn1(x, ei)) for ei in eis]
        zs = [torch.relu(self.gcn2(zs[i], eis[i])) for i in range(len(zs))]
        zs = torch.stack(zs)

        return self.rnn(zs)[0]

def to_snapshots(g, ts=None):
    # Assumes graph has src and col already
    ei = torch.stack([g.src, g.col])

    eis = []
    y = []
    for t in ts:
        mask = g.ts == t
        ei_t = ei[:, mask]

        eis.append(ei_t)

        if 'label' in g.keys():
            label = g.label[mask]
            y.append(label)

    x = torch.eye(g.x.size(0))
    return Data(x=x, edge_index=eis, label=y)

def train(tr,va,te):
    model = Euler(tr.x.size(0), 128, 64, device=DEVICE)
    opt = Adam(model.parameters(), lr=0.01)
    bce = nn.BCEWithLogitsLoss()

    def calc_loss(zs, eis, grad=True, num_nodes=None):
        tot_loss = 0
        for i in range(len(eis)):
            z = zs[i]; ei = eis[i]
            low = 0; high = num_nodes if num_nodes else ei.max()

            if ei.size(1) == 0:
                continue

            pos = (z[ei[0]] * z[ei[1]]).sum(dim=1)
            neg = (
                z[torch.randint(low, high, (pos.size(0),), device=DEVICE)] *
                z[torch.randint(low, high, (pos.size(0),), device=DEVICE)]
            ).sum(dim=1)

            labels = torch.zeros(pos.size(0)*2, device=DEVICE)
            labels[pos.size(0):] = 1

            loss = bce.forward(
                torch.cat([pos,neg]),
                labels
            )
            tot_loss += loss

        return tot_loss / len(eis)

    best = (100,0,0)
    for e in range(EPOCHS):
        model.train()
        opt.zero_grad()

        st = time.time() 
        zs = model.forward(tr.x, tr.edge_index)
        fwd_time = time.time() - st 

        st = time.time()
        loss = calc_loss(zs, tr.edge_index)
        loss_time = time.time() - st 

        st = time.time()
        loss.backward()
        bwd_time = time.time() - st 
        
        st = time.time()
        opt.step()
        step_time = time.time() - st 

        if SPEEDTEST: 
            with open('euler_speedtest.csv', 'a') as f:
                f.write(f'OpTC,{fwd_time},{loss_time},{bwd_time},{step_time}\n')
            exit()

        print(f'[{e}] Loss: {loss.item():0.4f}')

        if SPEEDTEST: 
            with open('argus_speedtest.csv', 'a') as f:
                f.write(f'LANL,{fwd_time},{loss_time},{bwd_time},{step_time}\n')
            exit()

        print(f'[{e}] Loss: {loss.item():0.4f}')

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index)
            va_loss = calc_loss(zs, va.edge_index, grad=False, num_nodes=1000)
            print(f'\tVal loss: {va_loss:0.4f}')

            preds = []
            ys = []
            for i in range(len(te.edge_index)):
                pred = (
                    zs[i][te.edge_index[i][0]] *
                    zs[i][te.edge_index[i][1]]
                ).sum(dim=1)
                preds.append(pred)

            preds = torch.sigmoid(torch.cat(preds))
            y = torch.cat(te.label).clamp(0,1)

            auc = auc_score(y, preds.cpu())
            ap = ap_score(y, preds.cpu())
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_loss < best[0]:
                best = (va_loss, auc, ap)
                print('*')
            else:
                print()

    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return {'auc': best[1], 'ap': best[2]}

if __name__ == '__main__':
    if not os.path.exists('tmp/euler_optc_te.pt'):
        tr = torch.load('../data/optc_tgraph_tr.pt', weights_only=False)
        tr.ts //= 3600
        va = torch.load('../data/optc_tgraph_va.pt', weights_only=False)
        va.ts //= 3600
        te = torch.load('../data/optc_tgraph_te.pt', weights_only=False)
        te.ts //= 3600

        ts = tr.ts.unique()

        tr = to_snapshots(tr, ts)
        va = to_snapshots(va, ts)
        te = to_snapshots(te, ts)

        torch.save(tr, 'tmp/euler_optc_tr.pt')
        torch.save(va, 'tmp/euler_optc_va.pt')
        torch.save(te, 'tmp/euler_optc_te.pt')
    
    else: 
        tr = torch.load('tmp/euler_optc_tr.pt', weights_only=False)
        va = torch.load('tmp/euler_optc_va.pt', weights_only=False)
        te = torch.load('tmp/euler_optc_te.pt', weights_only=False)


    best = [train(tr,va,te) for _ in range(10)]
    df = pd.DataFrame(best)
    print(df.mean())
    print(df.sem())
    df.to_csv('euler_results.csv')