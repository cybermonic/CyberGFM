from argparse import ArgumentParser
import pandas as pd
import time
import torch
from torch import nn
from torch.optim import Adam
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE, GIN
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

SPEEDTEST = True
PATIENCE = 25
DEVICE = 2

class Euler(nn.Module):
    def __init__(self, in_dim, hidden, emb_dim, device='cpu', gin=False):
        super().__init__()

        self.device = device
        self.gin = gin

        if not gin:
            self.gnn = GraphSAGE(in_dim, hidden, 2, out_channels=emb_dim).to(DEVICE)
        else:
            self.gnn = GIN(in_dim, hidden, 2, out_channels=emb_dim).to(DEVICE)

    def forward(self, x, ei):
        x = x.to(self.device)
        ei = add_remaining_self_loops(ei.to(self.device))[0]
        zs = self.gnn(x,ei)

        return zs


def train(tr,va,te, gin):
    model = Euler(tr.x.size(0), 128, 64, device=DEVICE, gin=gin)
    opt = Adam(model.parameters(), lr=0.01)
    bce = nn.BCEWithLogitsLoss()

    def calc_loss(z, ei, grad=True, num_nodes=None):
        low = 0; high = num_nodes if num_nodes else ei.max()

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

        return loss

    best = (100,0,0)
    time_since_best = 0
    e = 0
    while True:
        BS=32
        fwd_time=bwd_time=loss_time=step_time = 0

        model.train()
        opt.zero_grad()

        st = time.time()
        zs = model.forward(tr.x, tr.edge_index)
        fwd_time += time.time() - st

        st = time.time()
        loss = calc_loss(zs, tr.edge_index)
        loss_time += time.time() - st

        st = time.time()
        loss.backward()
        bwd_time += time.time() - st

        st = time.time()
        opt.step()
        step_time += time.time() - st

        print(f'[{e}] Loss: {loss.item():0.4f}')

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index)
            va_loss = calc_loss(zs, va.edge_index, grad=False)
            print(f'\tVal loss: {va_loss:0.4f}')

            preds = (
                zs[te.edge_index[0]] *
                zs[te.edge_index[1]]
            ).sum(dim=1)

            y = te.label.clamp(0,1)

            auc = auc_score(y, preds.cpu())
            ap = ap_score(y, preds.cpu())
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_loss < best[0]:
                best = (va_loss, auc, ap)
                print('*')
                time_since_best = 0
            else:
                print()
                time_since_best += 1

            if time_since_best >= PATIENCE:
                break

        e += 1



    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return {'auc': best[1], 'ap': best[2]}

def preprocess(g, uq=False):
    d = g.idxptr[1:] - g.idxptr[:-1]
    src = torch.arange(0, g.x.size(0)).repeat_interleave(d)
    ei = torch.stack([src, g.col])

    if uq:
        ei = ei.unique(dim=1)

    if hasattr(g, 'label'):
        label = g.label
    else:
        label = None

    return Data(
        x=torch.eye(g.x.size(0)),
        edge_index=ei,
        label=label
    )

if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--lanl', action='store_true')
    ap.add_argument('--unsw', action='store_true')
    ap.add_argument('--optc', action='store_true')
    ap.add_argument('--gin', action='store_true')
    args = ap.parse_args()

    if args.lanl:
        ds = 'lanl14argus'
    elif args.unsw:
        ds = 'unsw'
    else: # args.optc:
        ds = 'optc'
    #else:
    #    raise Exception('Provide one of --lanl, ---unsw, --optc')

    tr = torch.load(f'../data/{ds}_tgraph_tr.pt', weights_only=False)
    tr = preprocess(tr, uq=True)
    va = torch.load(f'../data/{ds}_tgraph_va.pt', weights_only=False)
    va = preprocess(va)
    te = torch.load(f'../data/{ds}_tgraph_te.pt', weights_only=False)
    te = preprocess(te)

    best = [train(tr,va,te, args.gin) for _ in range(10)]
    df = pd.DataFrame(best)
    print(df.mean())
    print(df.sem())

    if args.gin:
        tag = 'gin'
    else:
        tag = 'sage'
    df.to_csv(f'{tag}_results_{ds}.csv')