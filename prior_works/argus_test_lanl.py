from collections import defaultdict
import json
import os
import time

import pandas as pd
import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.metrics import \
    roc_auc_score as auc_score, \
    average_precision_score as ap_score

from argus_opt import SOAP
from fast_auc import fast_auc, fast_ap

SPEEDTEST = True 
EPOCHS = 100
DEVICE = 0

def squared_loss(margin, t): return (margin - t)** 2
def squared_hinge_loss(margin, t): return torch.max(margin - t, torch.zeros_like(t)) ** 2
def logistic_loss(margin, t): return torch.log(1+torch.log(-margin*t))

def _check_tensor_shape(inputs, shape=(-1, 1)):
    input_shape = inputs.shape
    target_shape = shape
    if len(input_shape) != len(target_shape):
        inputs = inputs.reshape(target_shape)
    return inputs

def _get_surrogate_loss(backend='squared_hinge'):
    if backend == 'squared_hinge':
       surr_loss = squared_hinge_loss
    elif backend == 'squared':
       surr_loss = squared_loss
    elif backend == 'logistic':
       surr_loss = logistic_loss
    else:
        raise ValueError('Out of options!')
    return surr_loss


class APLoss(torch.nn.Module):
    """AP Loss with squared-hinge function: a novel loss function to directly optimize AUPRC.

        Args:
            margin: margin for squred hinge loss, e.g., m in [0, 1]
            gamma: factors for moving average

        Return:
            loss value (scalar)

        Reference:
                Stochastic Optimization of Areas Under Precision-Recall Curves with Provable Convergence},
                Qi, Qi and Luo, Youzhi and Xu, Zhao and Ji, Shuiwang and Yang, Tianbao,
                Advances in Neural Information Processing Systems 2021.

        Notes: This version of AP loss reduces redundant computation for the original implementation by Qi Qi.
        In addition, it fixed a potential memory leaking issue related to 'u_all' and 'u_pos'. This version is contributed by Gang Li.
    """

    def __init__(self, pos_len=None, margin=1.0, gamma=0.99, surrogate_loss='squared_hinge', device=None):
        super(APLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # self.device = torch.device('cpu')

        else:
            self.device = device
        self.u_all = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.u_pos = torch.tensor([0.0]*pos_len).reshape(-1, 1).to(self.device)#.cpu()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = _get_surrogate_loss(surrogate_loss)

    def forward(self, y_pred, y_true, index_p):
        CHECKMEORY=False
        y_pred = _check_tensor_shape(y_pred, (-1, 1))
        y_true = _check_tensor_shape(y_true, (-1, 1))
        index_p = _check_tensor_shape(index_p, (-1,))
        index_p = index_p[index_p>=0] # only need indices from positive samples

        pos_mask = (y_true == 1).flatten()
        f_ps = y_pred[pos_mask]
        mat_data = y_pred.flatten().repeat(len(f_ps), 1)

        sur_loss = self.surrogate_loss(self.margin, (f_ps - mat_data))#.detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        pos_sur_loss = sur_loss * pos_mask
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_all[index_p] = (1 - self.gamma) * self.u_all[index_p] + self.gamma * (sur_loss.mean(1, keepdim=True)).detach() # memory leak here
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        self.u_pos[index_p] = (1 - self.gamma) * self.u_pos[index_p] + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        p = (self.u_pos[index_p] - (self.u_all[index_p]) * pos_mask) / (self.u_all[index_p] ** 2) # size of p: len(f_ps)* len(y_pred)
        # p.detach_()
        loss = torch.mean(p * sur_loss)
        # loss = (p * sur_loss).mean()
        if CHECKMEORY: print(torch.cuda.max_memory_allocated())

        # del sur_loss, pos_sur_loss, p
        return loss

class GRU(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, hidden_units=1):
        super(GRU, self).__init__()

        self.rnn = nn.GRU(
            x_dim, h_dim, num_layers=hidden_units
        )

        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

        self.z_dim = z_dim

    def forward(self, xs, h0, include_h=False):
        xs = self.drop(xs)

        if isinstance(h0, type(None)):
            xs, h = self.rnn(xs)
        else:
            xs, h = self.rnn(xs, h0)

        if not include_h:
            return self.lin(xs)

        return self.lin(xs), h

class Argus(nn.Module):
    def __init__(self, in_dim, edge_dim, h_dim, z_dim, device, s=5, pos_samples=315163):
        super().__init__()

        self.c1 = GCNConv(in_dim, h_dim).to(device)
        self.relu = nn.ReLU()
        self.c2 = GCNConv(h_dim, h_dim).to(device)
        self.drop = nn.Dropout(0.1)
        self.ac = nn.Tanh()
        self.c3 = GCNConv(h_dim, h_dim).to(device)
        nn4 = nn.Sequential(nn.Linear(edge_dim, 8, device=device), nn.ReLU(), # lanl: 3 or 10; optc: 5
                            nn.Linear(8, h_dim * h_dim, device=device))

        self.c4 = NNConv(h_dim, h_dim, nn4, aggr='mean').to(device)
        self.rnn = GRU(h_dim, h_dim, z_dim).to(device)

        self.decode_mlp = nn.Sequential(
            nn.Linear(z_dim, z_dim, device=device),
            nn.Softmax(dim=1)
        )

        self.device = device
        # UNSW
        # self.ap_loss = APLoss(pos_len=1775035, margin=0.8, gamma=0.1, surrogate_loss='squared', device=device)
        self.ap_loss = APLoss(pos_len=pos_samples, margin=0.8, gamma=0.01, surrogate_loss='squared', device=device)
        self.s = s


    def forward(self, x, eis, eas, idxs, ptrs):
        eis = [ei.to(self.device) for ei in eis]
        eas = [ea.to(self.device) for ea in eas]
        zs = []

        for i in range(len(eis)):
            ei = eis[i]; ea = eas[i]
            ei_self_loops = add_remaining_self_loops(ei)[0]

            z = self.c1(x.to(self.device), ei_self_loops)
            z = self.c2(z, ei_self_loops) # Comment out for UNSW
            z = self.relu(z)
            z = self.drop(z)
            z = self.c3(z, ei_self_loops) # Comment out for UNSW
            z = self.relu(z) # Comment out for UNSW
            z = self.drop(z) # Comment out for UNSW
            z = self.c4(z, ei, edge_attr=ea)
            z = self.ac(z)

            zs.append(z)

        zs = torch.stack(zs, dim=1)
        out = self.rnn(zs, None)

        '''
        Scores are much better when we skip the aggregation step
        AUC 0.05 -> what's reported (at least 0.6, but it's still training)
            Skipped in UNSW
        '''
        zs = []
        for t in range(out.size(1)):
            z = out[:, t, :]
            z = self.sample_z(z, idxs[i], ptrs[i])
            zs.append(z)

        out = torch.stack(zs)

        # out = out.transpose(1,0) # Uncomment if skipping aggr
        return out

    def sample_z(self, z, idx,ptr):
        z_agg = []
        idx = idx.to(self.device)
        ptr = ptr.to(self.device)

        for i in range(z.size(0)):
            n_neighbors = idx[i+1]-idx[i]

            if n_neighbors:
                neighbors = ptr[idx[i] + torch.ones(n_neighbors, device=self.device).multinomial(self.s, replacement=True)]
                z_agg.append((z[neighbors].sum(dim=0) + z[i]) / (self.s+1))
            else:
                z_agg.append(z[i])

        z_agg = torch.stack(z_agg)
        return self.decode_mlp(z_agg)

    def decode(self, ei, z):
        return (z[ei[0]] * z[ei[1]]).sum(dim=1)

    def calc_loss_argus(self, zs, eis):
        tot_loss = torch.zeros(1).to(self.device)
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue

            ns = torch.randint(0, zs.size(1), ps.size())

            t_index = torch.arange(0, ps.size(1), dtype=torch.int64, device=self.device).detach()
            neg_pred = self.decode(ps, zs[i])
            pos_pred = self.decode(ns, zs[i])

            tot_loss += self.ap_loss(
                torch.cat((pos_pred, neg_pred), 0),
                torch.cat((torch.ones(pos_pred.size(0)),torch.zeros(neg_pred.size(0))), 0).to(self.device).detach(),
                t_index)

        return tot_loss.true_divide(len(zs))

    def validate(self, zs, eis):
        pos, neg = [],[]
        for i in range(len(zs)):
            ps = eis[i]
            if ps.size(1) == 0:
                continue

            ns = torch.randint(0, zs.size(1), ps.size())
            pos_pred = self.decode(ps, zs[i])
            neg_pred = self.decode(ns, zs[i])

            pos.append(pos_pred)
            neg.append(neg_pred)

        pos = torch.cat(pos)
        neg = torch.cat(neg)
        return pos, neg


def to_snapshots(g, ts=None, add_csr=False, last_ts=None):
    # Assumes graph has src and col already
    ei = torch.stack([g.src, g.col])

    # Normalize (already done for LANL)
    #g.raw_edge_attr = g.raw_edge_attr / g.raw_edge_attr.max(dim=0).values

    eis = []
    eas = []
    idxs = []
    ptrs = []
    y = []
    for t in ts:
        if last_ts and t > last_ts:
            continue

        mask = g.ts == t
        ei_t = ei[:, mask]
        ea_t = g.raw_edge_attr[mask]

        eis.append(ei_t)
        eas.append(ea_t)

        idx = [0]
        ptr = []

        if add_csr:
            csr_dict = defaultdict(list)
            for i in range(ei_t.size(1)):
                src,dst = ei_t[:, i]
                csr_dict[src.item()].append(dst.item())

            for i in range(g.x.size(0)):
                ptr += csr_dict[i]
                idx.append(idx[-1] + len(csr_dict[i]))

            idxs.append(torch.tensor(idx))
            ptrs.append(torch.tensor(ptr))

        if 'label' in g.keys():
            label = g.label[mask]
            y.append(label)

    x = torch.eye(g.x.size(0))
    return Data(x=x, edge_index=eis, label=y, eas=eas, idxs=idxs, ptrs=ptrs)

def train(tr,va,te):
    pos_samples = sum([tr.edge_index[i].size(1) for i in range(len(tr.edge_index[:42]))])
    model = Argus(tr.x.size(0), tr.eas[0].size(1), 128, 64, DEVICE, pos_samples=pos_samples)
    opt = SOAP(model.parameters(), lr=0.01, mode='adam', weight_decay=0.0)

    best = (0,0,0)
    best_cheating = (0,0)
    PATIENCE = 3 # Default for lanl
    #BS = 4 # Largest it can be on GPU without OOM
    BS = len(tr.edge_index)
    no_progress = 0
    for e in range(EPOCHS):
        fwd_time=bwd_time=loss_time=step_time = 0 
        for i in range(len(tr.edge_index) // BS): 
            st_i = i*BS
            en_i = (i+1)*BS

            model.train()
            opt.zero_grad()

            st = time.time()
            print("Fwd", end='', flush=True)
            zs = model.forward(tr.x, tr.edge_index[st_i:en_i], tr.eas[st_i:en_i], tr.idxs[st_i:en_i], tr.ptrs[st_i:en_i])
            fwd_time += time.time() - st
            print(f' ({((fwd_time) / 60):0.2f} mins)')

            st = time.time()
            print("Loss", end='', flush=True)
            loss = model.calc_loss_argus(zs, tr.edge_index[st_i:en_i])
            loss_time += time.time() - st
            print(f' ({((time.time() - st) / 60):0.2f} mins)')

            st = time.time()
            print("Bwd", end='', flush=True)
            loss.backward()
            bwd_time += time.time() - st

            st = time.time()
            opt.step()
            step_time += time.time() - st
            print(f' ({((time.time() - st) / 60):0.2f} mins)')

            print(f'[{e}] Loss: {loss.item():0.4f}')

        if SPEEDTEST: 
            with open('argus_speedtest.csv', 'a') as f:
                f.write(f'LANL,{fwd_time},{loss_time},{bwd_time},{step_time}\n')
            exit()
            

        with torch.no_grad():
            model.eval()
            zs = model.forward(tr.x, tr.edge_index, tr.eas, tr.idxs, tr.ptrs)
            pos,neg = model.validate(zs[:42], va.edge_index)
            labels = torch.zeros(pos.size(0)+neg.size(0))
            labels[pos.size(0):] = 1

            preds = torch.cat([pos,neg]).numpy()
            labels = labels.numpy()

            va_auc = fast_auc(labels, preds)
            va_ap = fast_ap(labels, preds)

            print(f'\tVal AUC: {va_auc:0.4f}, AP: {va_ap:0.4f}')

            preds = []
            for i in range(len(te.edge_index)):
                pred = (
                    zs[i][te.edge_index[i][0]] *
                    zs[i][te.edge_index[i][1]]
                ).sum(dim=1)
                preds.append(pred)

            preds = torch.sigmoid(torch.cat(preds)).numpy()
            y = torch.cat(te.label).clamp(0,1).numpy()

            auc = fast_auc(y, preds)
            ap = fast_ap(y, preds)
            print(f'\tTe AUC: {auc:0.4f}, AP: {ap:0.4f}', end='', flush=True)

            if va_auc+va_ap > best[0]:
                best = (va_auc+va_ap, auc, ap)
                print('*')
                no_progress = 0
            else:
                print()
                no_progress += 1

            # Validation doesn't want to work. I want to give Argus a fair
            # run, so let's just keep track of the best scores without
            # using the val set (this is data snooping, but even with
            # snooping, it doesn't seem like it will perform well)
            if auc > best_cheating[0]:
                best_cheating = (auc, ap)

            if no_progress > PATIENCE:
                break

            print(json.dumps({'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap, 'auc_snooped': best_cheating[0], 'ap_snooped': best_cheating[1]}, indent=1))

    print(f"Best: AUC {best[1]:0.4f}, AP {best[2]:0.4f}")
    return {'auc': best[1], 'ap': best[2], 'auc_last': auc, 'ap_last': ap, 'auc_snooped': best_cheating[0], 'ap_snooped': best_cheating[1]}

if __name__ == '__main__':
    if os.path.exists('tmp/argus_lanl_tr.pt'):
        tr = torch.load('tmp/argus_lanl_tr.pt', weights_only=False)
        va = torch.load('tmp/argus_lanl_va.pt', weights_only=False)
        te = torch.load('tmp/argus_lanl_te.pt', weights_only=False)

    else:
        tr = torch.load('../data/lanl14argus_tgraph_raw_tr.pt', weights_only=False)
        tr.ts //= 60*60
        va = torch.load('../data/lanl14argus_tgraph_raw_va.pt', weights_only=False)
        va.ts //= 60*60
        te = torch.load('../data/lanl14argus_tgraph_raw_te.pt', weights_only=False)
        te.ts //= 60*60

        ts = tr.ts.unique()
        print(ts.size())

        tr = to_snapshots(tr, ts, add_csr=True)
        va = to_snapshots(va, ts)
        te = to_snapshots(te, ts)

        torch.save(tr, 'tmp/argus_lanl_tr.pt')
        torch.save(va, 'tmp/argus_lanl_va.pt')
        torch.save(te, 'tmp/argus_lanl_te.pt')


    torch.set_num_threads(64)
    best = []
    for _ in range(10):
        best.append(train(tr,va,te))
        with open('argus_lanl_log.csv', 'a') as f:
            f.write(json.dumps(best[-1], indent=1))

    df = pd.DataFrame(best)
    df.loc['mean'] = df.mean()
    df.loc['sem'] = df.sem()

    df.to_csv('argus_results_lanl.csv')
