'''
Argus (Xu, Shu & Li, IEEE S&P 2024): the model and its loss.

Mirrors github.com/C0ldstudy/Argus @ 0bd70e6 -- models/argus.py (Argus_LANL, Argus_OPTC,
DetectorEncoder), models/recurrent.py (GRU) and the vendored libauc APLoss. Anything that knowingly
differs from their code is a constructor argument, documented where it is defined.

SCORE CONVENTION. decode() is a reconstruction probability: HIGH MEANS NORMAL. Observed edges are
the positive class in the loss, and detection negates (their score_stats: `scores = 1 - scores`).
Flipping the labels instead is not equivalent -- AP is asymmetric in the two classes.

Everything that is not the model (snapshots, training loop, evaluation, arguments, result files)
is in argus_common.py.
'''
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, NNConv


class APLoss(nn.Module):
    '''
    libauc's APLoss (Qi et al., NeurIPS 2021) with surrogate_loss='squared', as Argus builds it.

    The moving averages u_all / u_pos are indexed by a positive's position within its snapshot
    (Argus passes `t_index = arange(n_pos)` for every snapshot), so slot k is shared by the k-th
    positive of every snapshot and persists across epochs. `pos_len` must be at least the largest
    number of positives one call sees.
    '''

    def __init__(self, pos_len, margin=0.8, gamma=0.01, device='cpu'):
        super().__init__()
        self.u_all = torch.zeros(pos_len, 1, device=device)
        self.u_pos = torch.zeros(pos_len, 1, device=device)
        self.margin, self.gamma = margin, gamma

    def forward(self, score, n_pos):
        '''score = cat(positive scores, negative scores); the first n_pos are the positives.'''
        pos_mask = torch.zeros(1, score.numel(), device=score.device)
        pos_mask[0, :n_pos] = 1
        sur = (self.margin - (score[:n_pos, None] - score[None, :])) ** 2   # (n_pos, n_all)
        k, g = slice(0, n_pos), self.gamma
        self.u_all[k] = (1 - g) * self.u_all[k] + g * sur.mean(1, keepdim=True).detach()
        self.u_pos[k] = (1 - g) * self.u_pos[k] + g * (sur * pos_mask).mean(1, keepdim=True).detach()
        p = (self.u_pos[k] - self.u_all[k] * pos_mask) / self.u_all[k] ** 2
        return (p * sur).mean()


class GRU(nn.Module):
    '''Their recurrent.GRU: dropout, nn.GRU over time (input is (T, N, h)), linear to z_dim.'''

    def __init__(self, x_dim, h_dim, z_dim, layers=1):
        super().__init__()
        self.rnn = nn.GRU(x_dim, h_dim, num_layers=layers)
        self.drop = nn.Dropout(0.25)
        self.lin = nn.Linear(h_dim, z_dim)

    def forward(self, xs, h0=None):
        xs, h = self.rnn(self.drop(xs), h0)
        return self.lin(xs), h


def undirected_view(ei, ea=None, ew=None):
    '''
    The snapshot with each missing reverse edge added, for MESSAGE PASSING only.

    Only reverses that are absent are added, so a pair seen both ways keeps its two real rows. An
    added row copies its edge's features and weight. The loss and every metric still use the
    directed edges the caller holds, and decode() is symmetric, so no scored edge gains anything.
    '''
    if ei.size(1) == 0:
        return ei, ea, ew
    n = int(ei.max()) + 1
    add = ~torch.isin(ei[1] * n + ei[0], ei[0] * n + ei[1]) & (ei[0] != ei[1])
    ei = torch.cat([ei, ei[:, add].flip(0)], dim=1)
    ea = None if ea is None else torch.cat([ea, ea[add]])
    ew = None if ew is None else torch.cat([ew, ew[add]])
    return ei, ea, ew


class Argus(nn.Module):
    '''
    Encoder per snapshot, GRU across snapshots, sigmoid dot-product decoder, AP loss.

    arch='lanl'   their Argus_LANL: c1, c2 GCNConv (no activation between them), relu+dropout,
                  c3 GCNConv, relu+dropout, c4 NNConv over edge features (aggr='mean'), tanh.
    arch='optc'   their Argus_OPTC: the same, but c4 is a fourth GCNConv, edge features are never
                  read, and DropEdge(0.5) is applied while training.
    The GCN layers take edge weights (their std_edge_w of the per-snapshot count) when given.
    Widths follow their main.py: every encoder layer h_dim, the GRU projects h_dim -> z_dim.

    Deviations from their code, all off by default:
      undirected  propagate over undirected_view(snapshot); see that function.
      beta        weight on the paper's L_DEC term, read here as BCE on the sampled edges. Their
                  code has no such term; the paper says beta=0.01 (LANL) / 0.5 (OpTC).
      max_pos     cap positives per snapshot; APLoss memory is quadratic in that count.
    Their code also never calls the paper's neighbour aggregation (inner_forward_sm), so neither
    does this.
    '''

    def __init__(self, num_nodes, edge_dim, h_dim=32, z_dim=16, gru_layers=1, arch='lanl',
                 margin=0.8, gamma=0.01, beta=0.0, undirected=False, max_pos=0, pos_len=1,
                 device='cpu'):
        super().__init__()
        assert arch in ('lanl', 'optc'), arch
        self.num_nodes, self.arch, self.device = num_nodes, arch, device
        self.beta, self.undirected, self.max_pos = beta, undirected, max_pos

        # identity node features, as their loaders build (torch.eye); made once, kept on device
        self.register_buffer('x', torch.eye(num_nodes), persistent=False)
        self.c1 = GCNConv(num_nodes, h_dim)
        self.c2 = GCNConv(h_dim, h_dim)
        self.c3 = GCNConv(h_dim, h_dim)
        if arch == 'optc':
            self.c4 = GCNConv(h_dim, h_dim)
        else:
            self.c4 = NNConv(h_dim, h_dim, nn.Sequential(
                nn.Linear(edge_dim, 8), nn.ReLU(), nn.Linear(8, h_dim * h_dim)), aggr='mean')
        self.drop = nn.Dropout(0.1)
        self.rnn = GRU(h_dim, h_dim, z_dim, layers=gru_layers)
        self.to(device)
        self.ap_loss = APLoss(max(pos_len, 1), margin, gamma, device)

    # -- embedding -------------------------------------------------------------------------------
    def encode(self, ei, ea, ew):
        ei = ei.to(self.device)
        ea = ea.to(self.device).float()
        ew = None if ew is None else ew.to(self.device).float()
        if self.undirected:
            ei, ea, ew = undirected_view(ei, ea, ew)
        if self.arch == 'optc' and self.training:          # their DropEdge(0.5)
            keep = torch.rand(ei.size(1), device=self.device) > 0.5
            ei, ew = ei[:, keep], (None if ew is None else ew[keep])

        z = self.c2(self.c1(self.x, ei, edge_weight=ew), ei, edge_weight=ew)
        z = self.drop(F.relu(z))
        z = self.drop(F.relu(self.c3(z, ei, edge_weight=ew)))
        z = self.c4(z, ei, edge_weight=ew) if self.arch == 'optc' else self.c4(z, ei, edge_attr=ea)
        return torch.tanh(z)

    def forward(self, eis, eas, ews=None, h0=None):
        '''Per-snapshot lists in, (T, N, z_dim) embeddings and the GRU state out.'''
        ews = ews if ews is not None else [None] * len(eis)
        zs = torch.stack([self.encode(ei, ea, ew) for ei, ea, ew in zip(eis, eas, ews)])
        return self.rnn(zs, h0)

    # -- scoring ---------------------------------------------------------------------------------
    @staticmethod
    def decode(ei, z):
        '''Reconstruction probability; HIGH MEANS NORMAL.'''
        return torch.sigmoid((z[ei[0]] * z[ei[1]]).sum(1))

    def negatives(self, pos):
        '''Uniform random pairs, redrawn where they hit a real edge (their fast_negative_sampling).'''
        n = self.num_nodes
        neg = torch.randint(0, n, pos.shape, device=pos.device)
        real = pos[0] + pos[1] * n
        for _ in range(5):
            bad = torch.isin(neg[0] + neg[1] * n, real)
            if not bad.any():
                break
            neg[:, bad] = torch.randint(0, n, (2, int(bad.sum())), device=pos.device)
        return neg

    def loss(self, zs, eis):
        '''Their calc_loss_argus: AP loss per snapshot, averaged over all snapshots (incl. empty).'''
        tot = torch.zeros((), device=self.device)
        for z, ei in zip(zs, eis):
            if ei.size(1) == 0:
                continue
            ei = ei.to(self.device)
            if self.max_pos and ei.size(1) > self.max_pos:
                ei = ei[:, torch.randperm(ei.size(1), device=self.device)[:self.max_pos]]
            pos, neg = self.decode(ei, z), self.decode(self.negatives(ei), z)
            score = torch.cat([pos, neg])
            tot = tot + self.ap_loss(score, pos.numel())
            if self.beta:
                y = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
                tot = tot + self.beta * F.binary_cross_entropy(score.clamp(1e-7, 1 - 1e-7), y)
        return tot / len(eis)