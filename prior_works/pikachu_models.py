from typing import Union, List, Tuple, Optional

from temporal_walks import temporal_rw
import torch 
from torch import nn, Tensor 


class CTDNE(nn.Module): 
    '''
    Mostly copied from PyG Node2Vec
    Kind of pointless, given UNSW has discrete timesteps, so 
    '''
    def __init__(
        self,
        embedding_dim: int,
        walk_length: int,
        context_size: int,
        num_nodes: int, 
        walks_per_node: int = 1, 
        device = 'cpu'
    ): 
        super().__init__()

        self.num_nodes = num_nodes 

        self.EPS = 1e-15
        assert walk_length >= context_size

        self.embedding_dim = embedding_dim
        self.walk_length = walk_length - 1
        self.context_size = context_size
        self.walks_per_node = walks_per_node
        self.num_negative_samples = 1

        self.embedding = nn.Embedding(self.num_nodes, embedding_dim, device=device)

    def forward(self, batch: Optional[Tensor] = None) -> Tensor:
        """Returns the embeddings for the nodes in :obj:`batch`."""
        emb = self.embedding.weight
        return emb if batch is None else emb[batch]

    @torch.jit.export
    def pos_sample(self, batch: Tensor, g) -> Tensor:
        batch = batch.repeat(self.walks_per_node)
        
        if 'ts' in g.keys(): 
            rw = temporal_rw(g.rowptr, g.col, g.ts, batch, self.walk_length)
        else: 
            rw,_ = torch.ops.torch_cluster.random_walk(g.rowptr, g.col, batch, self.walk_length, 1,1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def neg_sample(self, batch: Tensor) -> Tensor:
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples)

        rw = torch.randint(self.num_nodes, (batch.size(0), self.walk_length),
                           dtype=batch.dtype, device=batch.device)
        rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + self.context_size])
        return torch.cat(walks, dim=0)

    @torch.jit.export
    def sample(self, batch: Union[List[int], Tensor], g) -> Tuple[Tensor, Tensor]:
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        return self.pos_sample(batch, g), self.neg_sample(batch)

    @torch.jit.export
    def loss(self, pos_rw: Tensor, neg_rw: Tensor) -> Tensor:
        r"""Computes the loss given positive and negative random walks."""
        # Positive loss.
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(pos_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(pos_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(out) + self.EPS).mean()

        # Negative loss.
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()

        h_start = self.embedding(start).view(neg_rw.size(0), 1,
                                             self.embedding_dim)
        h_rest = self.embedding(rest.view(-1)).view(neg_rw.size(0), -1,
                                                    self.embedding_dim)

        out = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(1 - torch.sigmoid(out) + self.EPS).mean()

        return pos_loss + neg_loss


class GRUEncoder(nn.Module): 
    def __init__(self, in_dim, hidden_dim, out_dim, device):
        super().__init__()
        self.gru1 = nn.GRU(in_dim, hidden_dim, device=device)
        self.drop = nn.Dropout(0.3)
        self.gru2 = nn.GRU(hidden_dim, out_dim, device=device) 

    def forward(self, zs): 
        '''
        input  (ts x N x d)   matrix 
        output (ts x N x out) matrix
        '''
        zs = self.gru1(zs)[0]
        zs = self.drop(zs) 
        return self.gru2(zs)[0]
    

class Autoencoder(nn.Module): 
    def __init__(self, emb_dim, hidden_dim, latent_dim, device='cpu'):
        super().__init__()

        self.enc = GRUEncoder(emb_dim, hidden_dim, latent_dim, device)
        self.dec = GRUEncoder(latent_dim, hidden_dim, emb_dim, device)
        self.loss = nn.MSELoss()

    def forward(self, zs): 
        enc = self.enc(zs)
        zs_hat = self.dec(enc)
        return self.loss(zs, zs_hat)
    
        
class AnomalyDetector(nn.Module): 
    def __init__(self, latent_dim, num_nodes, s=10, device='cpu'):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(latent_dim, num_nodes, bias=False, device=device),
            nn.Softmax(dim=1)
        )

        self.s = s
        self.loss = nn.CrossEntropyLoss()
        self.device = device 

    def predict(self, z, te_ei, ptr, col): 
        '''
        Make sure idx and ptr are from the training set
        '''
        n_u = []
        for i in range(z.size(0)):             
            # Sample u's neighbors (original code uses same sample for every edge)
            neighbors = col[ptr[i]:ptr[i+1]].to(self.device)
            
            # If u has no neighbors, just make sample == z[u]
            if neighbors.size(0) == 0:
                n_u.append(torch.full((self.s,), i, device=self.device))
            else:      
                sample_idx = (torch.rand(self.s, device=self.device) * neighbors.size(0)).long()
                n_u.append(neighbors[sample_idx])

        n_u = torch.stack(n_u)
        u_aggr = (z[n_u].sum(dim=1) + z) / (self.s + 1)
        h_u = self.predictor(u_aggr)

        return h_u[te_ei[0]], h_u[te_ei[1]]
    
    def forward(self, z, edges, idx, col): 
        h_u, _ = self.predict(z, edges, idx, col)
        loss = self.loss(h_u, edges[1].to(self.device))
        return loss 
    
    def get_score(self, z, edges, idx, col): 
        h_u,h_v = self.predict(z, edges, idx, col)
        
        src_score = h_u[torch.arange(h_u.size(0)), edges[1]]
        dst_score = h_v[torch.arange(h_u.size(0)), edges[0]]

        return ( (1-src_score) + (1-dst_score) ) / 2 