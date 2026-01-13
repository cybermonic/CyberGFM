import torch
from torch import nn

from models.gpt import PositionalEncoding

class BERT(nn.Module):
    def __init__(self, num_tokens, emb_dim=128, device='cpu', hidden_size=768, layers=12):
        super().__init__()

        self.device=device
        self.args = (num_tokens)
        self.kwargs = dict(
            hidden_size=hidden_size,
            layers=layers,
            emb_dim=emb_dim
        )

        self.embed = nn.Embedding(num_tokens, emb_dim, device=self.device)
        self.proj = nn.Sequential(
            nn.Linear(emb_dim, hidden_size, device=self.device),
            nn.GELU()
        )
        self.pe = PositionalEncoding(hidden_size, device=self.device)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                hidden_size, hidden_size//64,
                dim_feedforward=hidden_size*4,
                activation='gelu',
                device=device
            ), num_layers=layers
        )
        self.out = nn.Linear(hidden_size, num_tokens, device=self.device)
        self.criterion = nn.CrossEntropyLoss()

    def embedding(self, seq):
        tokens = self.proj(self.embed(seq))
        pe = self.pe(tokens)
        tokens = tokens + pe
        preds = self.transformer.forward(tokens)
        return preds

    def forward(self, seq, masks, targets):
        '''
        Expects S x B list of Eulerian walks
        '''
        seq = seq.to(self.device)
        targets = targets.to(self.device)

        preds = self.embedding(seq)[masks]
        preds = self.out(preds)
        loss = self.criterion.forward(
            preds, targets
        )

        return loss