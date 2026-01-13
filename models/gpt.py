import math

import torch
import torch.functional as F
from torch import nn
from torch import Tensor

class PositionalEncoding(nn.Module):
    '''
    Stolen from stack overflow
    https://stackoverflow.com/questions/77444485/using-positional-encoding-in-pytorch
    '''
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000, device='cpu'):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, device=device) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, device=device)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        return self.pe[:x.size(0)]

class MaskedAttentionEmb(nn.Module):
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

        self.transformer = nn.Transformer(
            hidden_size, hidden_size//64,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=hidden_size*4,
            activation='gelu',
            device=device
        )
        self.out = nn.Linear(hidden_size, num_tokens, device=self.device)
        self.criterion = nn.CrossEntropyLoss()

    def embedding(self, seq, offset=True):
        # Generate NTPs
        tokens = self.proj(self.embed(seq))
        pe = self.pe(tokens)
        tokens = tokens + pe

        if offset:
            tokens = tokens[:-1]
        else:
            tokens = tokens

        mask = nn.Transformer.generate_square_subsequent_mask(tokens.size(0)-1)
        preds = self.transformer.forward(
            src=tokens,
            tgt=tokens,
            tgt_is_causal=True,
            tgt_mask=mask
        )

        return preds

    def forward(self, seq, targets):
        '''
        Expects S x B list of Eulerian walks
        '''
        seq = seq.to(self.device)
        targets = targets.to(self.device)

        preds = self.embedding(seq)
        preds = self.out(preds)
        loss = self.criterion.forward(
            preds[targets[1:]], seq[1:][targets[1:]]
        )

        return loss