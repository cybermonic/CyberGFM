import torch
from torch import nn

from models.bert import BERT

class FT_BERT(nn.Module):
    def __init__(self, gpt_files, device='cpu'):
        super().__init__()

        args,kwargs,sd = torch.load(gpt_files, weights_only=False, map_location='cpu')
        self.fm = BERT(args, **kwargs)
        self.fm.load_state_dict(sd)
        self.fm = self.fm.to(device)

        self.out = nn.Linear(kwargs['hidden_size']*4, 1, device=device)
        self.criterion = nn.BCEWithLogitsLoss()

        self.device = device

    def predict(self, seqs):
        seqs = seqs.to(self.device)
        embs = self.fm.embedding(seqs) # 4 x B x d
        embs = embs.transpose(0,1)               # B x 4 x d
        embs = embs.reshape(embs.size(0), -1)    # B x 4*d

        pred = self.out(embs)
        return pred

    def forward(self, seqs, labels):
        preds = self.predict(seqs)
        labels = labels.to(self.device)
        loss = self.criterion(preds, labels)
        return loss