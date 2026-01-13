import torch
from torch import nn

from models.gpt import MaskedAttentionEmb

class GraphGPT(nn.Module):
    def __init__(self, gpt_files, device='cpu'):
        super().__init__()

        args,kwargs,sd = torch.load(gpt_files, weights_only=False, map_location='cpu')
        self.fm = MaskedAttentionEmb(args, **kwargs)
        self.fm.load_state_dict(sd)
        self.fm = self.fm.to(device)

        self.out = nn.Linear(kwargs['hidden_size']*4, 1, device=device)
        self.criterion = nn.BCEWithLogitsLoss()

        self.device = device

    def predict(self, seqs, targets):
        '''
            Targets should be TARGET_TOKEN x B matrix

        '''
        seqs = seqs.to(self.device)

        embs = self.fm.embedding(seqs, offset=False)
        embs = embs[targets, torch.arange(embs.size(1))] # Target Tokens x B x d
        embs = embs.transpose(0,1) # B x Target tokens x d
        embs = embs.reshape(embs.size(0), -1) # B x Target Tokens * d
        # Not sure why .view isn't working up there?

        pred = self.out(embs)
        return pred

    def forward(self, seqs, targets, labels):
        preds = self.predict(seqs, targets)
        labels = labels.to(self.device)
        loss = self.criterion(preds, labels)

        return loss

    def simple_predict(self, seqs):
        seqs = seqs.to(self.device)
        embs = self.fm.embedding(seqs, offset=False) # 4 x B x d
        embs = embs.transpose(0,1)               # B x 4 x d
        embs = embs.reshape(embs.size(0), -1)    # B x 4*d

        pred = self.out(embs)
        return pred

    def simple_forward(self, seqs, labels):
        preds = self.simple_predict(seqs)
        labels = labels.to(self.device)
        loss = self.criterion(preds, labels)
        return loss