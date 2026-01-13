import torch 
from torch import nn 
from transformers import OpenAIGPTModel, OpenAIGPTLMHeadModel

from models.gnn_bert import GNNEmbedding

class GPT(nn.Module): 
    def __init__(self, config): 
        super().__init__()

        self.config = config 
        self.gpt = OpenAIGPTLMHeadModel(config)
        self.device = 'cpu'

    def to(self, device): 
        self.gpt = self.gpt.to(device)
        self.device = device
        return self 

    def modified_fwd(self, walks, *args, return_loss=True):
        # For bwd compatability  
        return self.forward(walks, return_loss=return_loss)

    def forward(self, walks, *args, return_loss=True, skip_cls=False):
        walks[walks < 0] += GNNEmbedding.OFFSET + self.config.num_nodes
        mask = walks == GNNEmbedding.PAD 

        pos_ids = torch.arange(
            walks.size(1),
            device=self.device
        ).repeat(walks.size(0), 1)

        if skip_cls: 
            return self.gpt.transformer(
                walks,
                attention_mask=mask,
                position_ids=pos_ids,
            )[0]

        preds = self.gpt.forward(
            walks, mask, position_ids=pos_ids, labels=walks
        )

        if return_loss: 
            return preds.loss 
        return preds
    
class GPT_Cls(nn.Module): 
    def __init__(self, config, sd, device='cpu', out_dim=1, from_random=False):
        super().__init__()
        self.fm = GPT(config)
        
        if not from_random: 
            self.fm.load_state_dict(sd)
            
        self.fm = self.fm.to(device)

        self.cls = nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size, device=device),
            torch.nn.ReLU(), 
            torch.nn.Linear(config.hidden_size, out_dim, device=device)
        )

        self.config = config
        self.device = device

    def predict(self, walks): 
        cls = torch.full((walks.size(0),1), GNNEmbedding.MASK, device=self.device)
        walks = torch.cat([walks, cls], dim=1)

        out = self.fm.forward(walks, skip_cls=True)
        emb = out[:, -1, :]
        return self.cls(emb)

    def forward(self, rw, target): 
        pred = self.predict(rw)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(pred,target)
        return loss 
