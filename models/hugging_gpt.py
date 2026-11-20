'''
Causal (GPT) counterparts of RWBert / RWBertFT for the bidirectional-vs-causal ablation.

Fixes relative to the previous version:
  * attention_mask is built from PAD *before* special tokens are remapped, with
    HF polarity (1 = attend). Previously it was all zeros and inverted.
  * padding positions are excluded from the next-token loss (label -100).
  * inputs are not modified in place.
  * GPT_Cls can actually freeze the backbone (freeze=True); the old
    `model.fm.requires_grad = False` only set an attribute and froze nothing.

Parameter names under `self.gpt` are unchanged, so state dicts keep the same keys.
'''
import torch
from torch import nn
from transformers import OpenAIGPTConfig, OpenAIGPTLMHeadModel

from models.gnn_bert import GNNEmbedding

PAD, MASK, OFFSET = GNNEmbedding.PAD, GNNEmbedding.MASK, GNNEmbedding.OFFSET


def gpt_config(num_tokens, hidden, layers, max_positions):
    num_tokens = int(num_tokens)
    return OpenAIGPTConfig(
        vocab_size=num_tokens + OFFSET,
        n_embd=hidden,
        n_layer=layers,
        n_head=hidden // 64,
        n_positions=max_positions,
        num_nodes=num_tokens,
    )


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gpt = OpenAIGPTLMHeadModel(config)

    def _prep(self, walks):
        attn = (walks != PAD).long()
        ids = walks.clone()
        ids[ids < 0] += OFFSET + self.config.num_nodes  # PAD -> num_nodes, MASK -> num_nodes + 1
        pos = torch.arange(ids.size(1), device=ids.device).expand(ids.size(0), -1)
        return ids, attn, pos

    def hidden(self, walks):
        ids, attn, pos = self._prep(walks)
        return self.gpt.transformer(ids, attention_mask=attn, position_ids=pos)[0]

    def logits(self, walks):
        ids, attn, pos = self._prep(walks)
        return self.gpt(ids, attention_mask=attn, position_ids=pos).logits

    def forward(self, walks, return_loss=True):
        '''Next-token prediction over the whole walk, ignoring padding.'''
        ids, attn, pos = self._prep(walks)
        labels = ids.masked_fill(attn == 0, -100)
        out = self.gpt(ids, attention_mask=attn, position_ids=pos, labels=labels)
        return out.loss if return_loss else out


class GPT_Cls(nn.Module):
    '''Binary classifier on the hidden state of a [CLS] (MASK) token appended to the walk.'''

    def __init__(self, config, sd, device='cpu', out_dim=1, from_random=False, freeze=False):
        super().__init__()
        self.fm = GPT(config)
        if not from_random:
            self.fm.load_state_dict(sd)
        if freeze:
            self.fm.requires_grad_(False)
        self.fm = self.fm.to(device)

        self.cls = nn.Sequential(
            nn.Linear(config.n_embd, config.n_embd),
            nn.ReLU(),
            nn.Linear(config.n_embd, out_dim),
        ).to(device)
        self.config = config
        self.device = device

    def predict(self, walks):
        cls = torch.full((walks.size(0), 1), MASK, device=walks.device, dtype=walks.dtype)
        walks = torch.cat([walks, cls], dim=1)
        # Left-align (PAD to the end) so tokens sit on pretrained positions; CLS is the last real token
        walks = walks.gather(1, torch.argsort((walks == PAD).int(), dim=1, stable=True))
        h = self.fm.hidden(walks)
        last = (walks != PAD).sum(dim=1) - 1
        return self.cls(h[torch.arange(walks.size(0), device=walks.device), last])

    def forward(self, walks, target):
        return nn.functional.binary_cross_entropy_with_logits(self.predict(walks), target)