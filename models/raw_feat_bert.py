'''
CyberGFM with raw (continuous) edge features instead of discretized feature tokens.

Each hop contributes ONE feature slot whose embedding is the raw feature vector projected to
hidden_size, rather than F separate tokens:

    tokenized:  [n0, f0_1 ... f0_F, n1, f1_1 ... f1_F, n2]
    raw:        [n0, FEAT,          n1, FEAT,          n2]        + a (B, S, F) float tensor

Sequences are (1+F)/2 times shorter, nothing is lost to discretization, and the vocabulary is
nodes only. Feature slots are marked in `walks` with the sentinel FEAT (-3); the sampler in
raw_sampler.py produces both tensors.

Backward compatible: calling modified_fwd() without `feats` falls through to RWBert, so a
RawFeatRWBert behaves exactly like the tokenized model when no features are supplied.
'''
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import MaskedLMOutput

from models.gnn_bert import GNNEmbedding, RWBert

FEAT = -3  # sentinel marking a feature slot; PAD=-2 and MASK=-1 keep their meanings


class FeatureNormalizer(nn.Module):
    '''log1p on heavy-tailed columns, then z-score. Fit once on training edges; saved in the state dict.'''

    def __init__(self, feat_dim):
        super().__init__()
        self.register_buffer('log_cols', torch.zeros(feat_dim, dtype=torch.bool))
        self.register_buffer('mean', torch.zeros(feat_dim))
        self.register_buffer('std', torch.ones(feat_dim))
        self.register_buffer('fitted', torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit(self, x, log_threshold=10.0, max_rows=5_000_000):
        x = x.float()
        if x.size(0) > max_rows:
            x = x[torch.randperm(x.size(0), device=x.device)[:max_rows]]
        x = x.to(self.mean.device)
        self.log_cols.copy_(x.max(dim=0).values > log_threshold)
        x = self._pre(x)
        self.mean.copy_(x.mean(dim=0))
        self.std.copy_(x.std(dim=0).clamp_min(1e-6))
        self.fitted.fill_(True)
        return self

    def _pre(self, x):
        return torch.where(self.log_cols, torch.log1p(x.clamp_min(0)), x)

    def forward(self, x):
        return (self._pre(x.float()) - self.mean) / self.std


class RawFeatRWBert(RWBert):
    '''
    RWBert whose feature slots are embedded by projecting raw feature vectors.

    config needs `feat_dim` (raw feature width) in addition to the usual fields; vocab_size should
    cover nodes only (num_nodes + GNNEmbedding.OFFSET), since features no longer take token ids.
    '''

    FEAT = FEAT

    def __init__(self, config):
        super().__init__(config)
        hidden = config.hidden_size
        self.normalizer = FeatureNormalizer(config.feat_dim)
        self.feat_proj = nn.Sequential(
            nn.Linear(config.feat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden, eps=config.layer_norm_eps),
        )

    def fit_normalizer(self, raw_edge_attr):
        self.normalizer.fit(raw_edge_attr)
        return self

    def embed(self, walks, feats):
        '''Token embeddings with feature slots replaced by projected raw features.'''
        feat_mask = walks == FEAT
        ids = walks.clone()
        ids[feat_mask] = GNNEmbedding.PAD  # any valid id; these embeddings are overwritten below
        ids[ids < 0] += GNNEmbedding.OFFSET + self.config.num_nodes

        embs = self.bert.embeddings.word_embeddings(ids)
        if feat_mask.any():
            projected = self.feat_proj(self.normalizer(feats[feat_mask]))
            embs = embs.masked_scatter(feat_mask.unsqueeze(-1), projected.to(embs.dtype))
        return embs

    def modified_fwd(self, walks, masks, targets, attn_mask, feats=None, return_loss=True, skip_cls=False):
        if feats is None:  # no raw features supplied -> tokenized behaviour
            return super().modified_fwd(walks, masks, targets, attn_mask,
                                        return_loss=return_loss, skip_cls=skip_cls)

        walks = walks.to(self.device)
        feats = feats.to(self.device)
        embs = self.embed(walks, feats)

        tgt = torch.full(masks.size(), -100, device=masks.device)
        if isinstance(targets, torch.Tensor):
            tgt[masks] = targets
        tgt = tgt.to(self.device)

        pos_ids = torch.arange(walks.size(1), device=self.device).repeat(walks.size(0), 1)
        if isinstance(attn_mask, torch.Tensor):
            attn_mask = attn_mask.to(self.device)

        # GraphBertForMaskedLM.forward() dereferences input_ids before checking for None,
        # so run the encoder and MLM head directly instead of going through it.
        sequence_output = self.bert(
            inputs_embeds=embs, attention_mask=attn_mask, position_ids=pos_ids, return_dict=True
        )[0]
        if skip_cls:
            return sequence_output

        logits = self.cls(sequence_output)
        loss = None
        if isinstance(targets, torch.Tensor):
            loss = CrossEntropyLoss()(logits.view(-1, self.config.vocab_size), tgt.view(-1))
        out = MaskedLMOutput(loss=loss, logits=logits)
        return out.loss if return_loss else out


class RawFeatRWBertFT(nn.Module):
    '''Classification head on top of RawFeatRWBert; mirrors RWBertFT but passes `feats` through.'''

    def __init__(self, config, sd, device='cpu', out_dim=1, from_random=False):
        super().__init__()
        self.fm = RawFeatRWBert(config)
        if not from_random:
            self.fm.load_state_dict(sd)
        self.fm = self.fm.to(device)

        self.cls = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size, device=device),
            nn.ReLU(),
            nn.Linear(config.hidden_size, out_dim, device=device),
        )
        self.config = config
        self.out_dim = out_dim
        self.device = device

    def predict(self, walks, attn_mask, tgt_mask, feats=None):
        out = self.fm.modified_fwd(walks, tgt_mask, None, attn_mask, feats=feats,
                                   return_loss=False, skip_cls=True)
        return self.cls(out[tgt_mask])

    def forward(self, rw, attn, tgt_mask, target, feats=None):
        pred = self.predict(rw, attn, tgt_mask, feats=feats)
        loss_fn = nn.BCEWithLogitsLoss() if self.out_dim == 1 else nn.CrossEntropyLoss()
        return loss_fn(pred, target)


def raw_feat_config(num_nodes, feat_dim, hidden, layers, max_positions, base_config=None):
    '''BertConfig for RawFeatRWBert: node-only vocabulary plus feat_dim.'''
    from transformers import BertConfig
    if base_config is not None:
        cfg = base_config
        cfg.feat_dim = feat_dim
        return cfg
    return BertConfig(
        int(num_nodes) + GNNEmbedding.OFFSET,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 64,
        intermediate_size=hidden * 4,
        num_nodes=int(num_nodes),
        feat_dim=int(feat_dim),
        max_position_embeddings=max_positions,
    )