# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from typing import List, Optional, Tuple, Union
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertModel,
    BertOnlyMLMHead,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


# copied and modified from `transformers/models/bert/modeling_bert.py::BertForMaskedLM`
class GraphBertForMaskedLM(BertPreTrainedModel):
    _tied_weights_keys = ["predictions.decoder.bias", "cls.predictions.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        self.cls.predictions.decoder = new_embeddings

    def modified_fwd(self, walks, masks, targets, attn_mask, return_loss=True, skip_cls=True):
        input_ids = walks.to(self.device)
        tgt = torch.full(masks.size(), -100, device=masks.device)
        
        if isinstance(targets, torch.Tensor):
            tgt[masks] = targets
        
        tgt = tgt.to(self.device)
        pos_ids = torch.arange(
            tgt.size(1),
            device=self.device).repeat(tgt.size(0), 1
        )

        if isinstance(attn_mask, torch.Tensor):
            attn_mask=attn_mask.to(self.device)

        out = self.forward(
            input_ids, labels=tgt, position_ids=pos_ids,
            return_dict=True, attention_mask=attn_mask, skip_cls=skip_cls
        )

        if return_loss:
            return out.loss
        else:
            return out

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        skip_cls: bool = False
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if len(input_ids.shape) == 3:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            # [bz, seq, feat, dim]
            inputs_embeds = torch.sum(inputs_embeds, dim=-2)
            # [bz, seq, dim]
            assert inputs_embeds.shape[:2] == input_ids.shape[:2]
            input_ids = None

        outputs = self.bert.forward(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        if skip_cls: 
            return sequence_output

        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return (
                ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            )

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GraphBertFT(torch.nn.Module):
    def __init__(self, config, weights, device='cpu'):
        super().__init__()
        self.fm = GraphBertForMaskedLM(config)
        sd = torch.load(weights, weights_only=True, map_location='cpu')
        self.fm.load_state_dict(sd)
        self.fm = self.fm.to(device)
        del self.fm.cls

        self.cls = torch.nn.Linear(config.hidden_size, 1, device=device)

        self.config = config
        self.device = device

    def predict(self, walks, masks) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        return_dict = self.config.use_return_dict

        input_ids = walks.to(self.device)
        position_ids = torch.arange(
            input_ids.size(1),
            device=self.device).repeat(input_ids.size(0), 1
        )

        attn_mask = (walks != self.config.padding_idx).to(self.device)

        if len(input_ids.shape) == 3:
            inputs_embeds = self.bert.embeddings.word_embeddings(input_ids)
            # [bz, seq, feat, dim]
            inputs_embeds = torch.sum(inputs_embeds, dim=-2)
            # [bz, seq, dim]
            assert inputs_embeds.shape[:2] == input_ids.shape[:2]
            input_ids = None

        outputs = self.fm.bert(
            input_ids,
            position_ids=position_ids,
            return_dict=return_dict,
            attention_mask=attn_mask
        )[0]
        predictions = self.cls(outputs[masks])
        return predictions

    def forward(self, walks, masks, targets) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        """
        predictions = self.predict(walks, masks)
        loss_fn = torch.nn.BCEWithLogitsLoss()

        loss = loss_fn(predictions, targets.to(self.device))
        return loss