from torch import nn
from typing import List, Optional, Tuple, Union
import torch
from gosh.model.gpt_big_code_attention import GPTBigCodeAttention
from gosh.model.gpt_big_code_mlp import GPTBigCodeMLP

import logging

logger = logging.getLogger(__name__)

class GoshBigCodeBlock(nn.Module):
    @classmethod
    def create(cls, config, attention = None, layer_idx = None):
        if attention is None:
            attention = GPTBigCodeAttention(config, layer_idx=layer_idx)
        return GoshBigCodeBlock(config, attention, layer_idx)

    @classmethod
    def create_knn_attention(cls, config, layer_idx = None):
        return GoshBigCodeBlock(
            config, 
            attention = GPTBigCodeAttention(config, is_knn_attention=True, layer_idx=layer_idx),
            layer_idx = layer_idx
        )

    def __init__(self, config, attention, layer_idx=None):
        super().__init__()
        self.layer_idx = layer_idx
        hidden_size = config.hidden_size
        self.inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = attention
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            if config.multi_query:
                raise NotImplementedError("Cross-attention not implemented for MQA")
            self.crossattention = GPTBigCodeAttention(config, is_cross_attention=True, layer_idx=layer_idx)
            self.ln_cross_attn = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = GPTBigCodeMLP(self.inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[Tuple[torch.Tensor]],
        layer_past: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        knn_memory = None,
        add_knn_memory: bool = True,
    ) -> Union[
        Tuple[torch.Tensor], Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ]:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            knn_memory=knn_memory,
            add_knn_memory=add_knn_memory
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_outputs = self.crossattention(
                hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attn_output = cross_attn_outputs[0]
            # residual connection
            hidden_states = residual + attn_output
            outputs = outputs + cross_attn_outputs[2:]  # add cross attentions if we output attention weights

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        if torch.isnan(hidden_states).any():
            logger.warning("hidden state after block {self.layer_idx} has NaN")

        return outputs  # hidden_states, present, (attentions, cross_attentions)
