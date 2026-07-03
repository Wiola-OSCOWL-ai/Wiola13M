# coding=utf-8
# Copyright 2025 The Wiola Project. All rights reserved.
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
"""PyTorch Wiola model — Gated Spiral Attention small language model.

Novel components implemented here:

* **Spiral Rotary Positional Encoding** — RoPE with a ``sqrt``-growing,
  per-dimension-pair frequency perturbation that fans phase trajectories
  outward for better long-range discrimination.
* **Gated Spiral Attention (GSA)** — masked multi-head self-attention whose
  pre-softmax scores are modulated by a per-head, content-adaptive scalar gate
  derived *causally* from a cumulative mean of the query projections.
* **Butterfly MLP** — a multiplicative feed-forward block with an intra-block
  bypass connection that matches a GeLU 4x FFN in parameter count while
  providing SwiGLU-class gating.

The implementation is fully compatible with the Hugging Face ``Auto*`` API and
supports KV-cached autoregressive generation (including the running state
required by the causal gate).
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from .configuration_wiola import WiolaConfig

logger = logging.get_logger(__name__)


# =============================================================================
# Normalisation
# =============================================================================
class WiolaRMSNorm(nn.Module):
    """Root-mean-square layer normalisation (Zhang & Sennrich, 2019)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


# =============================================================================
# Spiral Rotary Positional Encoding
# =============================================================================
class WiolaSpiralRotaryEmbedding(nn.Module):
    r"""Spiral Rotary Positional Encoding.

    For dimension pair ``i`` the base RoPE frequency is perturbed by a
    ``sqrt``-growing spiral factor::

        theta_tilde_i = (1 / base ** (2 i / d_h)) * (1 + alpha * sqrt(i + 1) / sqrt(d_h / 2))

    ``alpha = 0`` recovers standard RoPE exactly.
    """

    def __init__(self, dim: int, base: float = 10000.0, alpha: float = 0.05):
        super().__init__()
        self.dim = dim
        self.base = base
        self.alpha = alpha

        half = dim // 2
        # Standard geometric RoPE frequencies: 1 / base ** (2 i / d_h)
        base_inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )
        # Spiral perturbation phi_i = 1 + alpha * sqrt(i + 1) / sqrt(d_h / 2)
        idx = torch.arange(0, half, dtype=torch.float32)
        spiral = 1.0 + alpha * torch.sqrt(idx + 1.0) / math.sqrt(half)
        inv_freq = base_inv_freq * spiral  # [d_h / 2]
        # Registered as a non-persistent buffer (recomputed from config on load).
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq = self.inv_freq.to(x.device)
        inv_freq_expanded = inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1
        )  # [B, d_h/2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [B, 1, seq]

        # Force fp32 for the trig computation regardless of autocast.
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)  # [B, seq, d_h/2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [B, seq, d_h]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dimension (LLaMA convention)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to the query and key tensors.

    ``q``/``k`` are ``[B, H, seq, d_h]``; ``cos``/``sin`` are ``[B, seq, d_h]``.
    """
    cos = cos.unsqueeze(1)  # [B, 1, seq, d_h]
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# =============================================================================
# Butterfly MLP
# =============================================================================
class WiolaButterflyMLP(nn.Module):
    r"""Butterfly multiplicative feed-forward block.

    ::

        [a, b] = x @ W_up^T          # W_up:     d -> 2 * d_inner
        r      = SiLU(a) * b + W_bypass(x)   # W_bypass: d ->     d_inner
        out    = r @ W_down^T        # W_down:   d_inner -> d

    With ``d_inner = 2 * d`` this matches a GeLU 4x FFN in parameter count while
    providing SwiGLU-class multiplicative gating plus a stabilising bypass.
    """

    def __init__(self, config: WiolaConfig):
        super().__init__()
        d = config.hidden_size
        d_inner = config.intermediate_size
        self.up_proj = nn.Linear(d, 2 * d_inner, bias=False)
        self.bypass_proj = nn.Linear(d, d_inner, bias=False)
        self.down_proj = nn.Linear(d_inner, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.up_proj(x).chunk(2, dim=-1)
        r = F.silu(a) * b + self.bypass_proj(x)
        return self.down_proj(r)


# =============================================================================
# Gated Spiral Attention
# =============================================================================
class WiolaAttention(nn.Module):
    """Gated Spiral Attention (masked multi-head self-attention + causal gate)."""

    def __init__(self, config: WiolaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.attention_dropout = config.attention_dropout
        self.use_attention_gate = config.use_attention_gate
        self.gate_pre_rope = config.gate_pre_rope

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # Content-adaptive gate network:  W1: d_h -> H ,  W2: H -> H (with bias)
        if self.use_attention_gate:
            self.gate_fc1 = nn.Linear(self.head_dim, self.num_heads, bias=False)
            self.gate_fc2 = nn.Linear(self.num_heads, self.num_heads, bias=True)

        self.attn_dropout = nn.Dropout(self.attention_dropout)

        self.rotary_emb = WiolaSpiralRotaryEmbedding(
            dim=self.head_dim,
            base=config.rope_theta,
            alpha=config.spiral_alpha,
        )

    # -- gate running-state helpers (attached to the cache object) ------------
    def _get_gate_sum(self, cache: Cache) -> Optional[torch.Tensor]:
        store = getattr(cache, "_wiola_gate_sum", None)
        if store is None:
            return None
        return store.get(self.layer_idx, None)

    def _set_gate_sum(self, cache: Cache, value: torch.Tensor) -> None:
        if not hasattr(cache, "_wiola_gate_sum"):
            cache._wiola_gate_sum = {}
        cache._wiola_gate_sum[self.layer_idx] = value

    def _compute_gate(
        self,
        query_ctx: torch.Tensor,   # [B, H, q_len, d_h]  (pre- or post-RoPE)
        past_len: int,
        past_key_value: Optional[Cache],
        use_cache: bool,
    ) -> torch.Tensor:
        """Return the per-head gate broadcastable over scores: ``[B, H, q_len, 1]``."""
        bsz, _, q_len, _ = query_ctx.shape
        out_dtype = query_ctx.dtype

        # Head-averaged query context m_s = (1/H) sum_h q_s^(h)  ->  [B, q_len, d_h].
        # The cumulative statistic is accumulated in fp32: positions can exceed the
        # exact-integer range of bf16/fp16, and the running sum grows over decoding.
        m = query_ctx.float().mean(dim=1)

        prev_sum = None
        if past_key_value is not None:
            prev_sum = self._get_gate_sum(past_key_value)
        if prev_sum is None:
            prev_sum = torch.zeros(bsz, self.head_dim, dtype=torch.float32, device=m.device)

        # Cumulative (causal) sum of head-averaged contexts, offset by cached prefix.
        csum = torch.cumsum(m, dim=1) + prev_sum.to(m.dtype).unsqueeze(1)  # [B, q_len, d_h]
        positions = torch.arange(
            past_len + 1, past_len + q_len + 1, dtype=torch.float32, device=m.device
        ).view(1, -1, 1)
        c = (csum / positions).to(out_dtype)  # causal cumulative mean  [B, q_len, d_h]

        if use_cache and past_key_value is not None:
            # Persist the full (fp32) running sum so the next decode step continues the mean.
            self._set_gate_sum(past_key_value, csum[:, -1, :])

        # g_t = sigmoid(W2 · SiLU(W1 · c_t))  ->  [B, q_len, H]
        g = self.gate_fc1(c)
        g = F.silu(g)
        g = self.gate_fc2(g)
        g = torch.sigmoid(g)
        # -> [B, H, q_len, 1] so each query row is scaled by its per-head gate.
        return g.transpose(1, 2).unsqueeze(-1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_key_value is not None:
            past_len = past_key_value.get_seq_length(self.layer_idx)

        # Rotary embeddings (Spiral RoPE).
        cos, sin = self.rotary_emb(value_states, position_ids)

        # Optionally derive the gate from the *pre-RoPE* (content) queries.
        gate = None
        if self.use_attention_gate and self.gate_pre_rope:
            gate = self._compute_gate(query_states, past_len, past_key_value, use_cache)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Or from the *post-RoPE* queries (matches the figure in the design doc).
        if self.use_attention_gate and not self.gate_pre_rope:
            gate = self._compute_gate(query_states, past_len, past_key_value, use_cache)

        # KV cache update (store post-RoPE keys, as usual).
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        # Scaled dot-product scores, then multiplicative gate (pre-softmax).
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if gate is not None:
            attn_weights = attn_weights * gate

        if attention_mask is not None:
            # attention_mask is a 4D additive mask [B, 1, q_len, kv_len].
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, value_states)  # [B, H, q_len, d_h]
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


# =============================================================================
# Decoder layer
# =============================================================================
class WiolaDecoderLayer(nn.Module):
    def __init__(self, config: WiolaConfig, layer_idx: int):
        super().__init__()
        self.self_attn = WiolaAttention(config, layer_idx)
        self.mlp = WiolaButterflyMLP(config)
        self.input_layernorm = WiolaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = WiolaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Cache]]:
        # Pre-norm attention block with residual.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Pre-norm Butterfly MLP block with residual.
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


# =============================================================================
# Pre-trained base
# =============================================================================
class WiolaPreTrainedModel(PreTrainedModel):
    config_class = WiolaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["WiolaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-2 * std, b=2 * std)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, WiolaRMSNorm):
            module.weight.data.fill_(1.0)

        # Gate biases start at zero so gates begin close to the neutral 0.5.
        if isinstance(module, WiolaAttention) and module.use_attention_gate:
            nn.init.zeros_(module.gate_fc2.bias)


# =============================================================================
# Base model (embeddings + stack + final norm)
# =============================================================================
class WiolaModel(WiolaPreTrainedModel):
    def __init__(self, config: WiolaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [WiolaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = WiolaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def _build_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        q_len: int,
        kv_len: int,
        past_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        min_val = torch.finfo(dtype).min
        rows = torch.arange(q_len, device=device).view(-1, 1) + past_len
        cols = torch.arange(kv_len, device=device).view(1, -1)
        allowed = cols <= rows  # causal: key j visible to query i iff j <= past_len + i
        mask = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, min_val)
        mask = mask[None, None, :, :]  # [1, 1, q_len, kv_len]

        if attention_mask is not None:
            pad = (1.0 - attention_mask[:, None, None, :].to(dtype)) * min_val
            mask = mask + pad
        return mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,    # ← accepted and ignored
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of `input_ids` or `inputs_embeds`.")

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        bsz, q_len = inputs_embeds.shape[:2]

        past_len = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_len = past_key_values.get_seq_length()
        elif past_key_values is not None:
            past_len = past_key_values.get_seq_length()

        if cache_position is None:
            cache_position = torch.arange(
                past_len, past_len + q_len, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        kv_len = past_len + q_len
        causal_mask = self._build_causal_mask(
            attention_mask, q_len, kv_len, past_len, inputs_embeds.dtype, inputs_embeds.device
        )

        hidden_states = inputs_embeds

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = past_key_values if use_cache else None

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


# =============================================================================
# Causal LM head
# =============================================================================
class WiolaForCausalLM(WiolaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: WiolaConfig):
        super().__init__(config)
        self.model = WiolaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        self.tie_weights()                  # explicitly tie weights after init

    # -- embedding / head plumbing -------------------------------------------
    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,    # ← accepted and ignored
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modelling loss. Indices
            should be in ``[0, ..., config.vocab_size]`` or ``-100`` (ignored).
        """
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states).float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    # -- generation helpers ---------------------------------------------------
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        # Remove any token_type_ids that might be passed by the tokenizer
        kwargs.pop("token_type_ids", None)

        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_len = past_key_values.get_seq_length()
            else:
                past_len = past_key_values[0][0].shape[2]
            if input_ids.shape[1] > past_len:
                input_ids = input_ids[:, past_len:]
            else:
                input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids.contiguous()}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        # Reorder standard key/value cache.
        if isinstance(past_key_values, DynamicCache):
            for i in range(len(past_key_values.key_cache)):
                if past_key_values.key_cache[i].numel():
                    dev = past_key_values.key_cache[i].device
                    past_key_values.key_cache[i] = past_key_values.key_cache[i].index_select(
                        0, beam_idx.to(dev)
                    )
                    past_key_values.value_cache[i] = past_key_values.value_cache[i].index_select(
                        0, beam_idx.to(dev)
                    )
        elif past_key_values is not None and not isinstance(past_key_values, Cache):
            reordered = ()
            for layer_past in past_key_values:
                reordered += (
                    tuple(p.index_select(0, beam_idx.to(p.device)) for p in layer_past),
                )
            past_key_values = reordered

        # Reorder the gate running-state so beam search stays causally correct.
        store = getattr(past_key_values, "_wiola_gate_sum", None)
        if store is not None:
            for lidx, val in store.items():
                store[lidx] = val.index_select(0, beam_idx.to(val.device))
        return past_key_values