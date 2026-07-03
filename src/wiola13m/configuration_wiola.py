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
"""Wiola model configuration."""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class WiolaConfig(PretrainedConfig):
    r"""
    Configuration class for the **Wiola** Gated Spiral Attention small language model.

    This is the configuration class to store the configuration of a
    [`WiolaModel`] or [`WiolaForCausalLM`]. It is used to instantiate a Wiola
    model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults yields the **Wiola Nano**
    (~13M parameters) configuration.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to
    control the model outputs. Read the documentation of [`PretrainedConfig`]
    for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 32000):
            Vocabulary size of the Wiola model.
        hidden_size (`int`, *optional*, defaults to 256):
            Dimension of the hidden representations (``d`` in the paper).
        intermediate_size (`int`, *optional*, defaults to 512):
            Inner dimension of the Butterfly MLP (``d_inner``). For an exact
            parameter match with a GeLU 4x FFN, keep ``intermediate_size == 2 *
            hidden_size``.
        num_hidden_layers (`int`, *optional*, defaults to 6):
            Number of decoder layers (``L``).
        num_attention_heads (`int`, *optional*, defaults to 8):
            Number of attention heads (``H``). ``hidden_size`` must be divisible
            by this value.
        max_position_embeddings (`int`, *optional*, defaults to 1024):
            The maximum sequence length the model can handle.
        rope_theta (`float`, *optional*, defaults to 10000.0):
            The base period used to build the rotary frequencies (``base``).
        spiral_alpha (`float`, *optional*, defaults to 0.05):
            The spiral perturbation coefficient (``alpha``). ``0.0`` recovers
            standard RoPE.
        gate_pre_rope (`bool`, *optional*, defaults to `True`):
            If `True`, the content-adaptive gate is computed from the query
            projections **before** Spiral RoPE is applied (content-adaptive,
            position-independent, numerically stable). If `False`, the gate is
            computed from the rotated queries. See the architecture notes for
            details.
        use_attention_gate (`bool`, *optional*, defaults to `True`):
            If `False`, the Gated Spiral Attention degrades to standard
            (Spiral) RoPE attention, disabling the content-adaptive gate.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the RMSNorm layers.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio applied to the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated-normal initializer.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether the model returns the key/value cache used for fast decoding.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether to tie the input embedding and the output (LM head) weights.
        pad_token_id (`int`, *optional*): Padding token id.
        bos_token_id (`int`, *optional*, defaults to 1): Beginning-of-stream id.
        eos_token_id (`int`, *optional*, defaults to 2): End-of-stream id.

    Example:

    ```python
    >>> from wiola13m import WiolaConfig, WiolaForCausalLM

    >>> # Initialise a Wiola Nano configuration
    >>> configuration = WiolaConfig()

    >>> # Initialise a model (with random weights) from that configuration
    >>> model = WiolaForCausalLM(configuration)

    >>> # Access the configuration
    >>> configuration = model.config
    ```
    """

    model_type = "wiola"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 256,
        intermediate_size: int = 512,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        max_position_embeddings: int = 1024,
        rope_theta: float = 10000.0,
        spiral_alpha: float = 0.05,
        gate_pre_rope: bool = True,
        use_attention_gate: bool = True,
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        pad_token_id: int = None,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.spiral_alpha = spiral_alpha
        self.gate_pre_rope = gate_pre_rope
        self.use_attention_gate = use_attention_gate
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                f"`hidden_size` ({hidden_size}) must be divisible by "
                f"`num_attention_heads` ({num_attention_heads})."
            )

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    @property
    def head_dim(self) -> int:
        """Dimension of a single attention head (``d_h = hidden_size // H``)."""
        return self.hidden_size // self.num_attention_heads
