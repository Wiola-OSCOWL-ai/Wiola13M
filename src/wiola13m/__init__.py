# coding=utf-8
# Copyright 2025 The Wiola Project. Apache License 2.0.
"""Wiola — Gated Spiral Attention small language model."""

from .version import __version__
from .configuration_wiola import WiolaConfig
from .modeling_wiola import (
    WiolaForCausalLM,
    WiolaModel,
    WiolaPreTrainedModel,
    WiolaAttention,
    WiolaButterflyMLP,
    WiolaDecoderLayer,
    WiolaRMSNorm,
    WiolaSpiralRotaryEmbedding,
)

__all__ = [
    "__version__",
    "WiolaConfig",
    "WiolaForCausalLM",
    "WiolaModel",
    "WiolaPreTrainedModel",
    "WiolaAttention",
    "WiolaButterflyMLP",
    "WiolaDecoderLayer",
    "WiolaRMSNorm",
    "WiolaSpiralRotaryEmbedding",
]


def _register_for_auto_class() -> None:
    """Register Wiola with the Hugging Face ``Auto*`` factories.

    This makes ``AutoConfig.from_pretrained`` / ``AutoModelForCausalLM``
    resolve the ``"wiola"`` ``model_type`` locally (no ``trust_remote_code``
    needed once this package is installed). Safe to call more than once.
    """
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        AutoConfig.register("wiola", WiolaConfig, exist_ok=True)
        AutoModel.register(WiolaConfig, WiolaModel, exist_ok=True)
        AutoModelForCausalLM.register(WiolaConfig, WiolaForCausalLM, exist_ok=True)
    except Exception:  # pragma: no cover - transformers not installed / already registered
        pass


_register_for_auto_class()
