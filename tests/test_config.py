"""Configuration tests for Wiola."""
import pytest

from wiola13m import WiolaConfig


def test_default_config_is_nano():
    cfg = WiolaConfig()
    assert cfg.model_type == "wiola"
    assert cfg.hidden_size == 256
    assert cfg.num_hidden_layers == 6
    assert cfg.num_attention_heads == 8
    assert cfg.head_dim == 32
    assert cfg.intermediate_size == 512
    assert cfg.tie_word_embeddings is True


def test_head_dim_divisibility_is_enforced():
    with pytest.raises(ValueError):
        WiolaConfig(hidden_size=256, num_attention_heads=7)


def test_config_roundtrip(tmp_path):
    cfg = WiolaConfig(hidden_size=128, num_attention_heads=4, spiral_alpha=0.1)
    cfg.save_pretrained(tmp_path)
    loaded = WiolaConfig.from_pretrained(tmp_path)
    assert loaded.hidden_size == 128
    assert loaded.num_attention_heads == 4
    assert loaded.spiral_alpha == 0.1
    assert loaded.head_dim == 32
