# ruff: noqa: E402
"""Generation tests for Wiola. Require torch; skipped automatically otherwise."""
import pytest

torch = pytest.importorskip("torch")

from wiola13m import WiolaConfig, WiolaForCausalLM

def tiny_model():
    cfg = WiolaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
        bos_token_id=1,
        eos_token_id=2,
    )
    return WiolaForCausalLM(cfg).eval()

def test_greedy_generate_runs():
    model = tiny_model()
    ids = torch.randint(3, 64, (1, 4))
    out = model.generate(ids, max_new_tokens=8, do_sample=False)
    assert out.shape[1] == 4 + 8

def test_sampling_generate_runs():
    model = tiny_model()
    torch.manual_seed(0)
    ids = torch.randint(3, 64, (1, 4))
    out = model.generate(
        ids, max_new_tokens=8, do_sample=True, top_k=10, temperature=0.9
    )
    assert out.shape[1] == 4 + 8

def test_batched_generate_runs():
    model = tiny_model()
    ids = torch.randint(3, 64, (3, 5))
    attn = torch.ones_like(ids)
    out = model.generate(ids, attention_mask=attn, max_new_tokens=6, do_sample=False)
    assert out.shape == (3, 5 + 6)
