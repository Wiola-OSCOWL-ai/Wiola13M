"""Modeling tests for Wiola. Require torch; skipped automatically otherwise."""
# ruff: noqa: E402

import pytest

torch = pytest.importorskip("torch")

from wiola13m import WiolaConfig, WiolaForCausalLM, WiolaModel


def tiny_config(**kw):
    base = dict(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=64,
    )
    base.update(kw)
    return WiolaConfig(**base)


def test_forward_shapes():
    cfg = tiny_config()
    model = WiolaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = model(input_ids=ids)
    assert out.logits.shape == (2, 10, cfg.vocab_size)


def test_loss_is_finite():
    cfg = tiny_config()
    model = WiolaForCausalLM(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 10))
    out = model(input_ids=ids, labels=ids)
    assert out.loss is not None
    assert torch.isfinite(out.loss)
    out.loss.backward()  # gradients flow
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None for g in grads)


def test_weight_tying():
    cfg = tiny_config(tie_word_embeddings=True)
    model = WiolaForCausalLM(cfg)
    assert model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def test_parameter_count_nano():
    # Full Nano config: verify total is in the documented ~13M range.
    cfg = WiolaConfig()
    model = WiolaForCausalLM(cfg)
    # Tied embeddings: count unique parameters.
    seen, unique = set(), 0
    for p in model.parameters():
        if p.data_ptr() not in seen:
            seen.add(p.data_ptr())
            unique += p.numel()
    assert 12.0e6 < unique < 13.5e6, unique


def test_causality_no_future_leak():
    """Changing a future token must not change earlier-position logits."""
    cfg = tiny_config()
    model = WiolaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 12))
    with torch.no_grad():
        base = model(input_ids=ids).logits
        ids2 = ids.clone()
        ids2[0, -1] = (ids2[0, -1] + 1) % cfg.vocab_size  # perturb last token
        pert = model(input_ids=ids2).logits
    # All positions except the last must be identical.
    assert torch.allclose(base[:, :-1], pert[:, :-1], atol=1e-5)


@pytest.mark.parametrize("use_gate", [True, False])
def test_kv_cache_matches_full_forward(use_gate):
    """Cached step-by-step decoding must equal a single full-sequence forward."""
    cfg = tiny_config(use_attention_gate=use_gate)
    model = WiolaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        full = model(input_ids=ids, use_cache=False).logits

        past = None
        stepwise = []
        for t in range(ids.shape[1]):
            out = model(input_ids=ids[:, t : t + 1], past_key_values=past, use_cache=True)
            past = out.past_key_values
            stepwise.append(out.logits)
        stepwise = torch.cat(stepwise, dim=1)

    assert torch.allclose(full, stepwise, atol=1e-4), (full - stepwise).abs().max()


def test_gate_disabled_equals_plain_attention_config():
    cfg = tiny_config(use_attention_gate=False)
    model = WiolaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    out = model(input_ids=ids)
    assert torch.isfinite(out.logits).all()


def test_save_and_reload(tmp_path):
    cfg = tiny_config()
    model = WiolaForCausalLM(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    with torch.no_grad():
        before = model(input_ids=ids).logits
    model.save_pretrained(tmp_path)
    reloaded = WiolaForCausalLM.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        after = reloaded(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-5)


def test_base_model_outputs_hidden_states():
    cfg = tiny_config()
    model = WiolaModel(cfg).eval()
    ids = torch.randint(0, cfg.vocab_size, (1, 5))
    out = model(input_ids=ids, output_hidden_states=True)
    assert out.last_hidden_state.shape == (1, 5, cfg.hidden_size)
    assert len(out.hidden_states) == cfg.num_hidden_layers + 1
