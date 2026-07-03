<h1 align="center">Wiola</h1>

<p align="center">
  <b>Gated Spiral Attention — a small language model built for the 10–100M parameter regime</b><br>
  <i>Spiral RoPE · content-adaptive attention gating · Butterfly MLP</i>
</p>

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.9%2B-blue">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c">
  <img alt="Transformers" src="https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-compatible-yellow">
</p>

---

Wiola is a decoder-only small language model whose novelty lives entirely in two
sub-components of every layer. It is designed to run on a laptop, train on a
single consumer GPU in hours, and publish to the Hugging Face Hub — yet be
architecturally distinct enough to serve as a real experimental baseline.

| Variant | `d` | `L` | `H` | `d_inner` | Params |
|--------:|----:|----:|----:|----------:|-------:|
| **Nano**  | 256 | 6  | 8  | 512  | ~12.9M |
| Micro | 384 | 8  | 12 | 768  | ~40M |
| Small | 512 | 12 | 16 | 1024 | ~90M |

## What's novel

1. **Spiral Rotary Positional Encoding.** Standard RoPE frequencies are perturbed
   by a `sqrt`-growing, per-dimension-pair factor so phase trajectories *fan
   outward* instead of staying collinear — improving long-range discrimination at
   **zero** added parameters. Setting `spiral_alpha=0.0` recovers standard RoPE exactly.

2. **Gated Spiral Attention (GSA).** A per-head, content-adaptive scalar gate,
   derived *causally* from a cumulative mean of the query projections, modulates
   attention scores before softmax. Heads that don't help self-suppress — implicit
   soft head pruning with no sparsity loss. The gate adds `2·H·d_h + H²` params
   (a few hundred for Nano) and is fully KV-cache compatible.

3. **Butterfly MLP.** A multiplicative feed-forward block, `SiLU(a) ⊙ b`, plus an
   intra-block bypass `W_bypass·x`. With `d_inner = 2d` it matches a GeLU 4× FFN in
   parameter count while providing SwiGLU-class gating and steadier gradients in
   shallow stacks.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full math.

## Install

```bash
# from source (recommended while pre-1.0)
gh repo clone Wiola-OSCOWL-ai/Wiola13M
cd wiola
pip install -e .

# with training / hub extras
pip install -e ".[train,hub]"
```

From PyPI once published:

```bash
pip install wiola
```

> **Version note:** the model uses the modern `transformers` `Cache` API. Pinned to
> `transformers>=4.40,<4.46`, the range this release is tested against.

## Quickstart

```python
import torch
from wiola13m import WiolaConfig, WiolaForCausalLM

model = WiolaForCausalLM(WiolaConfig())          # Wiola Nano, random init
ids = torch.randint(0, 32000, (1, 16))

out = model(input_ids=ids, labels=ids)           # forward + LM loss
out.loss.backward()                              # gradients flow

model.eval()
gen = model.generate(ids[:, :4], max_new_tokens=20, do_sample=False)
```

Or run the bundled smoke test:

```bash
python scripts/quickstart.py
```

## Train on TinyStories

```bash
# 1) get a 32k tokenizer (reuse a LLaMA tokenizer, or train your own)
python examples/create_tokenizer.py reuse --source meta-llama/Llama-2-7b-hf --out ./wiola-tokenizer

# 2) pre-train Nano (~2h/epoch on an RTX 3090)
python examples/train_tinystories.py \
    --tokenizer ./wiola-tokenizer \
    --output-dir ./wiola-nano-tinystories \
    --max-steps 20000

# 3) generate
python examples/generate.py --model ./wiola-nano-tinystories --prompt "Once upon a time"
```

## Publish to the Hugging Face Hub

Wiola ships with `auto_map` support, so anyone can load your model without
installing this package:

```bash
huggingface-cli login
python examples/push_to_hub.py --model-dir ./wiola-nano-tinystories --repo-id your-name/wiola-nano
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("your-name/wiola-nano", trust_remote_code=True)
tok = AutoTokenizer.from_pretrained("your-name/wiola-nano")
```

If the `wiola` package is installed, the `"wiola"` architecture is auto-registered
and you don't even need `trust_remote_code=True`.

## Design decision: gate input

The design doc's figure feeds the gate from *post-RoPE* queries, while the prose
describes it as *content-adaptive*. Wiola defaults to computing the gate from the
**pre-RoPE** query projections (`gate_pre_rope=True`) — position-independent and
numerically stable — and exposes `gate_pre_rope=False` to match the figure. Both are
causally correct and KV-cache safe.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The suite verifies output shapes, weight tying, strict causality (no future-token
leakage), **exact equivalence between cached step-by-step decoding and a single
full-sequence forward** (with and without the gate), save/reload round-trips, and
greedy/sampling/batched/beam generation.

## Project layout

```
wiola/
├── src/wiola/
│   ├── configuration_wiola.py   # WiolaConfig
│   ├── modeling_wiola.py        # Spiral RoPE, GSA, Butterfly MLP, decoder, CausalLM
│   └── __init__.py              # Auto* registration
├── examples/                    # train / generate / tokenizer / push_to_hub
├── scripts/quickstart.py
├── tests/
└── docs/ARCHITECTURE.md
```

## Citation

If you use Wiola, please cite it (see [`CITATION.cff`](CITATION.cff)).

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
