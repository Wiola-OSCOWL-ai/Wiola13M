# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/), and this project adheres to
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2025

### Added
- Initial public release of **Wiola**, a Gated Spiral Attention small language model.
- `WiolaConfig`, `WiolaModel`, `WiolaForCausalLM` with full Hugging Face `Auto*`
  compatibility (local registration + `auto_map` for `trust_remote_code`).
- **Spiral Rotary Positional Encoding** (`spiral_alpha`, `alpha=0` ⇒ standard RoPE).
- **Gated Spiral Attention** with a causal, content-adaptive per-head gate that is
  fully KV-cache compatible; cached decoding is numerically identical to a
  full-sequence forward.
- **Butterfly MLP** (multiplicative gating + intra-block bypass).
- Nano / Micro / Small variants.
- Examples: TinyStories training, generation, tokenizer creation, Hub publishing.
- Test suite covering shapes, causality, cache equivalence, save/reload, and
  greedy/sampling/batched/beam generation.
- Apache-2.0 license, CI, packaging (`pip install wiola`).
