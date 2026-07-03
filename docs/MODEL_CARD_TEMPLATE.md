---
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
  - wiola
  - small-language-model
  - gated-spiral-attention
  - butterfly-mlp
---

# Wiola Nano

A ~13M-parameter Gated Spiral Attention small language model. See the
[architecture reference](https://github.com/wiola-project/wiola/blob/main/docs/ARCHITECTURE.md).

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tok = AutoTokenizer.from_pretrained("USERNAME/wiola-nano")
model = AutoModelForCausalLM.from_pretrained("USERNAME/wiola-nano", trust_remote_code=True)

ids = tok("Once upon a time", return_tensors="pt")
out = model.generate(**ids, max_new_tokens=100, do_sample=True, temperature=0.8, top_p=0.95)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Architecture

- **Spiral RoPE** — RoPE with a `sqrt`-growing per-pair frequency perturbation.
- **Gated Spiral Attention** — causal, content-adaptive per-head attention gate.
- **Butterfly MLP** — multiplicative FFN with an intra-block bypass.

| `d` | `L` | `H` | `d_inner` | vocab | ctx | params |
|----:|----:|----:|----------:|------:|----:|-------:|
| 256 | 6 | 8 | 512 | 32000 | 1024 | ~12.9M |

## Training

<!-- Fill in: dataset, steps, tokens, hardware, final loss/eval. -->

## Limitations

Small models hallucinate and have limited world knowledge. This model is intended
for research, education, edge deployment, and fine-tuning.

## License

Apache 2.0.
