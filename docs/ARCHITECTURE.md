# Wiola Architecture

Decoder-only Transformer: token embedding → `L` decoder layers → final RMSNorm →
tied LM head. Each layer is pre-norm with residuals:

```
H'  = H  + GSA(RMSNorm(H))
H'' = H' + ButterflyMLP(RMSNorm(H'))
```

## 1. Spiral Rotary Positional Encoding

Per head dimension `d_h`, dimension pair `i ∈ {0, …, d_h/2 − 1}`:

```
theta_i        = 1 / base^(2i / d_h)                       # standard RoPE
phi_i          = 1 + alpha · sqrt(i + 1) / sqrt(d_h / 2)    # spiral perturbation
theta_tilde_i  = theta_i · phi_i
```

with `base = 10000`, `alpha = 0.05`. Rotation for position `m`:
`Theta_{m,i} = m · theta_tilde_i`, applied to consecutive dim pairs via a 2×2
rotation (implemented with the LLaMA `rotate_half` convention). `alpha = 0`
recovers standard RoPE exactly. The `sqrt`-growing `phi_i` makes higher pairs
receive progressively larger frequency boosts, so distant positions get more
uniformly scattered phases (better long-range discrimination) while adjacent
positions keep small angular separations (local continuity).

## 2. Gated Spiral Attention

Projections per head `h` (dim `d_h = d/H`): `Q, K, V = H·W^{Q,K,V}`.
Spiral RoPE is applied to `Q` and `K`.

**Causal gate.** Head-averaged query context `m_s = (1/H) Σ_h q_s^{(h)}`, then a
causal cumulative mean:

```
c_t = (1/t) Σ_{s≤t} m_s                       # cumulative mean over positions
g_t = sigmoid(W2 · SiLU(W1 · c_t)) ∈ (0,1)^H  # W1: d_h→H, W2: H→H (+bias)
```

`c_t` depends only on positions `≤ t`, so the gate is strictly causal. During
KV-cached decoding the running sum `Σ_{s≤t} m_s` is carried in the cache, making
step-by-step decoding numerically identical to a full-sequence forward.

**Gated scores.** For head `h`, with per-position gate `g_t^{(h)}` broadcast over
keys:

```
A^{(h)}_{ts} = (1/sqrt(d_h)) · (Q^{(h)} K^{(h)T})_{ts} · g_t^{(h)}
Z^{(h)}      = softmax(A^{(h)} + M) · V^{(h)}      # M causal mask
GSA(H)       = concat(Z^{(1)}, …, Z^{(H)}) · W^O
```

If a head is unhelpful, gradients drive `g_t^{(h)} → 0` via the saturating
sigmoid — implicit soft head pruning with no explicit sparsity objective.

> **Gate input (implementation choice).** The design figure feeds the gate from
> post-RoPE queries; the prose calls it content-adaptive. Wiola defaults to
> **pre-RoPE** queries (`gate_pre_rope=True`) for position-independence and
> stability, with `gate_pre_rope=False` available to match the figure.

## 3. Butterfly MLP

```
[a, b] = x · W_up^T            # W_up:     d → 2·d_inner
r      = SiLU(a) ⊙ b + W_bypass·x   # W_bypass: d →   d_inner  (bypass)
out    = r · W_down^T          # W_down:   d_inner → d
```

At `d_inner = 2d` the three matrices total `4·d·d_inner` params — the same as a
GeLU 4× FFN — while providing SwiGLU-class multiplicative gating plus a bypass
that stabilises gradients in shallow networks.

## 4. Objective

Next-token cross-entropy with tied input/output embeddings:

```
L = -(1/T) Σ_t log p_theta(x_t | x_{<t}),   p_theta = softmax(W_out · H''_t)
```

## Complexity

Attention stays `O(T² d)`; the gate adds only `O(T d_h H)` (linear in `T`). The
Butterfly MLP is `O(T · d · d_inner) ≈ O(T · 2d²)`, matching a 4× GeLU FFN.

## Parameter budget (Nano, verified)

```
head_dim               = 32
attn per layer (QKVO)  = 4·256²          = 262,144
gate per layer         = H·d_h + (H²+H)  = 328
butterfly per layer    = 4·256·512       = 524,288
rmsnorm per layer (×2) = 2·256           = 512
per layer              ≈ 787,272
× 6 layers             ≈ 4,723,632
token embedding (tied) = 32000·256       = 8,192,000
final rmsnorm          = 256
------------------------------------------------
grand total (tied)     ≈ 12,915,888  (~12.9M)
```
