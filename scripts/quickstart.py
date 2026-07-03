"""End-to-end smoke test: build Wiola Nano, run a forward + backward, generate.

    python scripts/quickstart.py
"""
import torch

from wiola13m import WiolaConfig, WiolaForCausalLM


def main() -> None:
    torch.manual_seed(0)
    config = WiolaConfig()  # Nano
    model = WiolaForCausalLM(config)

    n_params = sum(p.numel() for p in model.parameters())
    unique = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    print(f"Wiola Nano — total params: {n_params/1e6:.2f}M  (unique/tied: {unique/1e6:.2f}M)")

    ids = torch.randint(3, 1000, (2, 32))

    # Forward + loss + backward.
    out = model(input_ids=ids, labels=ids)
    print(f"loss = {out.loss.item():.4f}")
    out.loss.backward()
    print("backward OK")

    # Generation smoke test with KV cache.
    model.eval()
    with torch.no_grad():
        gen = model.generate(ids[:1, :4], max_new_tokens=16, do_sample=False)
    print(f"generated ids: {gen[0].tolist()}")
    print("quickstart complete ✔")


if __name__ == "__main__":
    main()
