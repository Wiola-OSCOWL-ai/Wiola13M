"""Publish a trained Wiola model to the Hugging Face Hub with custom code.

This wires up `auto_map` so anyone can load your model with
`AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)` even
without the `wiola` package installed.

Usage:
    huggingface-cli login
    python examples/push_to_hub.py \
        --model-dir ./wiola-nano-tinystories \
        --repo-id your-username/wiola-nano

Requires: pip install "wiola[hub]"
"""
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

import wiola13m13m  # noqa: F401
from wiola13m import WiolaConfig, WiolaForCausalLM


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Local dir of a trained Wiola model.")
    parser.add_argument("--repo-id", required=True, help="Target Hub repo, e.g. user/wiola-nano.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer to bundle (default: --model-dir).")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    # Ensure the config carries auto_map so remote code loading works.
    config = WiolaConfig.from_pretrained(args.model_dir)
    config.auto_map = {
        "AutoConfig": "configuration_wiola.WiolaConfig",
        "AutoModel": "modeling_wiola.WiolaModel",
        "AutoModelForCausalLM": "modeling_wiola.WiolaForCausalLM",
    }

    # Registering for auto class tells `save_pretrained`/`push_to_hub` to copy
    # the defining source files into the repo.
    WiolaConfig.register_for_auto_class()
    WiolaForCausalLM.register_for_auto_class("AutoModelForCausalLM")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, config=config
    )
    model.config = config

    model.push_to_hub(args.repo_id, private=args.private)

    tok_src = args.tokenizer or args.model_dir
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        tokenizer.push_to_hub(args.repo_id, private=args.private)
    except Exception as exc:  # pragma: no cover
        print(f"[warn] Could not push a tokenizer from {tok_src}: {exc}")

    print(f"[done] Pushed to https://huggingface.co/{args.repo_id}")
    print("Load with: AutoModelForCausalLM.from_pretrained(repo_id, trust_remote_code=True)")


if __name__ == "__main__":
    main()
