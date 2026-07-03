"""Minimal text-generation example for Wiola.

Usage:
    python examples/generate.py --model path/or/hub-id --prompt "Once upon a time"

If you pass a freshly-initialised model (random weights) you'll get gibberish;
this script is mostly a smoke test / API demonstration.
"""
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import wiola13m13m  # noqa: F401  (registers the "wiola" architecture with Auto* classes)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Local path or Hub id of a Wiola model.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path/id (defaults to --model).")
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model is None:
        print("[info] No --model given; building a random Wiola Nano for a smoke test.")
        from wiola13m import WiolaConfig, WiolaForCausalLM

        model = WiolaForCausalLM(WiolaConfig()).to(device).eval()
        vocab = model.config.vocab_size
        input_ids = torch.randint(3, min(vocab, 1000), (1, 6), device=device)
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        print("[info] Generated token ids:", out[0].tolist())
        return

    tok_id = args.tokenizer or args.model
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.float32
    ).to(device).eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=not args.greedy,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    print(tokenizer.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
