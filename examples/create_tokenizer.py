"""Create a tokenizer for Wiola.

Two options:

1. **Reuse** an existing 32k SentencePiece/BPE tokenizer (recommended, matches
   the design doc which reuses the LLaMA tokenizer):

       python examples/create_tokenizer.py reuse \
           --source meta-llama/Llama-2-7b-hf --out ./wiola-tokenizer

2. **Train** a fresh byte-level BPE tokenizer on a text dataset:

       python examples/create_tokenizer.py train \
           --dataset roneneldan/TinyStories --vocab-size 32000 --out ./wiola-tokenizer

Requires: pip install "wiola[train]"  (datasets, tokenizers)
"""
import argparse


def reuse(args) -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.source)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    #  Prevent token_type_ids from being generated
    tok.model_input_names = ["input_ids", "attention_mask"]

    tok.save_pretrained(args.out)
    print(f"[done] Saved reused tokenizer (vocab {len(tok)}) to {args.out}")


def train(args) -> None:
    from datasets import load_dataset
    from tokenizers import ByteLevelBPETokenizer
    from transformers import PreTrainedTokenizerFast

    ds = load_dataset(args.dataset, split="train")

    def corpus():
        for i in range(0, len(ds), 1000):
            yield [t for t in ds[i : i + 1000]["text"]]

    bpe = ByteLevelBPETokenizer()
    specials = ["<unk>", "<s>", "</s>", "<pad>"]
    bpe.train_from_iterator(
        (line for batch in corpus() for line in batch),
        vocab_size=args.vocab_size,
        min_frequency=2,
        special_tokens=specials,
    )

    fast = PreTrainedTokenizerFast(
        tokenizer_object=bpe._tokenizer,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
    )
    #  Ensure no token_type_ids
    fast.model_input_names = ["input_ids", "attention_mask"]

    fast.save_pretrained(args.out)
    print(f"[done] Trained + saved BPE tokenizer (vocab {args.vocab_size}) to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reuse = sub.add_parser("reuse")
    p_reuse.add_argument("--source", required=True)
    p_reuse.add_argument("--out", default="./wiola-tokenizer")
    p_reuse.set_defaults(func=reuse)

    p_train = sub.add_parser("train")
    p_train.add_argument("--dataset", default="roneneldan/TinyStories")
    p_train.add_argument("--vocab-size", type=int, default=32000)
    p_train.add_argument("--out", default="./wiola-tokenizer")
    p_train.set_defaults(func=train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
