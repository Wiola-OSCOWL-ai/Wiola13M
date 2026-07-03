"""Pre-train Wiola Nano on TinyStories with the Hugging Face Trainer.

This mirrors the recommended configuration in the design doc (AdamW, cosine LR,
bf16, ~512-token sequences). It is intentionally compact and single-file.

Quick start (needs `datasets` and a tokenizer):
    pip install "wiola[train]"
    python examples/train_tinystories.py \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --output-dir ./wiola-nano-tinystories \
        --max-steps 20000

Notes:
* You need access to a SentencePiece/BPE tokenizer with a 32k vocab. The LLaMA
  tokenizer is a good match; any `AutoTokenizer` works as long as you set
  `--vocab-size` to match, or let the script read it from the tokenizer.
* On an RTX 3090 the Nano config trains ~2h/epoch at seq-len 512.
"""
import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from wiola13m import WiolaConfig, WiolaForCausalLM


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True, help="AutoTokenizer id/path (e.g. a LLaMA tokenizer).")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--output-dir", default="./wiola-nano-tinystories")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--variant", choices=["nano", "micro", "small"], default="nano")
    p.add_argument("--per-device-batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    return p


VARIANTS = {
    "nano": dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, intermediate_size=512),
    "micro": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=12, intermediate_size=768),
    "small": dict(hidden_size=512, num_hidden_layers=12, num_attention_heads=16, intermediate_size=1024),
}


def main() -> None:
    args = build_argparser().parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = WiolaConfig(
        vocab_size=len(tokenizer),
        max_position_embeddings=args.seq_len,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2,
        pad_token_id=tokenizer.pad_token_id,
        **VARIANTS[args.variant],
    )
    model = WiolaForCausalLM(config)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] Wiola {args.variant}: {n_params/1e6:.1f}M parameters, vocab {config.vocab_size}")

    raw = load_dataset(args.dataset)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.seq_len)

    tokenized = raw.map(
        tokenize, batched=True, num_proc=args.num_proc, remove_columns=raw["train"].column_names
    )

    def group(examples):
        concat = sum(examples["input_ids"], [])
        total = (len(concat) // args.seq_len) * args.seq_len
        chunks = [concat[i : i + args.seq_len] for i in range(0, total, args.seq_len)]
        return {"input_ids": chunks, "attention_mask": [[1] * args.seq_len for _ in chunks]}

    lm_dataset = tokenized.map(group, batched=True, num_proc=args.num_proc)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=args.bf16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset.get("validation"),
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] Saved model + tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
