"""Pre-train Wiola Nano on TinyStories with the Hugging Face Trainer.

Memory‑friendly defaults for a 6 GB GPU (RTX 4050):
    per_device_batch_size=4 + grad_accum=8 → effective batch 32.
"""
import argparse
from itertools import chain

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

from wiola13m import WiolaConfig, WiolaForCausalLM


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True,
                   help="Path or HF id of a SentencePiece/BPE tokenizer.")
    p.add_argument("--dataset", default="roneneldan/TinyStories")
    p.add_argument("--output-dir", default="./wiola-nano-tinystories")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--variant", choices=["nano", "micro", "small"], default="nano")
    p.add_argument("--per-device-batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-steps", type=int, default=1000)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--num-proc", type=int, default=4)
    p.add_argument("--no-eval", action="store_true",
                   help="Skip evaluation (if no validation set).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    p.add_argument("--resume-from-checkpoint", default=None,
                   help="Resume training from a checkpoint directory.")
    return p


VARIANTS = {
    "nano": dict(hidden_size=256, num_hidden_layers=6, num_attention_heads=8, intermediate_size=512),
    "micro": dict(hidden_size=384, num_hidden_layers=8, num_attention_heads=12, intermediate_size=768),
    "small": dict(hidden_size=512, num_hidden_layers=12, num_attention_heads=16, intermediate_size=1024),
}


def main() -> None:
    args = build_argparser().parse_args()

    # ----- Reproducibility -----
    set_seed(args.seed)

    # ----- Tokenizer -----
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----- Config & Model -----
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

    # ----- Dataset -----
    raw = load_dataset(args.dataset)

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.seq_len,
        )
        enc.pop("token_type_ids", None)      # remove if present
        return enc

    tokenized = raw.map(
        tokenize, batched=True, num_proc=args.num_proc,
        remove_columns=raw["train"].column_names
    )

    def group(examples):
        all_input_ids = list(chain.from_iterable(examples["input_ids"]))
        all_attention_mask = list(chain.from_iterable(examples["attention_mask"]))
        total = (len(all_input_ids) // args.seq_len) * args.seq_len

        chunks = []
        for i in range(0, total, args.seq_len):
            chunks.append({
                "input_ids": all_input_ids[i : i + args.seq_len],
                "attention_mask": all_attention_mask[i : i + args.seq_len],
            })

        return {
            "input_ids": [c["input_ids"] for c in chunks],
            "attention_mask": [c["attention_mask"] for c in chunks],
        }

    lm_dataset = tokenized.map(
        group,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=tokenized["train"].column_names,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ----- Training arguments -----
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
        save_strategy="steps",                  # save at regular step intervals
        eval_steps=args.eval_steps if not args.no_eval else None,
        evaluation_strategy="steps" if not args.no_eval else "no",
        load_best_model_at_end=not args.no_eval,
        metric_for_best_model="loss" if not args.no_eval else None,
        save_total_limit=3,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        report_to=["tensorboard"],
        logging_dir=f"{args.output_dir}/logs",
    )

    # ----- Trainer -----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset.get("validation") if not args.no_eval else None,
        data_collator=collator,
    )

    # ----- Training (with resume support) -----
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # ----- Save final model + tokenizer + generation config -----
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    generation_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
    )
    generation_config.save_pretrained(args.output_dir)

    print(f"[done] Saved model, tokenizer, and generation config to {args.output_dir}")


if __name__ == "__main__":
    main()