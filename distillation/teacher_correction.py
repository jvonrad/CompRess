#!/usr/bin/env python3
"""
finetune_pile_10M.py  –  Fine-tune Llama-3-8B for 3 Epochen auf exakt
100 000 000 Tokens aus *The Pile-Uncopyrighted* (4096-Token-Blöcke).

‣ 1 Stream-Pass: liest Tokens, stoppt bei 10 M
‣ 90 % → Training, 10 % → Validation
‣ BF16, Gradient-Checkpointing
"""

import argparse, torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer, LlamaForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments
)
import wandb
import time
from transformers.integrations import WandbCallback

# ────────── Konstante Parameter ────────────────────────────────────────────
CACHE_DIR     = "/home/geiger/gwb082/.cache/huggingface/datasets"
DATASET_NAME  = "monology/pile-uncopyrighted"
TOKEN_TARGET  = 1_000_000_000
BLOCK_SIZE    = 4096
TRAIN_SHARE   = 0.90                 # 90 % train, 10 % val

# ────────── Datensatz-Builder (Streaming) ──────────────────────────────────
def build_pile_splits(tokenizer) -> DatasetDict:
    stream = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )

    buf, tok_cnt = [], 0
    for ex in stream:
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]
        if tok_cnt + len(ids) >= TOKEN_TARGET:
            ids = ids[: TOKEN_TARGET - tok_cnt]
            buf.extend(ids)
            break
        buf.extend(ids)
        tok_cnt += len(ids)

    # Aufteilen: 90 % / 10 %
    split_idx = int(TRAIN_SHARE * len(buf))
    train_buf, val_buf = buf[:split_idx], buf[split_idx:]

    def chunk(sub_buf):
        usable = (len(sub_buf) // BLOCK_SIZE) * BLOCK_SIZE
        for i in range(0, usable, BLOCK_SIZE):
            yield {"input_ids": sub_buf[i : i + BLOCK_SIZE]}

    return DatasetDict({
        "train":      Dataset.from_generator(lambda: chunk(train_buf)),
        "validation": Dataset.from_generator(lambda: chunk(val_buf)),
    })

# ────────── Hauptprogramm ─────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="Pfad oder HF-Tag des Start-Checkpoints (Llama-3-8B).")
    ap.add_argument("--out_dir",    default="./ft_pile10M_seq4096",
                    help="Zielverzeichnis für feingetunten Checkpoint.")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--acc_steps",  type=int, default=8)
    args = ap.parse_args()

    torch.cuda.empty_cache()
    
    wandb.init(
        project="llama-distillation",
        name="ft_pile100M_seq4096",
        entity="jonathan-von-rad",
        config={
            "checkpoint": args.checkpoint,
            "out_dir":    args.out_dir,
            "batch_size": args.batch_size,
            "acc_steps":  args.acc_steps,
        }
    )
    
        # ──⏱️  Start runtime + memory tracking ─────────────────────────────────────
    start_time = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Tokenizer -------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    # Modell ----------------------------------------------------------------
    model = LlamaForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.gradient_checkpointing_enable()

    # Dataset ---------------------------------------------------------------
    splits    = build_pile_splits(tokenizer)
    train_ds  = splits["train"]
    val_ds    = splits["validation"]

    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # TrainingArgs ----------------------------------------------------------
    targs = TrainingArguments(
        output_dir                 = args.out_dir,
        overwrite_output_dir       = True,
        num_train_epochs           = 3,
        per_device_train_batch_size= args.batch_size,
        gradient_accumulation_steps= args.acc_steps,
        learning_rate              = 1e-6,
        warmup_steps               = 50,
        logging_steps              = 50,
        save_strategy            = "no",
        eval_strategy        = "steps",
        eval_steps            = 100,
        per_device_eval_batch_size = 1,
        bf16                       = True,
        report_to                  = "wandb",
    )

    # Trainer ---------------------------------------------------------------
    trainer = Trainer(
        model         = model,
        args          = targs,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        data_collator = collator,
        tokenizer     = tokenizer,
        callbacks=[WandbCallback]
    )

    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    
    wandb.finish()
