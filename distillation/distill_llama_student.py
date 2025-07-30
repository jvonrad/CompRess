#!/usr/bin/env python
"""
distill_llama_student.py
-----------------------
Creates a student model with roughly half the parameters of a Llama‑2‑7B (~3.5 B) or Qwen3 model and
distils knowledge from the original teacher.

Example (single GPU)
--------------------
python distill_llama_student.py \
  --teacher_path meta-llama/Llama-2-7b-hf \
  --output_dir /tmp/llama2_3b_distilled

Example with Qwen3:
python distill_llama_student.py \
  --teacher_path Qwen/Qwen2.5-7B \
  --output_dir /tmp/qwen3_3b_distilled

Call‑ready for SLURM via srun / sbatch with plenty of CLI flags.
"""
import glob
import argparse
from torchao.sparsity.training import SemiSparseLinear, swap_linear_with_semi_sparse_linear
from datasets import load_from_disk
import os
import math
import torch
import time
from torch.nn import KLDivLoss
import transformers, inspect
from bitsandbytes.optim import Adam8bit
from auto_gptq import AutoGPTQForCausalLM
import torch.nn as nn
from torch.nn.utils import prune
from datasets import load_dataset, Dataset, DatasetDict
import wandb

from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTQConfig,
    Trainer,
    EvalPrediction,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from transformers.integrations import WandbCallback
import wandb
import os

print("Transformers-Version:", transformers.__version__)
print("TrainingArguments   :", inspect.getfile(TrainingArguments))

###############################################################################
# Helper functions                                                             #
###############################################################################

def get_model_architecture_info(model):
    """Detect model architecture and return relevant paths"""
    model_type = model.config.model_type
    
    if model_type == "llama":
        return {
            "layers_path": "model.layers",
            "embed_tokens_path": "model.embed_tokens",
            "lm_head_path": "lm_head",
            "norm_path": "model.norm"
        }
    elif model_type in ["qwen2", "qwen", "qwen2_moe"]:  # Qwen family
        return {
            "layers_path": "model.layers",
            "embed_tokens_path": "model.embed_tokens", 
            "lm_head_path": "lm_head",
            "norm_path": "model.norm"
        }
    else:
        # Try to auto-detect
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            return {
                "layers_path": "model.layers",
                "embed_tokens_path": "model.embed_tokens",
                "lm_head_path": "lm_head",
                "norm_path": "model.norm"
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

def get_nested_attr(obj, attr_path):
    """Get nested attribute using dot notation"""
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj

def set_nested_attr(obj, attr_path, value):
    """Set nested attribute using dot notation"""
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)

def reverse_kl_distillation_loss(student_logits,
                                 teacher_logits,
                                 temperature: float = 2.0):
    """
    D_KL(student || teacher) * T²  (MiniLLM-Style)
    """
    # 1) Softmax / Log-Softmax mit Temperatur
    s_log_probs = torch.log_softmax(student_logits / temperature, dim=-1)
    t_log_probs = torch.log_softmax(teacher_logits / temperature, dim=-1)

    # 2) Wahrscheinlichkeiten des Studenten (exp(log p))
    s_probs = s_log_probs.exp()

    # 3) Reverse KL = Σ S * (log S − log T)
    loss_per_token = torch.sum(s_probs * (s_log_probs - t_log_probs), dim=-1)

    # 4) Mittelwert über Batch & Seq-Länge
    loss = loss_per_token.mean()

    # 5) Temperatur-Scaling (wie bei Hinton KD)
    return loss * (temperature ** 2)

def build_student_from_teacher(teacher: AutoModelForCausalLM, layer_ratio: float = 0.5, keep_last: int = 1):
    """
    Create a student model that keeps:
      - the bottom `int(num_hidden_layers * layer_ratio)` layers
      - PLUS the last `keep_last` layers of the teacher
    """
    # 1) Pull the teacher's config
    teacher_cfg = teacher.config
    
    # Handle different config attributes for number of layers
    if hasattr(teacher_cfg, 'num_hidden_layers'):
        total = teacher_cfg.num_hidden_layers
    elif hasattr(teacher_cfg, 'n_layer'):
        total = teacher_cfg.n_layer
    elif hasattr(teacher_cfg, 'n_layers'):
        total = teacher_cfg.n_layers
    else:
        raise ValueError("Cannot determine number of layers from config")
    
    print(f"Teacher has {total} layers")
    bottom = max(1, int(total * layer_ratio))
    student_layers = bottom + keep_last
    print(f"Building student with {student_layers} layers (bottom: {bottom}, keep_last: {keep_last})")
    
    # 2) Create a modified config dict
    cfg_dict = teacher_cfg.to_dict()
    
    # Update the number of layers based on config type
    if 'num_hidden_layers' in cfg_dict:
        cfg_dict["num_hidden_layers"] = student_layers
    elif 'n_layer' in cfg_dict:
        cfg_dict["n_layer"] = student_layers
    elif 'n_layers' in cfg_dict:
        cfg_dict["n_layers"] = student_layers

    # 3) Instantiate the same config class from dict
    new_cfg = teacher_cfg.__class__.from_dict(cfg_dict)

    # 4) Build a fresh student model from this config
    # Use the same model class as the teacher
    model_class = teacher.__class__
    student = model_class(new_cfg)
    
    return student

def copy_layers(student: AutoModelForCausalLM, teacher: AutoModelForCausalLM, keep_last: int = 2):
    """
    Copy into the student:
      - teacher.layers[0:bottom]
      - teacher.layers[-keep_last:]
      - plus embeddings & lm_head
    """
    with torch.no_grad():
        # Get architecture info
        arch_info = get_model_architecture_info(teacher)
        
        # Get layers
        teacher_layers = get_nested_attr(teacher, arch_info["layers_path"])
        student_layers = get_nested_attr(student, arch_info["layers_path"])
        
        total = len(teacher_layers)
        bottom = len(student_layers) - keep_last
        
        print(f"Copying layers: bottom {bottom} layers + last {keep_last} layers")
        
        # 1) Copy bottom layers
        for i in range(bottom):
            print(f"  Copying layer {i} -> {i}")
            student_layers[i].load_state_dict(
                teacher_layers[i].state_dict()
            )
        
        # 2) Copy last `keep_last` layers
        for j in range(keep_last):
            teacher_idx = total - keep_last + j
            student_idx = bottom + j
            print(f"  Copying layer {teacher_idx} -> {student_idx}")
            student_layers[student_idx].load_state_dict(
                teacher_layers[teacher_idx].state_dict()
            )

        # 3) Copy embeddings
        try:
            teacher_embed = get_nested_attr(teacher, arch_info["embed_tokens_path"])
            student_embed = get_nested_attr(student, arch_info["embed_tokens_path"])
            student_embed.load_state_dict(teacher_embed.state_dict())
            print("  Copied embeddings")
        except Exception as e:
            print(f"  Warning: Could not copy embeddings: {e}")
        
        # 4) Copy lm_head
        try:
            teacher_lm_head = get_nested_attr(teacher, arch_info["lm_head_path"])
            student_lm_head = get_nested_attr(student, arch_info["lm_head_path"])
            student_lm_head.load_state_dict(teacher_lm_head.state_dict())
            print("  Copied lm_head")
        except Exception as e:
            print(f"  Warning: Could not copy lm_head: {e}")
        
        # 5) Copy final norm layer if exists
        try:
            teacher_norm = get_nested_attr(teacher, arch_info["norm_path"])
            student_norm = get_nested_attr(student, arch_info["norm_path"])
            student_norm.load_state_dict(teacher_norm.state_dict())
            print("  Copied final norm layer")
        except Exception as e:
            print(f"  Warning: Could not copy norm layer: {e}")

def distillation_loss(student_logits, teacher_logits, temperature: float = 2.0):
    s_logits = student_logits.float()
    t_logits = teacher_logits.float()
    
    # More aggressive clamping
    s_logits = torch.clamp(s_logits, min=-50, max=50)
    t_logits = torch.clamp(t_logits, min=-50, max=50)
    
    student_log_probs = torch.log_softmax(s_logits / temperature, dim=-1)
    teacher_probs = torch.softmax(t_logits / temperature, dim=-1)
    
    # Ensure teacher probs sum to 1 and add epsilon
    teacher_probs = teacher_probs / (teacher_probs.sum(dim=-1, keepdim=True) + 1e-8)
    teacher_probs = teacher_probs + 1e-8
    
    loss = KLDivLoss(reduction="batchmean")(student_log_probs, teacher_probs) * (temperature**2)
    
    # More reasonable clamp
    loss = torch.clamp(loss, max=10.0)  # Instead of 100.0
    
    return loss

###############################################################################
# KD‑Trainer                                                                   #
###############################################################################
import csv
import math
from transformers import TrainerCallback

class CombinedPPLCallback(TrainerCallback):
    def __init__(self, csv_filename="eval_ppl_by_epoch_{args.}.csv"):
        self.csv_filename = csv_filename
        # CSV-Header anlegen
        with open(self.csv_filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "step", "eval_perplexity"])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        
        loss = metrics.get("eval_loss", None)
        if loss is None:
            return

        # 1) compute once
        try:
            ppl = math.exp(loss)
        except OverflowError:
            ppl = float("inf")
        metrics["perplexity"] = ppl

        # 2) log to W&B (falls Du das weiter willst)
        import wandb
        wandb.log({"eval_perplexity": ppl}, step=state.global_step)

        # 3) print in stdout
        print(f"\n>>> Eval Perplexity at step {state.global_step}, epoch {metrics.get('epoch'):.2f}: {ppl:.2f}\n")

        # 4) append to CSV
        epoch = metrics.get("epoch", None)
        with open(self.csv_filename, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, state.global_step, ppl])
                
class DistilTrainer(Trainer):
    """HF-Trainer that linearly anneals α_distill → 1.0."""
    def __init__(
        self,
        teacher: AutoModelForCausalLM,
        reverse_kl: bool = False,
        temperature: float = 2.0,
        alpha_distill: float = 0.5,
        dynamic_alpha: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.teacher = teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.reverse_kl = reverse_kl
        self.temperature = temperature
        self.alpha_start = alpha_distill
        self.dynamic_alpha = dynamic_alpha

        # total #updates that HF Trainer will run
        self.total_steps = self.args.max_steps if self.args.max_steps > 0 else self.state.max_steps

    def _current_alpha(self) -> float:
        """Linear ramp-up from alpha_start → 1.0 over the whole training."""
        step = max(0, self.state.global_step)          # 0 … total_steps
        frac = min(1.0, step / float(self.total_steps))
        return self.alpha_start + (1.0 - self.alpha_start) * frac
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Debug first few batches
        if self.state.global_step < 2:
            print(f"\n=== Batch {self.state.global_step} ===")
            print(f"Input shape: {inputs['input_ids'].shape}")
            print(f"First 20 tokens: {inputs['input_ids'][0, :20].tolist()}")
            decoded = self.tokenizer.decode(inputs['input_ids'][0, :50], skip_special_tokens=False)
            print(f"Decoded text: {decoded[:100]}")
            
            # Check if BOS token matches
            first_token = inputs['input_ids'][0, 0].item()
            expected_bos = self.tokenizer.bos_token_id
            if first_token != expected_bos:
                print(f"WARNING: First token {first_token} doesn't match expected BOS {expected_bos}")
        
        # forward student
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # evaluation: just LM-loss
        if not model.training:
            return (lm_loss, outputs) if return_outputs else lm_loss

        # teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(**inputs)

        if self.reverse_kl:
            kd_loss = reverse_kl_distillation_loss(
                outputs.logits, teacher_out.logits, self.temperature
            )
        else:
            kd_loss = distillation_loss(
                outputs.logits, teacher_out.logits, self.temperature
            )

        if self.dynamic_alpha:
            alpha = self._current_alpha()          # <-- dynamic α
        else:
            alpha = self.alpha_start               # <-- static α
            
        loss  = alpha * kd_loss + (1 - alpha) * lm_loss

        # (optional) send α to W&B
        if self.args.report_to and "wandb" in self.args.report_to:
            wandb.log({"alpha_distill": alpha}, step=self.state.global_step)
            
        
        if self.state.global_step % 100 == 0:  # Log every 100 steps
            print(f"\nStep {self.state.global_step} losses:")
            print(f"  LM Loss: {lm_loss.item():.4f}")
            print(f"  KD Loss: {kd_loss.item():.4f}")
            print(f"  Alpha: {alpha:.4f}")
            print(f"  Combined Loss: {loss.item():.4f}")
    
        # Check for NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN loss detected at step {self.state.global_step}")
            print(f"  Student logits range: [{outputs.logits.min():.2f}, {outputs.logits.max():.2f}]")
            print(f"  Teacher logits range: [{teacher_out.logits.min():.2f}, {teacher_out.logits.max():.2f}]")
        

        return (loss, outputs) if return_outputs else loss

###############################################################################
# Dataset functions                                                            #
###############################################################################

from collections import Counter

def build_pile_splits(tokenizer,
                       DATASET_NAME: str,
                       CACHE_DIR: str,
                       TOKEN_TARGET: int,
                       TRAIN_SHARE: float = 0.9,
                       BLOCK_SIZE: int = 1024) -> DatasetDict:
    # Dataset im Streaming-Modus öffnen
    stream = load_dataset(
        DATASET_NAME,
        split="train",
        streaming=True,
        cache_dir=CACHE_DIR,
    )

    buf, tok_cnt = [], 0
    subset_token_counts = Counter()

    # 1) Sammle exakt TOKEN_TARGET Tokens, zähle pro Subset
    for ex in stream:
        # Subset-Name herausfinden
        subset = (
            ex.get("pile_set_name")
            or ex.get("pile_type")
            or (ex.get("meta") or {}).get("pile_set_name")
            or (ex.get("meta") or {}).get("pile_type")
            or "UNKNOWN"
        )

        # Tokenize und IDs holen
        ids = tokenizer(ex["text"], add_special_tokens=False)["input_ids"]

        # Wenn wir damit über das Ziel hinausschießen:
        if tok_cnt + len(ids) >= TOKEN_TARGET:
            # wie viele Tokens brauchen wir noch?
            needed = TOKEN_TARGET - tok_cnt
            buf.extend(ids[:needed])
            subset_token_counts[subset] += needed
            tok_cnt += needed
            break

        # sonst ganzen Block nehmen
        buf.extend(ids)
        tok_cnt += len(ids)
        subset_token_counts[subset] += len(ids)

    # 2) Verteilung ausgeben
    print(f"\n=== Pile-Subset-Verteilung für {tok_cnt} Tokens ===")
    for subset, count in subset_token_counts.most_common():
        pct = count / tok_cnt * 100
        print(f"{subset:30s}: {count:10d} Tokens ({pct:5.2f} %)")

    # 3) In Trainings- / Validierungs-Buffer splitten
    split_idx = int(TRAIN_SHARE * len(buf))
    train_buf, val_buf = buf[:split_idx], buf[split_idx:]

    # 4) In Blöcke der Länge BLOCK_SIZE aufteilen
    def chunk(sub_buf):
        usable = (len(sub_buf) // BLOCK_SIZE) * BLOCK_SIZE
        for i in range(0, usable, BLOCK_SIZE):
            yield {"input_ids": sub_buf[i : i + BLOCK_SIZE]}

    return DatasetDict({
        "train":      Dataset.from_generator(lambda: chunk(train_buf)),
        "validation": Dataset.from_generator(lambda: chunk(val_buf)),
    })

def get_tokenized_dataset(tokenizer,
                          raw_dataset,
                          max_seq_length: int,
                          cache_dir: str):
    if os.path.isdir(cache_dir):
        print(f"Loading tokenized dataset from {cache_dir}…")
        return load_from_disk(cache_dir)

    print("Tokenized cache not found; running tokenization…")
    # 1) split raw into train/valid
    splits = raw_dataset.train_test_split(test_size=0.005, seed=42)
    raw_train, raw_valid = splits["train"], splits["test"]

    # 2) tokenization function
    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=True,  # Don't add BOS/EOS during tokenizatio
        )

    # 3) run .map()
    num_proc = min(8, os.cpu_count())
    tok_train = raw_train.map(
        tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=num_proc,
        desc="Tokenising train"
    )
    tok_valid = raw_valid.map(
        tokenize,
        batched=True,
        remove_columns=raw_dataset.column_names,
        num_proc=max(1, num_proc // 2),
        desc="Tokenising valid"
    )

    tokenized = DatasetDict({"train": tok_train, "validation": tok_valid})
    print(f"Saving tokenized dataset to {cache_dir}…")
    tokenized.save_to_disk(cache_dir)
    return tokenized

###############################################################################
# CLI                                                                          #
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--teacher_path", default="meta-llama/Llama-2-7b-hf")
    p.add_argument("--student_path", default=None, help="Optional: pretrained student path")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="wikipedia")
    p.add_argument("--dataset_config", default="20220301.en")
    p.add_argument("--layer_ratio", type=float, default=0.5)
    p.add_argument("--local_rank", type=int, default=-1,
                   help="automatisch von torchrun gesetzt")
    p.add_argument("--dynamic_alpha", action="store_true",)
    p.add_argument("--warmup_steps", type=int, default=500, help="Warmup steps for learning rate scheduler")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Truncate examples to this length")
    p.add_argument("--per_device_train_batch_size", type=int, default=8)
    p.add_argument("--per_device_eval_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--min_learning_rate", type=float, default=1e-5)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--wiki_pct", type=float, default=2.0, help="Prozentsatz von Wikipedia für Training (z.B. 2.0 für '[:2%]')")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha_distill", type=float, default=0.5)
    p.add_argument("--reverse_kl", action="store_true", help="Use reverse KL loss instead of standard KL divergence")

    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)

    return p.parse_args()

###############################################################################
# Main                                                                         #
###############################################################################

def main():
    args = parse_args()
    start_time = time.time()
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print("=== Hyperparameter ===")
    print(f"  Teacher model path:             {args.teacher_path}")
    print(f"  Learning rate:                  {args.learning_rate}")
    print(f"  Min learning rate:              {args.min_learning_rate}")   
    print(f"  Warmup steps:                   {args.warmup_steps}")
    print(f"  Gradient accumulation steps:    {args.gradient_accumulation_steps}")
    print(f"  Per-device batch size:          {args.per_device_train_batch_size}")
    print(f"  Num epochs:                     {args.num_train_epochs}")
    print(f"  Temperature (T):                {args.temperature}")
    print(f"  Distillation α:                 {args.alpha_distill}")
    print(f"  Layer ratio:                    {args.layer_ratio}")
    print(f"  GPUs (torch.cuda.device_count): {num_gpus}")
    print(f"  Output directory:               {args.output_dir}")
    print(f"  WandB project:                  {args.wandb_project}")
    print("======================\n")
    os.makedirs(args.output_dir, exist_ok=True)

    # 0) WandB – optional
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  # von torchrun gesetzt

    if local_rank != 0:
        # wandb nur im ersten Prozess initialisieren
        os.environ["WANDB_MODE"] = "disabled"

    #settings = wandb.Settings(disable_stats=True)
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "llama2-7b_to_student",
        config=vars(args),
    )

    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    # 1) Load teacher model with trust_remote_code for Qwen models
    print(f"Loading teacher model from {args.teacher_path}...")
    
    # CRITICAL: Load the correct tokenizer for the teacher model
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True)
    
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_path,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True  # Important for Qwen3!
    ).eval()
    
    # Verify tokenizer matches model
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    print(f"Model vocab size: {teacher.config.vocab_size}")
    print(f"BOS token ID: {tokenizer.bos_token_id} (expected: {teacher.config.bos_token_id})")
    print(f"EOS token ID: {tokenizer.eos_token_id} (expected: {teacher.config.eos_token_id})")
    
    # if len(tokenizer) != teacher.config.vocab_size:
    #     raise ValueError(f"Tokenizer vocab size {len(tokenizer)} doesn't match model vocab size {teacher.config.vocab_size}!")
    
    # Debug info
    print(f"Teacher model type: {type(teacher)}")
    print(f"Teacher config type: {type(teacher.config)}")
    print(f"Teacher model architecture: {teacher.config.model_type}")
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Build & initialize custom student by layerdropping
    print("\nBuilding student model...")
    
    if args.student_path:
        # Load pretrained student if provided
        student = AutoModelForCausalLM.from_pretrained(
            args.student_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
    else:
        # Build student from teacher
        student = build_student_from_teacher(teacher, layer_ratio=args.layer_ratio, keep_last=1)
        copy_layers(student, teacher, keep_last=1)
    
    # Move student to device/dtype
    student = student.to(device, torch.bfloat16)
    
    # Enable gradient checkpointing if available
    if hasattr(student, 'gradient_checkpointing_enable'):
        student.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    else:
        print("Warning: Gradient checkpointing not available for this model")
    
    optimizer = Adam8bit(student.parameters(), lr=args.learning_rate)

    # 2) Tokenizer & Dataset
    # Ensure we're using the correct tokenizer throughout
    print(f"\nTokenizer details:")
    print(f"  Type: {type(tokenizer)}")
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"  BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    raw = load_dataset(
        args.dataset_name,
        args.dataset_config,
        split=f"train[:{int(args.wiki_pct)}%]",
        trust_remote_code=True
    ).shuffle(seed=42)
    
    # CRITICAL: Use model-specific cache dir to avoid tokenizer mismatches
    # Replace this line:
    model_name = args.teacher_path.split('/')[-1]
    cache_dir = f"/home/geiger/gwb082/Jonathans_Thesis/datasets/tokenized_{model_name}_wiki{args.wiki_pct}_seq{args.max_seq_length}"
        
    tokenized = get_tokenized_dataset(
        tokenizer,
        raw,
        max_seq_length=args.max_seq_length,
        cache_dir=cache_dir
    )
    train_tok = tokenized["train"]
    valid_tok = tokenized["validation"]
    
    # 3) Data collator
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    # 4) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        bf16=True,
        bf16_full_eval=True,
        max_grad_norm=0.5,
        ddp_find_unused_parameters=False,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        report_to="wandb" if args.wandb_project else "none",
    )

    csv_name = f"distill_results/{args.wandb_run_name}.csv" if args.wandb_run_name else "distill_results/results.csv"
    os.makedirs("distill_results", exist_ok=True)

    # 5) Trainer
    trainer = DistilTrainer(
        teacher=teacher,
        temperature=args.temperature,
        dynamic_alpha=args.dynamic_alpha,
        alpha_distill=args.alpha_distill,
        reverse_kl=args.reverse_kl,
        model=student,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=None,
        callbacks=[CombinedPPLCallback(csv_filename=csv_name)],
        optimizers=(optimizer, None),
    )

    # Train
    print("\nStarting training...")
    trainer.train()

    # 6) Save
    print(f"\nSaving model to {args.output_dir}...")
    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if args.wandb_project:
        wandb.finish()

    # Am Ende: Laufzeit ausgeben
    elapsed = time.time() - start_time
    hrs, rem = divmod(elapsed, 3600)
    mins, secs = divmod(rem, 60)
    print(f"\n=== Distillation completed in {int(hrs)}h {int(mins)}m {secs:.1f}s ===")

if __name__ == "__main__":
    main()