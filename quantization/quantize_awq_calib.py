import os
import argparse

from pathlib import Path

import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier
from llmcompressor.utils import dispatch_for_generation


# ======================
# Configuration
# ======================

BASE_SAVE_DIR = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized"

NUM_CALIBRATION_SAMPLES = 512
MAX_SEQUENCE_LENGTH = 1024
SEED = 42

# Just for naming the output dir
DATASET_TAGS = ["arc", "gsm8k", "math"]


# ======================
# Model / Tokenizer
# ======================

def load_model_and_tokenizer(model_id: str, device: str = "cuda"):
    """Load the base model and tokenizer."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        trust_remote_code=True,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )

    return model, tokenizer


# ======================
# Dataset loaders
# ======================

def load_commonsenseqa(num_samples: int):
    """
    Load CommonsenseQA as a calibration dataset.
    We keep the 'question' field for later use.
    """
    ds = load_dataset(
        "tau/commonsense_qa",
        "default",
        split=f"train[:{num_samples}]",
    )
    # CommonsenseQA already has a "question" field.
    return ds

def load_c4(num_samples: int):
    ds = load_dataset(
        "allenai/c4",
        "en",
        split="train",
        streaming=True,
    )
    ds = ds.take(num_samples)
    return ds

def load_arc(num_samples: int, seed: int = SEED):
    ds = load_dataset(
        "allenai/ai2_arc",
        "ARC-Easy",
        split=f"train[:{num_samples}]"
    ).shuffle(seed=seed)
    
    return ds

def load_gsm8k(num_samples: int, seed: int = SEED):
    """Load GSM8K calibration subset with 'question' field."""
    ds = load_dataset(
        "openai/gsm8k",
        "main",
        split=f"train[:{num_samples}]",
    ).shuffle(seed=seed)
    # GSM8K already has "question"
    return ds


def load_math(num_samples: int, seed: int = SEED):
    """
    Load Hendrycks MATH across categories, merge,
    and expose a 'question' field.
    """
    categories = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    math_datasets = []
    for cat in categories:
        ds = load_dataset("EleutherAI/hendrycks_math", cat, split="train")
        math_datasets.append(ds)

    full_math = concatenate_datasets(math_datasets)
    full_math = full_math.shuffle(seed=seed).select(range(num_samples))

    # Rename "problem" -> "question" for consistency
    full_math = full_math.rename_column("problem", "question")
    return full_math


def build_calibration_dataset(tokenizer, num_calibration_samples: int):
    """
    Build the combined calibration dataset:
      ARC + GSM8K + MATH
    All examples will have a 'text' field with a chat-formatted prompt.
    """
    third = num_calibration_samples // 3

    arc_ds = load_arc(third)
    gsm8k_ds = load_gsm8k(third)
    math_ds = load_math(third)

    combined = concatenate_datasets([arc_ds, gsm8k_ds, math_ds])

    def preprocess(example):
        return {
            "text": tokenizer.apply_chat_template(
                [{"role": "user", "content": example["question"]}],
                tokenize=False,
            )
        }

    combined = combined.map(preprocess)
    return combined


# ======================
# Quantization
# ======================

def build_save_dir(base_dir: str, model_id: str, dataset_tags):
    """Construct a descriptive save directory path."""
    model_name = model_id.rstrip("/").split("/")[-1]
    tag_str = "-".join(dataset_tags)
    save_dir = Path(base_dir) / f"{model_name}-awq-{tag_str}"
    os.makedirs(save_dir, exist_ok=True)
    return str(save_dir)


def run_awq_quantization(
    model,
    dataset,
    save_dir: str,
    max_seq_length: int,
    num_calibration_samples: int,
):
    """Run the AWQ oneshot quantization process."""
    recipe = [
        AWQModifier(
            ignore=["lm_head"],
            scheme="W4A16",
            targets=["Linear"],
        ),
    ]

    oneshot(
        model=model,
        dataset=dataset,
        recipe=recipe,
        max_seq_length=max_seq_length,
        num_calibration_samples=num_calibration_samples,
        output_dir=save_dir,
        save_compressed=False
    )


# ======================
# Sanity check generation
# ======================

def sample_generation(model, tokenizer, prompt: str = "Hello my name is", max_new_tokens: int = 100):
    """Do a simple generation with the quantized model to sanity check behavior."""
    print("\n\n========== SAMPLE GENERATION ==============")

    # Patch for autoregressive generation
    dispatch_for_generation(model)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens)

    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("==========================================\n\n")


# ======================
# Main
# ======================

def main():
    # 0. Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    args = ap.parse_args()
    MODEL_ID = args.model
    
    # 1. Load model & tokenizer
    model, tokenizer = load_model_and_tokenizer(args.model, device="cuda")

    # 2. Build calibration dataset
    calibration_dataset = build_calibration_dataset(tokenizer, NUM_CALIBRATION_SAMPLES)

    # 3. Build save dir
    save_dir = build_save_dir(BASE_SAVE_DIR, MODEL_ID, DATASET_TAGS)
    print(f"Saving quantized model to: {save_dir}")

    # 4. Run quantization
    run_awq_quantization(
        model=model,
        dataset=calibration_dataset,
        save_dir=save_dir,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # 5. Sanity check generation
    sample_generation(model, tokenizer)

    # 6. Save tokenizer
    tokenizer.save_pretrained(save_dir)
    print(f"Tokenizer saved to: {save_dir}")


if __name__ == "__main__":
    main()
