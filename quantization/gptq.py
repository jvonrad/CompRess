import argparse
import os
import time
import tracemalloc

import psutil
import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig


def quantize(model_path: str, output_dir: str, bits: int = 4, dataset: str = "c4"):
    """Load a HF model, quantize it with GPTQ and track runtime + memory.

    Metrics are logged to Weights & Biases and printed to stdout.
    """

    # ──⏱️  Start runtime + memory tracking ─────────────────────────────────────
    start_time = time.perf_counter()
    tracemalloc.start()  # Python‑level allocations (CPU)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # ───────────────────────────────────────────────────────────────────────────

    # Tokenizer + quantization config
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    gptq_config = GPTQConfig(bits=bits, dataset=dataset, tokenizer=tokenizer)

    max_memory = {
        0: "78GiB",  # GPU 0 (adjust to your GPU)
        "cpu": "200GiB",  # or however much your host allows
    }

    # Load & quantize the model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=gptq_config,
        max_memory=max_memory,
        trust_remote_code=True,
    )
    
    

    # Move to CPU and save
    model.to("cpu")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # ──📊  Collect metrics ────────────────────────────────────────────────────
    
    duration_s = time.perf_counter() - start_time
    print(f"Quantization took {duration_s:.2f} seconds")
    # Python allocations (peak)
    _current, peak_py = tracemalloc.get_traced_memory()  # bytes
    peak_py_mb = peak_py / (1024 ** 2)

    # RSS of current process (more realistic CPU usage)
    rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)

    # GPU peak (if any)
    peak_gpu_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
    )

    metrics = {
        "quant_duration_s": duration_s,
        "peak_gpu_mem_mb": peak_gpu_mb,
    }

    # Log to W&B and stdout
    wandb.log(metrics)
    print(
        f"Quantized {model_path} → {output_dir}\n"
        f"Duration: {duration_s:.2f}s | "
        f"Peak GPU Mem: {peak_gpu_mb:.1f} MB"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="llama-2-7b-hf", help="Model repo or local path")
    parser.add_argument(
        "--output_dir",
        default="/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/llama_2_7b_gptq4bit",
        help="Directory to save the quantized model",
    )
    parser.add_argument("--bits", type=int, default=4, help="Number of quantization bits")

    # W&B settings
    parser.add_argument("--wandb_project", default="llama-distillation", help="W&B project name")
    parser.add_argument("--wandb_entity", default="jonathan-von-rad", help="W&B entity (user/team)")
    parser.add_argument("--wandb_run_name", default=None, help="Optional custom run name")
    args = parser.parse_args()

    # Initialise W&B
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "quantize_gptq",
        config=vars(args),
    )

    # Run quantisation
    quantize(args.model_path, args.output_dir, bits=args.bits)

    
    # Finish W&B run explicitly (good practice for long jobs)
    wandb.finish()
