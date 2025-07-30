#!/usr/bin/env python
# investigate_layer_removal_qwen8b.py
"""
Reproduces Fig. 9 for Qwen-3-8B on Winogrande (5-shot) with lm_eval.

Usage
-----
python investigate_layer_removal_qwen8b.py \
    --model_name_or_path Qwen/Qwen1.5-8B \
    --output_dir results_qwen8b_layers \
    --device cuda \
    --dtype float16

Requires
--------
pip install transformers accelerate lm_eval matplotlib rich
"""

import argparse, copy, json, os, time, warnings
from pathlib import Path
from typing import List

import torch, transformers
from lm_eval import evaluator
import matplotlib.pyplot as plt
from rich.progress import Progress, TimeElapsedColumn, BarColumn, TextColumn
from lm_eval.models.huggingface import HFLM

# ----------------------------- helper --------------------------------- #

def prune_contiguous_layers(
    model: transformers.PreTrainedModel,
    start: int,
    length: int,
    keep_last: int = 0,
) -> transformers.PreTrainedModel:
    """
    Return a *new* model where layers [start, start+length) are removed.
    For Qwen (GPT-style) the layers live under model.layers[...].
    """
    cfg = copy.deepcopy(model.config)
    total = cfg.num_hidden_layers
    keep = [i for i in range(total) if i < start or i >= start + length]
    if keep_last:
        # ensure we never drop the very top layers if requested
        for i in range(total - keep_last, total):
            if i not in keep:
                keep.append(i)
    keep.sort()

    cfg.num_hidden_layers = len(keep)
    new_model = transformers.AutoModelForCausalLM.from_config(cfg)
    # copy parameters
    with torch.no_grad():
        for new_idx, old_idx in enumerate(keep):
            new_model.model.layers[new_idx].load_state_dict(
                model.model.layers[old_idx].state_dict()
            )
        # embeddings & lm_head
        new_model.get_input_embeddings().load_state_dict(
            model.get_input_embeddings().state_dict()
        )
        # Output-Embeddings (LM-Head), falls vorhanden
        if new_model.get_output_embeddings() is not None:
            new_model.get_output_embeddings().load_state_dict(
                model.get_output_embeddings().state_dict()
            )

    return new_model


def run_lm_eval(model, tokenizer, device="cuda"):
    """Return Winogrande-5shot accuracy via lm_eval (v0.4+)."""
    # move HF model to device & dtype first
    model = model.to(device)
    # wrap into HFLM so lm_eval can call it
    lm = HFLM(pretrained=model, tokenizer=tokenizer)
    # Pass tokenizer via model_args per new API:
    results = evaluator.simple_evaluate(
	lm,
       	None,
        tasks=["mmlu"],
        num_fewshot=4,
        batch_size=1,
        device=device,
        cache_requests=False,
        rewrite_requests_cache=False,
        delete_requests_cache=False,
    )
    task_res = results["results"]["mmlu"]
    # inspect available keys
    # print("Winogrande result keys:", task_res.keys())

    # pick the first numeric entry
    for k, v in task_res.items():
        if isinstance(v, (int, float)):
            return v
    raise KeyError(f"No numeric metric found in Winogrande results: {task_res}")

# ----------------------------- main ------------------------------------ #

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", default="/home/geiger/gwb082/LLMs/llama-3/Meta-Llama-3-8B")
    parser.add_argument("--output_dir", default="layer_drop_results_llama3_8b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--layers_to_drop", type=int, default=16)
    parser.add_argument("--max_layers_to_test", type=int, default=None,
                        help="cut sweeps short for quick debugging")
    parser.add_argument("--evaluate_best_noncontiguous", action="store_true",
                        help="run a quick saliency-based pick for 16 non-contiguous layers")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # --- load baseline ---
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path, torch_dtype=dtype_map[args.dtype], device_map="auto"
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    total_layers = model.config.num_hidden_layers
    L = args.layers_to_drop
    sweep_starts = list(range(0, total_layers - L + 1))
    if args.max_layers_to_test:
        sweep_starts = sweep_starts[: args.max_layers_to_test]

    records = []

    print(f"Running baseline ({total_layers} layers)…")
    baseline_acc = run_lm_eval(model, tokenizer, args.device)
    records.append({"label": "baseline", "start": -1, "acc": baseline_acc})
    print(f"  baseline accuracy = {baseline_acc:.4f}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(), transient=True
    ) as progress:
        task = progress.add_task("[cyan]dropping contiguous blocks", total=len(sweep_starts))
        for s in sweep_starts:
            tag = f"drop_{s}_{s+L-1}"
            progress.update(task, description=f"[cyan]{tag}")
            student = prune_contiguous_layers(model, s, L)
            acc = run_lm_eval(student, tokenizer, args.device)
            records.append({"label": tag, "start": s, "acc": acc})
            progress.advance(task)

    if args.evaluate_best_noncontiguous:
        # naive saliency: drop every 2nd layer, keep first+last
        noncontig = list(range(1, total_layers - 1, total_layers // L))[:L]
        mask = [i for i in range(total_layers) if i not in noncontig]
        cfg = copy.deepcopy(model.config)
        cfg.num_hidden_layers = len(mask)
        non_model = transformers.AutoModelForCausalLM.from_config(cfg)
        with torch.no_grad():
            for new_idx, old_idx in enumerate(mask):
                non_model.model.layers[new_idx].load_state_dict(model.model.layers[old_idx].state_dict())
                non_model.get_input_embeddings().load_state_dict(
                    model.get_input_embeddings().state_dict()
                )
                if non_model.get_output_embeddings() is not None:
                    non_model.get_output_embeddings().load_state_dict(
                        model.get_output_embeddings().state_dict()
                    )
        acc = run_lm_eval(non_model, tokenizer, args.device)
        records.append({"label": "noncontiguous_best_guess", "start": -2, "acc": acc})

    # --- save CSV ---
    csv_path = Path(args.output_dir) / "layer_drop_results.csv"
    with open(csv_path, "w") as f:
        f.write("label,start,acc\n")
        for r in records:
            f.write(f"{r['label']},{r['start']},{r['acc']}\n")
    print(f"Saved CSV → {csv_path}")

    # --- plot ---
    xs, ys = [], []
    for r in records:
        if r["start"] >= 0:
            xs.append(r["start"] + 1)          # layer numbering 1…32
            ys.append(r["acc"])
    plt.figure(figsize=(5,3))
    plt.plot(xs, ys, color="mediumpurple", label=f"drop {L} layers")
    plt.axhline(baseline_acc, color="steelblue", label="baseline", linewidth=1.5)
    if args.evaluate_best_noncontiguous:
        plt.axhline(records[-1]["acc"], color="dodgerblue", linestyle="--",
                    label="drop 16 layers non-contiguous")
    plt.xlabel("layer no.")
    plt.ylabel("MMLU (5-shot acc)")
    plt.title(f"Accuracy when removing {L} contiguous layers\n(Qwen-3-8B, MMLU)")
    plt.legend()
    out_png = Path(args.output_dir) / "figure_layer_drop_qwen8b.png"
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    print(f"Plot saved → {out_png}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # transformer shape-cast warnings
    torch.backends.cuda.matmul.allow_tf32 = True
    main()
