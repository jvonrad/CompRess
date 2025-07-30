from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
import torch
import time
import wandb

wandb.init(
    project="llama-distillation",
    entity="jonathan-von-rad",
    name="quantize_qwen3_awq"
)
model_id = "/home/geiger/gwb082/LLMs/Qwen/Qwen3-8B-Base"
out_dir  = "/home/geiger/gwb082/Jonathans_Thesis/compressed-models/quantized/qwen-3-8b-awq4_test"
device = torch.cuda.current_device()

# load model in full precision first
model = AutoAWQForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    safetensors=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

# ► ONE config dict ◄ – include every AWQ option you want
quant_cfg = {
    "w_bit": 4,             # weight bits
    "q_group_size": 128,    # group size
    "zero_point": True      # Symmetric / asymmetric
    # "fuse_qkv": True,     # add these if you need kernel fusion
    # "fuse_mlp": True,
}
start_time = time.perf_counter()
torch.cuda.reset_peak_memory_stats(device)

# run calibration + quantisation
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_cfg        # ← pass the dict **once**
)

quantization_time = time.perf_counter() - start_time
max_mem_bytes = torch.cuda.max_memory_allocated(device)
print(f"quantization took {quantization_time:.2f} seconds")
print(f"max memory usage: {max_mem_bytes / 1024**3:.2f} GB")
wandb.log({
    "quantization_time_sec": quantization_time,
    "gpu_max_mem_GB": max_mem_bytes / 1024**3,
})
wandb.finish()

# save the 4-bit model
model.save_quantized(out_dir)
tokenizer.save_pretrained(out_dir)

print("✓ AWQ-quantised model written to", out_dir)
