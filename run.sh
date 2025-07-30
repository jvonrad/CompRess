#!/bin/bash
#SBATCH -J distill_llama2_3b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1           # ein Task (Prozess) pro GPU
#SBATCH --cpus-per-task=8             # pro Prozess
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:1                  # vier GPUs
#SBATCH --mem=80G                   # Gesamt-RAM (optional anpassen)
#SBATCH --time=0-20:00:00             # z.B. 2 Tage
#SBATCH --output=logs/distill.%j.out
#SBATCH --error=logs/distill.%j.err
#SBATCH --mail-user=your-email@gmail.com
#SBATCH --mail-type=END




# Fail-Fast / Debug
echo "Starte Job $SLURM_JOB_ID am $(date)"

# -------- Conda & HF setup --------
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate thesis

export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HOME/.cache/huggingface/datasets"
CODE_DIR="/home/geiger/username/Jonathans_Thesis/LLMCBench"
MODEL_DIR="/home/geiger/username/LLMs"
LLAMA_2_7B="$MODEL_DIR/llama-2/llama-2-7b-hf"
LLAMA_2_13B="$MODEL_DIR/llama-2/llama-2-13b-hf"
LLAMA_2_70B="$MODEL_DIR/llama-2/llama-2-70b-hf"
LLAMA_3_1B="$MODEL_DIR/llama-3/Llama-3.2-1B"
LLAMA_3_8B="/home/geiger/username/LLMs/llama-3/Meta-Llama-3-8B"
LLAMA_3_8B_MINITRON="/home/geiger/username/LLMs/llama-3/Llama-3.1-Minitron-4B-Depth-Base"
LLAMA_3_8B_IT="/home/geiger/username/LLMs/llama-3/Meta-Llama-3-8B-Instruct"
LLAMA_3_8B_FT="/home/geiger/username/LLMs/llama-3/llama_3_1_8b_pile_finetuned_100m"  # Fine-tuned Llama-3-8B
LLAMA_3_70B="$MODEL_DIR/llama-3/Llama-3.1-70B"
QWEN_3_0_6B_BASE="/home/geiger/username/LLMs/Qwen/Qwen3-0.6B-Base"
QWEN_3_1_7B_BASE="$MODEL_DIR/Qwen/Qwen3-1.7B-Base"
QWEN_3_4B_BASE="$MODEL_DIR/Qwen/Qwen3-4B-Base"
QWEN_3_8B_BASE="/home/geiger/username/LLMs/Qwen/Qwen3-8B-Base"
QWEN_3_14B_BASE="$MODEL_DIR/Qwen/Qwen3-14B-Base"
QWEN_3_32B="$MODEL_DIR/Qwen/Qwen3-32B"
QWEN_3_0_6B_IT="$MODEL_DIR/Qwen/Qwen3-0.6B"
QWEN_3_1_7B_IT="$MODEL_DIR/Qwen/Qwen3-1.7B"
QWEN_3_4B_IT="$MODEL_DIR/Qwen/Qwen3-4B"
QWEN_3_8B_IT="$MODEL_DIR/Qwen/Qwen3-8B"
QWEN_3_14B_IT="$MODEL_DIR/Qwen/Qwen3-14B"
QWEN_QWQ="$MODEL_DIR/Qwen/Qwen-QwQ-32B"
QWEN_3_30B_A3B="$MODEL_DIR/Qwen/Qwen3-30B-A3B-Base"
DEEPSEEK_MOE_16B_BASE="$MODEL_DIR/deepseek/deepseek-moe-16b-base"
DEEPSEEK_R1_LLAMA_3_8B="$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-8B"
DEEPSEEK_R1_LLAMA_3_70B="$MODEL_DIR/deepseek/DeepSeek-R1-Distill-Llama-70B"

CHECKPOINT="/home/geiger/username/Jonathans_Thesis/compressed-models/distilled/ds-r1-llama-3-8b-distill-on-wiki25pct_exponential_alpha0.25/checkpoint-24977"  
#/home/geiger/username/Jonathans_Thesis/compressed-models/pruned/qwen-3-8b-SparseGPT-0.5

####################### EVALUATION ########################
# srun --tasks=1  --cpus-per-task=8 --nodes=1        --partition=h100-ferranti  --time=0-03:35     --gres=gpu:1    --mem=80G  --pty bash
# srun python -u $CODE_DIR/evaluate_math.py	

# srun python $CODE_DIR/finetune_wiki2.py

# # # Run WIKI evaluator
srun python -u $CODE_DIR/evaluate_wiki2.py \
     --path  "$CHECKPOINT" \
     --batch_size 1 \
     --max_len 4096  

# python investigate_layer_importance.py \
#   --dtype bfloat16 \
#   --device cuda


# python -u $CODE_DIR/Track_4/evaluate_flops.py --path $CHECKPOINT

# srun python -u $CODE_DIR/Track_6/evaluate_tQA.py \
#      --path  $CHECKPOINT 

# python -u $CODE_DIR/Track_6/evaluate_advglue.py --path  $CHECKPOINT --ntrain 0

# ############### GTPQ QUANTIZATION ########################

# python $CODE_DIR/quantization/gptq.py \
#   --model_path $QWEN_3_8B_BASE \
#   --output_dir "/home/geiger/username/Jonathans_Thesis/compressed-models/quantized/qwen3_8b_gptq4bit" \
#   --bits 4 \
#   --wandb_project      llama-distillation \
#   --wandb_entity       jonathan-von-rad \
#   --wandb_run_name     "qwen_gptq_memory_test" 

# python $CODE_DIR/quantization/quantize_awq.py

# ######################## PRUNING #########################
export WANDB_MODE=disabled

# srun python -u  $CODE_DIR/pruning/wanda/main.py \
#   --model "$LLAMA_3_8B" \
#   --prune_method sparsegpt \
#   --sparsity_ratio 0.5 \
#   --sparsity_type unstructured \
#   --save /home/geiger/username/Jonathans_Thesis/compressed-models/pruned/llama-3-8b-sparsegpt-0.5_mixed/logs \
#   --save_model /home/geiger/username/Jonathans_Thesis/compressed-models/pruned/llama-3-8b-sparsegpt-0.5_mixed \
#   --wandb_run_name "llama-3-8b-it-sparsegpt-0.5_test_gpu??" \
#   --calib_dataset mixed_reasoning


# ######################## LM_EVAL_HARNESS #########################


# lm_eval --model hf \
#     --model_args "pretrained=$CHECKPOINT,device_map=auto" \
#     --tasks gsm8k_cot \
#     --num_fewshot 8 \
#     --batch_size auto

# lm_eval \
#   --model hf \
#   --model_args "pretrained=$QWEN_3_32B,dtype=bfloat16,device_map=auto" \
#   --tasks gsm8k_cot \
#   --num_fewshot 4 \
#   --batch_size auto \
#   --fewshot_as_multiturn \
#   --apply_chat_template

# lm_eval --model hf \
#     --model_args "pretrained=$CHECKPOINT,device_map=auto,dtype=bfloat16" \
#     --tasks mmlu,arc_challenge,arc_easy,hellaswag,piqa,winogrande \
#     --batch_size auto

  




# ##################### AWQ QUANTIZATION ########################

# QUANTIZE_MODEL="$QWEN_3_8B_BASE"  # Change this to the model you want to quantize
# WBIT=4
# QGROUP_SIZE=128
# SAVE_PATH="quant_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE-awq"


# cd $CODE_DIR/quantization/llm-awq/awq
# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --run_awq --dump_awq "awq_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE.pt"

# echo -e "\n AWQ quantization completed for $QUANTIZE_MODEL model with $WBIT bits and $QGROUP_SIZE group size.\n"

# mkdir quant_cache
# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --load_awq "awq_cache/qwen-3-8b-base-w$WBIT-g$QGROUP_SIZE.pt" \
#     --q_backend real --dump_quant "$SAVE_PATH"

# echo -e "\n AWQ quantization cache created and model_state_dict saved at $SAVE_PATH. \n"

# python -m awq.entry --model_path "$QUANTIZE_MODEL" \
#     --tasks gsm8k \
#     --w_bit "$WBIT" --q_group_size "$QGROUP_SIZE" \
#     --load_quant "$SAVE_PATH"  

# echo -e "\n AWQ quantization tasks completed \n"

# rom 2

# vllm bench throughput   --model "$CHECKPOINT"   --backend vllm   --dataset-name random   --input-len 32 --output-len 128   --num-prompts 1000   --max-num-batched-tokens 32768   --dtype auto    --seed 42   --output-json llama3-8b_w4a16.json






#################### Teacher Correction ######################

# python $CODE_DIR/distillation/teacher_correction.py \
#   --checkpoint $LLAMA_3_8B \
#   --out_dir    ./llama_3_1_8b_pile_finetuned_1B \
#   --batch_size 2 \
#   --acc_steps  8


############### Distillation #######################


# ─── Edit only these three ────────────────────────────────────────────

# LR=1e-4
# ALPHA=1
# SEQ_LEN=1024
# LAYER_RATIO=0.5
# WIKI_PCT=25
# # # # # # # ─────────────────────────────────────────────────────────────────────



# # # # # # # Der Rest passt sich automatisch an
# RUN_NAME="llama-3-8b-distill-on-wiki${WIKI_PCT}pct_exponential_alpha${ALPHA}"
# OUTPUT_DIR="/home/geiger/username/Jonathans_Thesis/compressed-models/distilled/${RUN_NAME}"

# echo "=== Starte Distillation von LLAMA-3-8B ==="
# echo "  Teacher:   LLAMA-3-8B"
# echo "  Output:    ${OUTPUT_DIR}"
# echo "  Run name:  ${RUN_NAME}"
# echo "=================================================="
# echo

# srun torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --master_port=29500 \
#   /home/geiger/username/Jonathans_Thesis/LLMCBench/distillation/distill_llama_student.py \
#     --teacher_path       "$LLAMA_3_8B" \
#     --output_dir         "${OUTPUT_DIR}" \
#     --layer_ratio       "${LAYER_RATIO}" \
#     --dataset_name       wikipedia \
#     --dataset_config     20220301.en \
#     --wiki_pct           "${WIKI_PCT}" \
#     --warmup_steps       40 \
#     --max_seq_length     "${SEQ_LEN}" \
#     --per_device_train_batch_size 4 \
#     --per_device_eval_batch_size 1 \
#     --gradient_accumulation_steps 8\
#     --learning_rate      "${LR}" \
#     --num_train_epochs   3 \
#     --temperature        2.0 \
#     --alpha_distill      "${ALPHA}" \
#     --wandb_project      llama-distillation \
#     --wandb_entity       jonathan-von-rad \
#     --wandb_run_name     "${RUN_NAME}" 





########


# torchrun --nproc_per_node=2 finetune.py \
#   --base-path        $BASE_PATH \
#   --model-path       /home/geiger/username/LLMs/llama-3/llama-3.2-1b-hf \
#   --teacher-model-path /home/geiger/username/LLMs/llama-3/llama-3.1-8b-hf \
#   --ckpt-name        llama3-1B \
#   --teacher-ckpt-name llama3-8B \
#   --model-type       llama \
#   --teacher-model-fp16 \
#   --model-parallel --model-parallel-size 4 \
#   --gradient-checkpointing \
#   --data-dir         $BASE_PATH/processed_data/dolly/full/llama3 \
#   --batch-size       1 \
#   --eval-batch-size  8 \
#   --gradient-accumulation-steps 16 \
#   --lr               1e-5 \
#   --epochs           3 \
#   --max-length       512 \
#   --max-prompt-length 256 \
#   --save             $BASE_PATH/results/llama3/train/minillm \
#   --log-interval     10 \
#   --eval-interval    200 \
#   --deepspeed        --deepspeed_config $BASE_PATH/configs/deepspeed/ds_config_zero2_fp16.json
