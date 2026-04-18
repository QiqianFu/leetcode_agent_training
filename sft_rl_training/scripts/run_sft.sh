#!/bin/bash
# SFT Training for LeetCode Agent - Qwen3-1.7B with LoRA
set -e

echo "=== LeetCode Agent SFT Training ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "GPUs: $(nvidia-smi -L 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"

BASE_DIR="/home/qiqianf2/LC-Agent/sft_rl_training"
SCRIPT_DIR="${BASE_DIR}/scripts"
DATA_DIR="${BASE_DIR}/data"
OUTPUT_DIR="${BASE_DIR}/output/sft_qwen3_1.7b_lora"

# Step 1: Generate training data (if not already generated)
if [ ! -f "${DATA_DIR}/sft_train.jsonl" ]; then
    echo ""
    echo ">>> Step 1: Generating training data..."
    conda run -n qwen_RL --no-capture-output \
        python3 "${DATA_DIR}/generate_sft_data.py" \
        --n_samples 1000 \
        --seed 42 \
        --output "${DATA_DIR}/sft_train.jsonl"
else
    echo ">>> Step 1: Training data already exists, skipping generation."
fi

# Step 2: Run SFT training
echo ""
echo ">>> Step 2: Starting SFT training..."

# Detect GPUs via CUDA_VISIBLE_DEVICES (condor sets this)
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
echo "Using ${NUM_GPUS} GPUs"

if [ "${NUM_GPUS}" -gt 1 ]; then
    echo "Using accelerate for multi-GPU training"
    conda run -n qwen_RL --no-capture-output \
        accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/train_sft.py" \
        --model_name_or_path "Qwen/Qwen3-1.7B" \
        --cache_dir "/shared/rsaas/qiqianf2/hf_models" \
        --data_path "${DATA_DIR}/sft_train.jsonl" \
        --max_seq_length 2048 \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 128 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --bf16 True \
        --logging_steps 10 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 2 \
        --dataloader_num_workers 4 \
        --report_to none \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False
else
    echo "Using single GPU training"
    conda run -n qwen_RL --no-capture-output \
        python3 "${SCRIPT_DIR}/train_sft.py" \
        --model_name_or_path "Qwen/Qwen3-1.7B" \
        --cache_dir "/shared/rsaas/qiqianf2/hf_models" \
        --data_path "${DATA_DIR}/sft_train.jsonl" \
        --max_seq_length 2048 \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 128 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --bf16 True \
        --logging_steps 10 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 2 \
        --dataloader_num_workers 4 \
        --report_to none \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False
fi

echo ""
echo "=== SFT Training finished ==="
echo "Output: ${OUTPUT_DIR}"
echo "End: $(date)"
