#!/bin/bash
# SFT Training - Qwen3-1.7B with LoRA on REAL DeepSeek trajectories (Exp-002)
set -e

echo "=== LeetCode Agent SFT Training (Real Data) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Start: $(date)"

SCRIPT_DIR="/home/qiqianf2/LC-Agent/sft_rl_training/scripts"
DATA_PATH="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/all_trajectories.jsonl"
OUTPUT_DIR="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_real_exp002"

echo "Data: ${DATA_PATH}"
echo "Output: ${OUTPUT_DIR}"

# Detect GPUs
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
else
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
fi
echo "Using ${NUM_GPUS} GPUs"

if [ "${NUM_GPUS}" -gt 1 ]; then
    conda run -n qwen_RL --no-capture-output \
        accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/train_sft.py" \
        --model_name_or_path "Qwen/Qwen3-1.7B" \
        --cache_dir "/shared/rsaas/qiqianf2/hf_models" \
        --data_path "${DATA_PATH}" \
        --max_seq_length 4096 \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 128 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --bf16 True \
        --logging_steps 5 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 3 \
        --dataloader_num_workers 2 \
        --report_to none \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False
else
    conda run -n qwen_RL --no-capture-output \
        python3 "${SCRIPT_DIR}/train_sft.py" \
        --model_name_or_path "Qwen/Qwen3-1.7B" \
        --cache_dir "/shared/rsaas/qiqianf2/hf_models" \
        --data_path "${DATA_PATH}" \
        --max_seq_length 4096 \
        --use_lora True \
        --lora_r 64 \
        --lora_alpha 128 \
        --output_dir "${OUTPUT_DIR}" \
        --num_train_epochs 3 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --gradient_accumulation_steps 16 \
        --learning_rate 2e-4 \
        --lr_scheduler_type cosine \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --bf16 True \
        --logging_steps 5 \
        --save_strategy epoch \
        --eval_strategy epoch \
        --save_total_limit 3 \
        --dataloader_num_workers 2 \
        --report_to none \
        --gradient_checkpointing True \
        --ddp_find_unused_parameters False
fi

echo ""
echo "=== SFT Training finished ==="
echo "Output: ${OUTPUT_DIR}"
echo "End: $(date)"
