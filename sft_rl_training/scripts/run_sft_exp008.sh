#!/bin/bash
# Exp-008 SFT: train_merged.jsonl (653) on Qwen3-1.7B + LoRA, 4 GPUs (c22: 4,5,6,7), 5 epochs.
set -e

cd /home/qiqianf2/LC-Agent/sft_rl_training

# GPUs are allocated by condor (request_gpus=4 in the sub file).
# CUDA_VISIBLE_DEVICES will be set by condor — do not override here.

echo "=== Exp-008 SFT ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

PY=/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python
ACCEL=/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/accelerate

SCRIPT=/home/qiqianf2/LC-Agent/sft_rl_training/scripts/train_sft.py
DATA=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/train_merged.jsonl
OUT=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_exp008

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Num GPUs: $NUM_GPUS"
echo "Data:     $DATA"
echo "Output:   $OUT"
echo ""

"$ACCEL" launch \
    --num_processes "$NUM_GPUS" \
    --mixed_precision bf16 \
    "$SCRIPT" \
    --model_name_or_path "Qwen/Qwen3-1.7B" \
    --cache_dir "/shared/rsaas/qiqianf2/hf_models" \
    --data_path "$DATA" \
    --max_seq_length 4096 \
    --use_lora True \
    --lora_r 64 \
    --lora_alpha 128 \
    --output_dir "$OUT" \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --bf16 True \
    --logging_steps 5 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 5 \
    --dataloader_num_workers 2 \
    --report_to none \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False

echo ""
echo "=== Exp-008 finished ==="
date
