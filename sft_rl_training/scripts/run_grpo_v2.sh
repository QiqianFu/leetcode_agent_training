#!/bin/bash
# Exp-015 GRPO v2: loosen KL + bump LR.
# Exp-014 (beta=0.1, lr=5e-7) showed reward flat at ~0.60, kl stuck at 3-5e-4.
# v2: beta 0.1 → 0.01 (10x smaller; still keeps ref model for drift safety)
#     lr   5e-7 → 2e-6 (4x larger; still 1% of SFT lr)
# Same 20 instances × 5 epochs × group=4.
set -e

cd /home/qiqianf2/LC-Agent/sft_rl_training

echo "=== Exp-015 GRPO v2 (looser params) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

PY=/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python

SCRIPT=/home/qiqianf2/LC-Agent/sft_rl_training/scripts/train_grpo.py
MODEL=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_exp009_merged
DATA=/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.parquet
INSTANCES=/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.jsonl
OUT=/shared/rsaas/qiqianf2/lc_agent_experiments/grpo_v2_exp015

export DEEPSEEK_API_KEY="sk-REDACTED"
export RL_INSTANCES_PATH="$INSTANCES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Num GPUs:   $NUM_GPUS"
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Instances:  $INSTANCES"
echo "Output:     $OUT"
echo "Config:     beta=0.01, lr=2e-6 (vs Exp-014's 0.1, 5e-7)"
echo ""

"$PY" "$SCRIPT" \
    --model_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --beta 0.01 \
    --learning_rate 2e-6 \
    --max_completion_length 2048

echo ""
echo "=== Exp-015 finished ==="
date
