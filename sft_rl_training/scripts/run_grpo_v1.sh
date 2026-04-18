#!/bin/bash
# Exp-014 GRPO v1: 20 rubric instances, group=4, beta=0.1, 5 epochs.
set -e

cd /home/qiqianf2/LC-Agent/sft_rl_training

echo "=== Exp-014 GRPO v1 ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

PY=/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python

SCRIPT=/home/qiqianf2/LC-Agent/sft_rl_training/scripts/train_grpo.py
MODEL=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_exp009_merged
DATA=/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.parquet
INSTANCES=/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.jsonl
OUT=/shared/rsaas/qiqianf2/lc_agent_experiments/grpo_v1_exp014

export DEEPSEEK_API_KEY="sk-REDACTED"
export RL_INSTANCES_PATH="$INSTANCES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Num GPUs:   $NUM_GPUS"
echo "Model:      $MODEL"
echo "Data:       $DATA"
echo "Instances:  $INSTANCES"
echo "Output:     $OUT"
echo ""

# Single-process, 1 GPU. grad_accum=4 and num_gen=4 → gen_batch=4:
# 1 unique prompt per step × 4 rollouts.
# 20 prompts × 5 epochs / 1 prompt-per-step = 100 steps.
"$PY" "$SCRIPT" \
    --model_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 5 \
    --beta 0.1 \
    --learning_rate 5e-7 \
    --max_completion_length 2048

echo ""
echo "=== Exp-014 finished ==="
date
