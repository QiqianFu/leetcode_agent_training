#!/bin/bash
# GRPO dry-run: verify train_grpo.py pipeline end-to-end on a tiny config.
# 1 GPU, 4 prompts, num_generations=2, max_steps=2. Should finish in minutes.
set -e

cd /home/qiqianf2/LC-Agent/sft_rl_training

echo "=== GRPO Dry-run (Exp-011) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

PY=/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python

SCRIPT=/home/qiqianf2/LC-Agent/sft_rl_training/scripts/train_grpo.py
MODEL=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_merged_model
DATA=/shared/rsaas/qiqianf2/lc_agent_experiments/rl_prompts.parquet
OUT=/shared/rsaas/qiqianf2/lc_agent_experiments/grpo_dryrun

# Load DEEPSEEK_API_KEY (and other secrets) from project-root .env — fail hard if missing
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$PROJECT_ROOT/.env" ] || { echo "ERROR: $PROJECT_ROOT/.env missing — cp .env.example .env and fill it in" >&2; exit 1; }
set -a; . "$PROJECT_ROOT/.env"; set +a
: "${DEEPSEEK_API_KEY:?ERROR: DEEPSEEK_API_KEY not set in .env}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
echo "Num GPUs: $NUM_GPUS"
echo "Data:     $DATA"
echo "Output:   $OUT"
echo ""

# Single-process for dry-run (no accelerate). 1 GPU.
"$PY" "$SCRIPT" \
    --model_path "$MODEL" \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps 2 \
    --limit_prompts 4 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2

echo ""
echo "=== GRPO Dry-run finished ==="
date
