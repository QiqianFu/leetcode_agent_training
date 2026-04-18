#!/bin/bash
# GRPO Training for LeetCode Agent using TRL (Exp-004)
set -e

echo "=== LeetCode Agent GRPO Training (TRL) ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
echo "Start: $(date)"

# Override to use free GPUs on c22
export CUDA_VISIBLE_DEVICES="3,4,5,6"
NUM_GPUS=4
echo "Using GPUs: ${CUDA_VISIBLE_DEVICES}"

# Try to install flash-attn on GPU node (has nvcc)
echo "Attempting flash-attn install on GPU node..."
CUDA_HOME=/usr/local/cuda conda run -n qwen_RL --no-capture-output pip install flash-attn --no-build-isolation --no-cache-dir 2>&1 | tail -3 || echo "flash-attn install failed, using eager attention"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="/home/qiqianf2/LC-Agent/sft_rl_training/scripts"
MODEL_PATH="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_merged_model"
DATA_PATH="/shared/rsaas/qiqianf2/lc_agent_experiments/rl_prompts.parquet"
OUTPUT_DIR="/shared/rsaas/qiqianf2/lc_agent_experiments/grpo_trl_exp001"

# Load DEEPSEEK_API_KEY (and other secrets) from project-root .env — fail hard if missing
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$PROJECT_ROOT/.env" ] || { echo "ERROR: $PROJECT_ROOT/.env missing — cp .env.example .env and fill it in" >&2; exit 1; }
set -a; . "$PROJECT_ROOT/.env"; set +a
: "${DEEPSEEK_API_KEY:?ERROR: DEEPSEEK_API_KEY not set in .env}"

if [ "${NUM_GPUS}" -gt 1 ]; then
    conda run -n qwen_RL --no-capture-output \
        accelerate launch \
        --num_processes ${NUM_GPUS} \
        --mixed_precision bf16 \
        "${SCRIPT_DIR}/train_grpo.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}"
else
    conda run -n qwen_RL --no-capture-output \
        python3 "${SCRIPT_DIR}/train_grpo.py" \
        --model_path "${MODEL_PATH}" \
        --data_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}"
fi

echo ""
echo "=== GRPO Training finished ==="
echo "Output: ${OUTPUT_DIR}"
echo "End: $(date)"
