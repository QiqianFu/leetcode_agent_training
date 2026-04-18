#!/bin/bash
# Run Exp-009 checkpoint-60 benchmark on a condor node (c23).
set -e
cd /home/qiqianf2/LC-Agent/sft_rl_training

echo "=== Benchmark on condor ==="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

/shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python -u scripts/run_benchmark.py \
    --tag exp009_ep3_c23 \
    --adapter /shared/rsaas/qiqianf2/lc_agent_experiments/sft_exp008/checkpoint-60 \
    --output /shared/rsaas/qiqianf2/lc_agent_experiments/benchmark_results/results_exp009_ep3_c23.jsonl

echo "=== Done ==="
date
