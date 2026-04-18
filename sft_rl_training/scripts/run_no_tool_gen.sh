#!/bin/bash
# Run no-tool-call data generation with multi-GPU fan-out.
# Uses `conda run -n qwen_RL` pattern (same as run_sft.sh) to avoid activation hangs.

set -e

cd /home/qiqianf2/LC-Agent/sft_rl_training

# Override whatever condor sets — we manually pin to avoid collisions with
# non-condor jobs that grabbed GPUs outside condor's accounting.
export CUDA_VISIBLE_DEVICES=1,4,5,7

echo "=== Environment ==="
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Host=$(hostname)"
date

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "ERROR: CUDA_VISIBLE_DEVICES not set"
    exit 1
fi
IFS=',' read -ra GPUS <<< "$CUDA_VISIBLE_DEVICES"
NUM_GPUS=${#GPUS[@]}
echo "Num GPUs detected: $NUM_GPUS"

USER_ARGS="$@"
echo "User args: $USER_ARGS"

# --- Warm up NFS page cache for the conda env before Python tries to import ---
# On cold execute nodes, /shared/rsaas is slow; sequential `cat` + dd pulls
# the bytes into the NFS client cache so later random-access imports are fast.
echo ""
echo "=== Warming NFS cache (conda env + HF cache) ==="
WARM_START=$(date +%s)
find /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/lib/python3.10/site-packages/torch \
     /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/lib/python3.10/site-packages/transformers \
     /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/lib/python3.10/site-packages/tokenizers \
     /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/lib/python3.10/site-packages/safetensors \
     -type f \( -name '*.so' -o -name '*.pyc' -o -name '*.py' \) \
     -exec cat {} + > /dev/null 2>&1 || true
WARM_END=$(date +%s)
echo "Warm up took $((WARM_END - WARM_START))s"

# --- Stage 1: scenario pool (DeepSeek only, single process, no GPU) ---
echo ""
echo "=== Stage 1: Scenario pool prep ==="
CUDA_VISIBLE_DEVICES=${GPUS[0]} /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python -u \
    scripts/generate_no_tool_data.py --prep_scenarios_only $USER_ARGS
echo "Scenario pool ready."

# --- Stage 2: fan-out across GPUs ---
ALL_CATS=(greeting_or_feature status_sharing ambiguous_intent mid_problem_chat complaint_or_giveup off_topic)
OUT_DIR=/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories

declare -a WORKER_CATS
for i in "${!ALL_CATS[@]}"; do
    widx=$((i % NUM_GPUS))
    WORKER_CATS[$widx]="${WORKER_CATS[$widx]} ${ALL_CATS[$i]}"
done

rm -f $OUT_DIR/no_tool_data_part*.jsonl

echo ""
echo "=== Stage 2: Parallel sampling ==="
PIDS=()
for i in "${!GPUS[@]}"; do
    if [ -z "${WORKER_CATS[$i]}" ]; then
        echo "GPU ${GPUS[$i]}: (no categories assigned)"
        continue
    fi
    gpu=${GPUS[$i]}
    cats="${WORKER_CATS[$i]}"
    out_file="$OUT_DIR/no_tool_data_part${i}.jsonl"
    log_file="/home/qiqianf2/LC-Agent/sft_rl_training/logs/no_tool_worker_${i}.log"
    echo "GPU $gpu → categories:$cats → $out_file"

    (
        CUDA_VISIBLE_DEVICES=$gpu /shared/rsaas/qiqianf2/anaconda3/envs/qwen_RL/bin/python -u \
            scripts/generate_no_tool_data.py \
                --categories $cats \
                --output "$out_file" \
                $USER_ARGS
    ) > "$log_file" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "Waiting for ${#PIDS[@]} worker(s)..."
FAIL=0
for pid in "${PIDS[@]}"; do
    if ! wait $pid; then
        echo "WARN: worker pid $pid exited with error"
        FAIL=$((FAIL+1))
    fi
done

echo ""
echo "=== Stage 3: Concat outputs ==="
FINAL_OUT="$OUT_DIR/no_tool_data.jsonl"
cat $OUT_DIR/no_tool_data_part*.jsonl > "$FINAL_OUT"
TOTAL=$(wc -l < "$FINAL_OUT")
echo "Total samples: $TOTAL"
echo "Final output: $FINAL_OUT"
date

if [ $FAIL -gt 0 ]; then
    echo "WARN: $FAIL worker(s) had errors — check per-worker logs"
    exit 1
fi
