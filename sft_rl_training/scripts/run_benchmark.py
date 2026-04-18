"""
Evaluate a checkpoint on the behavior benchmark (41 entries).

Per entry:
  - Feed `input_messages` + TOOL_SCHEMAS to the model, generate next assistant turn
  - Parse <tool_call>...</tool_call> blocks out of the output
  - Rule-based score vs expected_behavior:
      - should_call_tool: model must call a tool; tool name must be in
        expected_tools ∪ alternative_tools (if either is non-empty)
      - should_clarify:   model must NOT call a tool AND content must contain
        a question mark (clarification heuristic)
      - should_not_call_tool: model must NOT call a tool
  - Output per-entry results + per-bucket summary

Usage:
    # Base Qwen3-1.7B, no adapter
    python run_benchmark.py --output results_base.jsonl --tag base

    # Exp-002 SFT adapter
    python run_benchmark.py --adapter /shared/rsaas/qiqianf2/lc_agent_experiments/sft_real_exp002 \\
        --output results_exp002.jsonl --tag exp002

    # Exp-009 checkpoint-60
    python run_benchmark.py --adapter /shared/rsaas/qiqianf2/lc_agent_experiments/sft_exp008/checkpoint-60 \\
        --output results_exp009_ep3.jsonl --tag exp009_ep3
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# leetcode_agent TOOL_SCHEMAS
LEETCODE_SRC = Path(__file__).resolve().parents[2] / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))
from lc.tool_defs import TOOLS as TOOL_SCHEMAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-1.7B"
CACHE_DIR = "/shared/rsaas/qiqianf2/hf_models"
BENCH_PATH = "/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/benchmark.jsonl"

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
QUESTION_RE = re.compile(r"[?？]")


# ─── Model loading ───

def load_model(adapter_path: str | None):
    logger.info("Loading tokenizer (%s)", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True)
    logger.info("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, cache_dir=CACHE_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )
    if adapter_path:
        from peft import PeftModel
        logger.info("Loading adapter from %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    logger.info("Model loaded on %s", model.device)
    return model, tokenizer


# ─── Inference ───

def generate(model, tokenizer, messages, max_new_tokens=512, temperature=0.3):
    # enable_thinking=False: Qwen3 default would emit <think>...</think> blocks
    try:
        text = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS,
            add_generation_prompt=True, tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS,
            add_generation_prompt=True, tokenize=False,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = out[0][input_len:]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any leftover <think> block defensively
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return raw


def parse_response(raw: str) -> dict:
    """Return {'content': str_without_tool_tags, 'tool_calls': [{'name','arguments'}...]}"""
    tool_calls = []
    for match in TOOL_CALL_RE.findall(raw):
        try:
            parsed = json.loads(match)
            tool_calls.append({
                "name": parsed.get("name", ""),
                "arguments": parsed.get("arguments", {}),
            })
        except json.JSONDecodeError:
            pass
    content = TOOL_CALL_RE.sub("", raw).strip()
    return {"content": content, "tool_calls": tool_calls, "raw": raw}


# ─── Scoring ───

def rule_score(entry: dict, response: dict) -> tuple[int, str]:
    """Return (score 0 or 1, reason)."""
    expected = entry["expected_behavior"]
    tcs = response["tool_calls"]
    content = response["content"]
    has_tc = bool(tcs)

    if expected == "should_call_tool":
        if not has_tc:
            return 0, "expected tool call, got text only"
        called = tcs[0]["name"]
        allowed = set(entry.get("expected_tools") or []) | set(entry.get("alternative_tools") or [])
        if not allowed:
            # free trajectories: any valid tool is OK
            return 1, f"any tool OK, got '{called}'"
        if called in allowed:
            return 1, f"'{called}' ∈ {sorted(allowed)}"
        return 0, f"'{called}' ∉ {sorted(allowed)}"

    if expected == "should_clarify":
        if has_tc:
            return 0, f"expected clarify, got tool '{tcs[0]['name']}'"
        if not content:
            return 0, "empty content"
        if not QUESTION_RE.search(content):
            return 0, "no question mark — not a clarifying reply"
        return 1, "abstained + asked question"

    if expected == "should_not_call_tool":
        if has_tc:
            return 0, f"expected no-tool, got '{tcs[0]['name']}'"
        if not content:
            return 0, "empty content"
        return 1, "correctly abstained"

    return 0, f"unknown expected_behavior: {expected}"


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="Optional LoRA adapter path (omit = run base Qwen3-1.7B)")
    parser.add_argument("--tag", type=str, required=True,
                        help="Short tag for this run (used in output filenames + stdout)")
    parser.add_argument("--output", type=str, required=True,
                        help="Per-entry results JSONL")
    parser.add_argument("--summary_out", type=str, default=None,
                        help="Per-bucket summary JSON (defaults to <output>.summary.json)")
    parser.add_argument("--benchmark", type=str, default=BENCH_PATH)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.3)
    args = parser.parse_args()

    if args.summary_out is None:
        args.summary_out = args.output + ".summary.json"

    # Load benchmark
    entries = [json.loads(l) for l in open(args.benchmark, encoding="utf-8") if l.strip()]
    logger.info("Loaded %d benchmark entries from %s", len(entries), args.benchmark)

    # Load model
    model, tokenizer = load_model(args.adapter)

    # Run inference + scoring
    results = []
    bucket_stats: dict[str, list[int]] = defaultdict(list)
    beh_stats: dict[str, list[int]] = defaultdict(list)

    t0 = time.time()
    for i, e in enumerate(entries):
        raw = generate(model, tokenizer, e["input_messages"],
                       max_new_tokens=args.max_new_tokens, temperature=args.temperature)
        resp = parse_response(raw)
        score, reason = rule_score(e, resp)

        bucket = "/".join(e["stratification"])
        bucket_stats[bucket].append(score)
        beh_stats[e["expected_behavior"]].append(score)

        out_row = {
            "id": e["id"],
            "source": e["source"],
            "stratification": e["stratification"],
            "test_user_message": e["test_user_message"],
            "expected_behavior": e["expected_behavior"],
            "expected_tools": e.get("expected_tools"),
            "alternative_tools": e.get("alternative_tools"),
            "model_content": resp["content"][:500],
            "model_tool_calls": resp["tool_calls"],
            "model_raw_preview": resp["raw"][:300],
            "score": score,
            "reason": reason,
        }
        results.append(out_row)
        logger.info("  [%02d/%d] %s  score=%d  %s", i + 1, len(entries),
                    e["id"], score, reason[:80])

    elapsed = time.time() - t0

    # Write per-entry results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total = len(results)
    correct = sum(r["score"] for r in results)
    summary = {
        "tag": args.tag,
        "adapter": args.adapter or "base",
        "n_total": total,
        "n_correct": correct,
        "overall_accuracy": correct / total if total else 0.0,
        "elapsed_sec": round(elapsed, 1),
        "by_expected_behavior": {
            beh: {"n": len(v), "correct": sum(v), "acc": sum(v) / len(v) if v else 0}
            for beh, v in beh_stats.items()
        },
        "by_bucket": {
            b: {"n": len(v), "correct": sum(v), "acc": sum(v) / len(v) if v else 0}
            for b, v in sorted(bucket_stats.items())
        },
    }
    Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print report
    print(f"\n=== Benchmark results [{args.tag}] ===")
    print(f"Adapter:   {args.adapter or '(none — base Qwen3-1.7B)'}")
    print(f"Overall:   {correct}/{total} = {correct/total*100:.1f}%   ({elapsed:.0f}s)")
    print(f"\nBy expected_behavior:")
    for beh, v in summary["by_expected_behavior"].items():
        print(f"  {beh:25s}  {v['correct']}/{v['n']}  = {v['acc']*100:5.1f}%")
    print(f"\nBy bucket:")
    for b, v in summary["by_bucket"].items():
        print(f"  {b:55s}  {v['correct']}/{v['n']}  = {v['acc']*100:5.1f}%")
    print(f"\nOutput:    {out_path}")
    print(f"Summary:   {args.summary_out}")


if __name__ == "__main__":
    main()
