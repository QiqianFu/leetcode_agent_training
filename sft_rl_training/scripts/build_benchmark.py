"""
Build a behavior benchmark by sampling ~5% from each (source × sub-class) of
the SFT training data.

Sources:
  - all_trajectories.jsonl  (Exp-002: 511 structured+free)
  - clarify_all.jsonl       (Exp-006: 66 clarify)
  - no_tool_data.jsonl      (Exp-007: 117 no-tool)

Stratification:
  - structured : step1_variant × code_quality  (9 sub-classes)
  - free       : 1 class
  - clarify    : concrete_variant (random/specific/topic/difficulty)
  - no_tool    : category × turns

Each class gets max(1, round(n * RATIO)) samples. Default RATIO=0.05.

Output: benchmark.jsonl with first-user-message + expected_behavior +
attributes so that downstream LLM-as-judge can score model responses.

Usage:
    python build_benchmark.py
    python build_benchmark.py --ratio 0.05 --seed 42 --output benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path("/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories")

SOURCES = [
    ("all", "all_trajectories.jsonl"),
    ("clarify", "clarify_all.jsonl"),
    ("no_tool", "no_tool_data.jsonl"),
]


# ─── Stratification helpers ───

def stratify_key(source: str, entry: dict) -> tuple[str, ...]:
    """Return the sub-class key for this entry (for stratification buckets)."""
    t = entry.get("type", "")
    if source == "all":
        if t == "structured":
            cfg = entry.get("config", {})
            return ("structured", cfg.get("step1", "?"), cfg.get("code_quality", "?"))
        elif t == "free":
            return ("free",)
        return (t or "unknown",)
    if source == "clarify":
        cfg = entry.get("config", {})
        return ("clarify", cfg.get("concrete_variant", "?"))
    if source == "no_tool":
        return ("no_tool", entry.get("category", "?"), f"{entry.get('turns','?')}t")
    return (source,)


# ─── Expected behavior logic ───

def expected_behavior(source: str, entry: dict) -> dict:
    """Determine expected first-turn assistant behavior given the user's first message.

    Returns a dict with:
      - expected_behavior: one of {should_call_tool, should_clarify, should_not_call_tool}
      - expected_tools:    list of tool names that are acceptable first calls (empty = judge decides)
      - alternative_tools: list of tools that are also acceptable as a "safer" alternative
      - judge_notes:       natural-language guidance for the LLM judge
    """
    t = entry.get("type", "")
    cfg = entry.get("config", {})

    if source == "no_tool":
        cat = entry.get("category", "")
        if cat == "ambiguous_intent":
            return {
                "expected_behavior": "should_clarify",
                "expected_tools": [],
                "alternative_tools": [],
                "judge_notes": "用户意图模糊（如'帮我看看'、'随便来点'），assistant 第一轮应该用自然话反问澄清需求，不应该调用任何工具。",
            }
        notes_by_cat = {
            "greeting_or_feature": "打招呼或询问功能，应用文字介绍/问候回应，不调工具。",
            "status_sharing": "用户分享情绪/状态，应共情回应，不调工具（可轻度引导但不必）。",
            "mid_problem_chat": "做题过程中的纯讨论（思路/复杂度等），应直接文字讲解，不调工具。",
            "complaint_or_giveup": "用户抱怨或放弃，应共情回应，不调工具。",
            "off_topic": "偏离主题的闲聊，应礼貌回应（可选轻度拉回），不调工具。",
        }
        return {
            "expected_behavior": "should_not_call_tool",
            "expected_tools": [],
            "alternative_tools": [],
            "judge_notes": notes_by_cat.get(cat, "纯文字对话，不应调工具。"),
        }

    if source == "clarify":
        return {
            "expected_behavior": "should_clarify",
            "expected_tools": [],
            "alternative_tools": [],
            "judge_notes": "用户第一句话意图模糊（非具体题号/题型/随机推荐），assistant 应先反问澄清需求，不应立即调用工具。",
        }

    if source == "all":
        if t == "structured":
            step1 = cfg.get("step1", "")
            if step1 == "random":
                return {
                    "expected_behavior": "should_call_tool",
                    "expected_tools": ["pick_problem"],
                    "alternative_tools": [],
                    "judge_notes": "用户说'来一道题'类随机意图，应调 pick_problem（从高频题推荐）。",
                }
            if step1 == "specific":
                return {
                    "expected_behavior": "should_call_tool",
                    "expected_tools": ["start_problem"],
                    "alternative_tools": ["check_problem"],
                    "judge_notes": "用户指定题号，应调 start_problem 开题；check_problem 先查也可接受。",
                }
            if step1 == "topic":
                return {
                    "expected_behavior": "should_call_tool",
                    "expected_tools": ["search_problem"],
                    "alternative_tools": [],
                    "judge_notes": "用户指定题型/方向（如 DP、树），应调 search_problem 按关键词搜索。",
                }
        if t == "free":
            return {
                "expected_behavior": "should_call_tool",
                "expected_tools": [],
                "alternative_tools": [],
                "judge_notes": "自由交互意图（具体意图见 first_user_message），judge 自行判断该调哪个工具或是否该调。",
            }

    return {
        "expected_behavior": "unknown",
        "expected_tools": [],
        "alternative_tools": [],
        "judge_notes": "未分类 — 人工审核。",
    }


# ─── Extraction helpers ───

def extract_test_input(entry: dict, source: str) -> list[dict]:
    """Return the message list we feed to the model at eval time.
    The model is expected to produce the NEXT assistant turn after these messages,
    and that response is what we compare against `expected_behavior`.

    - For mid_problem_chat entries: we include the borrowed problem-setup prefix
      plus the mid-problem user question, because the behavior under test is
      "pure discussion after a problem was already started"; without the prefix
      the test question makes no sense.
    - For everything else: we test the very first user turn.
    """
    messages = entry.get("messages", [])
    if source == "no_tool" and entry.get("category") == "mid_problem_chat":
        # Drop the final assistant response (training target); keep everything before.
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                return messages[: i + 1]
        return messages
    if source == "clarify":
        cfg = entry.get("config", {})
        if cfg.get("vague_first_msg"):
            return [{"role": "user", "content": cfg["vague_first_msg"]}]
    for m in messages:
        if m.get("role") == "user":
            return [{"role": "user", "content": m.get("content", "") or ""}]
    return []


def extract_reference_response(entry: dict, source: str) -> dict:
    """Return the original assistant turn AT THE TEST POINT (what the training data
    considered the 'correct' response to the input_messages). Useful for the LLM
    judge to compare against, but not authoritative."""
    messages = entry.get("messages", [])
    if source == "no_tool" and entry.get("category") == "mid_problem_chat":
        # Final message in the trajectory is the training-target assistant response.
        if messages and messages[-1].get("role") == "assistant":
            m = messages[-1]
            return {
                "content": m.get("content", ""),
                "tool_calls": [tc.get("function", tc) for tc in (m.get("tool_calls") or [])],
            }
        return {"content": "", "tool_calls": []}
    # Otherwise: the FIRST assistant turn after the first user.
    seen_first_user = False
    for m in messages:
        if m.get("role") == "user":
            seen_first_user = True
            continue
        if seen_first_user and m.get("role") == "assistant":
            return {
                "content": m.get("content", ""),
                "tool_calls": [tc.get("function", tc) for tc in (m.get("tool_calls") or [])],
            }
    return {"content": "", "tool_calls": []}


def extract_reference_tools(entry: dict) -> list[str]:
    """All tool names that were actually called anywhere in the original trajectory (first 5)."""
    tools = []
    for m in entry.get("messages", []):
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls", []) or []:
                fn = tc.get("function", {}) if "function" in tc else tc
                name = fn.get("name") or tc.get("name")
                if name:
                    tools.append(name)
        if len(tools) >= 5:
            break
    return tools


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default=str(DATA_DIR / "benchmark.jsonl"))
    parser.add_argument("--stats_out", type=str,
                        default=str(DATA_DIR / "benchmark_stats.json"))
    parser.add_argument("--train_out", type=str,
                        default=str(DATA_DIR / "train_merged.jsonl"),
                        help="Merged SFT train set (all 3 sources) with benchmark entries excluded")
    args = parser.parse_args()

    random.seed(args.seed)

    # ── Load and bucket ──
    # Note: all_trajectories.jsonl has internal ID collisions (multiple shards
    # concatenated without rename), so we track (source, line_idx) as the unique
    # key, not `id`.
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    per_source_count: Counter = Counter()

    for source, fname in SOURCES:
        p = DATA_DIR / fname
        if not p.exists():
            print(f"[WARN] Missing {p}")
            continue
        with open(p, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                d["_unique_key"] = (source, idx)
                key = (source,) + stratify_key(source, d)
                buckets[key].append(d)
                per_source_count[source] += 1

    print("\n=== Source totals ===")
    for s, n in per_source_count.items():
        print(f"  {s:10s}  {n}")

    print(f"\n=== Stratified classes ({len(buckets)} buckets) ===")
    for k in sorted(buckets.keys()):
        print(f"  {'/'.join(k):55s}  n={len(buckets[k])}")

    # ── Sample ~ratio per bucket (min 1) ──
    samples: list[dict] = []
    per_bucket_sampled: Counter = Counter()

    for key, entries in sorted(buckets.items()):
        n = len(entries)
        k = max(1, round(n * args.ratio))
        picked = random.sample(entries, k=min(k, n))
        per_bucket_sampled[key] = len(picked)

        source = key[0]
        for d in picked:
            exp = expected_behavior(source, d)
            input_messages = extract_test_input(d, source)
            if not input_messages or input_messages[-1].get("role") != "user":
                print(f"[WARN] Bad input_messages for {d.get('id')}; skipping")
                continue
            # Convenience: a flat "last user" string (what the judge will highlight)
            last_user = input_messages[-1].get("content", "")
            bench = {
                "id": f"bench_{len(samples):04d}",
                "source": source,
                "source_file": dict(SOURCES)[source],
                "source_id": d.get("id", ""),
                "source_line_idx": d["_unique_key"][1],
                "stratification": list(key),
                "attributes": {
                    "type": d.get("type"),
                    "config": d.get("config", {}),
                    "category": d.get("category"),
                    "turns": d.get("turns"),
                    "shard": d.get("shard"),
                },
                "input_messages": input_messages,
                "test_user_message": last_user,
                "expected_behavior": exp["expected_behavior"],
                "expected_tools": exp["expected_tools"],
                "alternative_tools": exp["alternative_tools"],
                "judge_notes": exp["judge_notes"],
                "reference_response": extract_reference_response(d, source),
                "reference_trajectory_tools": extract_reference_tools(d),
            }
            samples.append(bench)

    # ── Write output ──
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Stats JSON for quick reference
    stats = {
        "total_samples": len(samples),
        "ratio": args.ratio,
        "seed": args.seed,
        "per_source_population": dict(per_source_count),
        "per_bucket_population": {"/".join(k): len(v) for k, v in sorted(buckets.items())},
        "per_bucket_sampled": {"/".join(k): v for k, v in sorted(per_bucket_sampled.items())},
        "expected_behavior_breakdown": dict(Counter(s["expected_behavior"] for s in samples)),
    }
    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n=== Sampled ({len(samples)} total) ===")
    for k in sorted(per_bucket_sampled.keys()):
        print(f"  {'/'.join(k):55s}  sampled={per_bucket_sampled[k]:3d} / {len(buckets[k]):3d}")

    print("\n=== Expected behavior breakdown ===")
    for b, n in stats["expected_behavior_breakdown"].items():
        print(f"  {b:25s}  {n}")

    # ── Write train split (all sources merged, benchmark entries excluded) ──
    # Match by (source, line_idx) because all_trajectories.jsonl has duplicate `id`s.
    held_out_keys = {(s["source"], s["source_line_idx"]) for s in samples}
    train_path = Path(args.train_out)
    train_count = 0
    dropped_count = 0
    with open(train_path, "w", encoding="utf-8") as out:
        for source, fname in SOURCES:
            p = DATA_DIR / fname
            if not p.exists():
                continue
            with open(p, encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    if (source, idx) in held_out_keys:
                        dropped_count += 1
                        continue
                    d = json.loads(line)
                    if "_bench_source" not in d:
                        d["_bench_source"] = source
                    out.write(json.dumps(d, ensure_ascii=False) + "\n")
                    train_count += 1

    stats["train_size"] = train_count
    stats["train_dropped_as_benchmark"] = dropped_count
    Path(args.stats_out).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBenchmark: {out_path}     ({len(samples)} entries)")
    print(f"Train:     {train_path}     ({train_count} entries, dropped {dropped_count} as benchmark)")
    print(f"Stats:     {args.stats_out}")


if __name__ == "__main__":
    main()
