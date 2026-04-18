"""Analyze rubric discrimination from judged rollouts.

Answers the three pre-flight questions:
  1. Does same-model × same-instance show variance? (advantage != 0 for GRPO)
  2. Do 3 tiers rank correctly (DeepSeek > SFT > Base)?
  3. Which rubric criteria are useless (always pass or always fail)?

Usage:
    python analyze_rubric_discrimination.py \\
      --instances /path/to/v1.jsonl \\
      --judged base=/path/to/judged_base.jsonl \\
      --judged sft=/path/to/judged_sft.jsonl \\
      --judged deepseek=/path/to/judged_deepseek.jsonl \\
      [--output /path/to/report.md]
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from collections import defaultdict
from pathlib import Path


def parse_judged(path: Path):
    """Yield judged records from a jsonl file."""
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--judged", action="append", required=True,
                    help="tag=path pairs, e.g. base=judged_base.jsonl")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional markdown report path; otherwise stdout")
    args = ap.parse_args()

    # Parse tag=path
    judged_sources = {}
    for spec in args.judged:
        if "=" not in spec:
            print(f"Invalid --judged {spec}, expected tag=path", file=sys.stderr)
            sys.exit(1)
        tag, path = spec.split("=", 1)
        judged_sources[tag] = Path(path)

    # Load instances
    instances_by_id = {}
    with open(args.instances) as f:
        for line in f:
            inst = json.loads(line)
            instances_by_id[inst["id"]] = inst

    # Load judged: {tag: {instance_id: [scores...]}}
    # And per-criterion verdicts: {tag: {instance_id: {rubric_id: [verdict...]}}}
    scores = {tag: defaultdict(list) for tag in judged_sources}
    verdicts = {tag: defaultdict(lambda: defaultdict(list))
                for tag in judged_sources}
    for tag, path in judged_sources.items():
        if not path.exists():
            print(f"WARN: {tag} judged file missing: {path}", file=sys.stderr)
            continue
        for j in parse_judged(path):
            scores[tag][j["instance_id"]].append(j["score"])
            for rid, v in j.get("verdicts", {}).items():
                verdicts[tag][j["instance_id"]][rid].append(v["verdict"])

    out_lines = []

    def emit(s: str = ""):
        out_lines.append(s)

    emit("# Rubric discrimination pre-flight analysis\n")
    emit(f"Instances: {len(instances_by_id)}")
    emit(f"Models: {', '.join(judged_sources.keys())}\n")

    # ─── Q1: per-instance variance per model (advantage != 0 check) ────

    emit("## Q1. Per-instance variance (GRPO advantage sanity)")
    emit("")
    emit("std=0 → 4 rollouts scored identically → **advantage=0 → that prompt is wasted in GRPO**.")
    emit("For the model being RL'd (SFT), we want std>0 on most instances.\n")

    emit("| id | label | " + " | ".join(f"{t} mean±std" for t in judged_sources) + " |")
    emit("|---|---|" + "---|" * len(judged_sources))
    zero_var_per_tag = defaultdict(int)
    low_var_per_tag = defaultdict(int)
    for inst_id in sorted(instances_by_id):
        inst = instances_by_id[inst_id]
        row = [inst_id, inst["label"]]
        for tag in judged_sources:
            s = scores[tag].get(inst_id, [])
            if len(s) < 2:
                row.append("—")
                continue
            m = stats.mean(s)
            sd = stats.stdev(s)
            if sd == 0:
                zero_var_per_tag[tag] += 1
            elif sd < 0.05:
                low_var_per_tag[tag] += 1
            flag = ""
            if sd == 0:
                flag = " ⚠0"
            elif sd < 0.05:
                flag = " ⚠"
            row.append(f"{m:.2f}±{sd:.2f}{flag}")
        emit("| " + " | ".join(row) + " |")

    emit("")
    emit("**Zero-variance count** (⚠0):")
    for tag in judged_sources:
        n = zero_var_per_tag[tag]
        emit(f"- {tag}: {n}/{len(instances_by_id)} "
             f"({'BAD — would waste RL budget' if tag == 'sft' and n > len(instances_by_id)//2 else 'OK/expected' if n < 5 else 'review'})")
    emit("")

    # ─── Q2: 3-tier ranking check ──────────────────────────────────────

    emit("## Q2. 3-tier ranking (DeepSeek > SFT > Base) check")
    emit("")
    emit("Per-instance mean score comparison. Row = correct ranking respected.\n")

    tiers = ["base", "sft", "deepseek"]  # expected ascending
    if all(t in judged_sources for t in tiers):
        correct = 0
        partial = 0
        wrong = 0
        emit("| id | base | sft | deepseek | ranking |")
        emit("|---|---|---|---|---|")
        for inst_id in sorted(instances_by_id):
            mb = stats.mean(scores["base"].get(inst_id, [0]))
            ms = stats.mean(scores["sft"].get(inst_id, [0]))
            md = stats.mean(scores["deepseek"].get(inst_id, [0]))
            if mb <= ms <= md:
                status = "✓"
                correct += 1
            elif mb <= md and ms <= md:
                status = "~"  # DeepSeek on top at least
                partial += 1
            else:
                status = "✗"
                wrong += 1
            emit(f"| {inst_id} | {mb:.2f} | {ms:.2f} | {md:.2f} | {status} |")

        emit("")
        n = len(instances_by_id)
        emit(f"**Ranking correctness**: ✓ {correct}/{n} strict | ~ {partial}/{n} DeepSeek-top | ✗ {wrong}/{n} broken")
        if wrong > n // 3:
            emit("")
            emit("⚠ More than 1/3 of instances show broken ranking. "
                 "This suggests rubric is measuring something other than quality. "
                 "Review specific judge verdicts on ✗ rows.")
    else:
        emit(f"Skipped (need all three tiers: {tiers}; got {list(judged_sources.keys())})")
    emit("")

    # ─── Q3: per-criterion flip rate ───────────────────────────────────

    emit("## Q3. Per-criterion flip rate (which judges are useless)")
    emit("")
    emit("For each (instance, criterion), across K rollouts of a given model:")
    emit("- `all-pass`: every rollout scored YES → criterion is trivially satisfied (on this model)")
    emit("- `all-fail`: every rollout scored NO → criterion is trivially missed (on this model)")
    emit("- `mixed`: at least one flip → criterion is **actually discriminating**")
    emit("")
    emit("Criteria that are `all-pass` or `all-fail` on SFT contribute 0 advantage in GRPO. "
         "Fix: either tighten to make them flippable, or drop them.\n")

    # Aggregate across all instances×criterions for SFT
    for tag in judged_sources:
        if tag != "sft":
            continue
        all_pass = mixed = all_fail = unknown = 0
        problem_criteria = []
        for inst_id, inst in instances_by_id.items():
            for r in inst["rubric"]:
                rid = r["id"]
                vs = verdicts[tag][inst_id].get(rid, [])
                if not vs:
                    continue
                yes_n = sum(1 for v in vs if v == "YES")
                no_n = sum(1 for v in vs if v == "NO")
                unk_n = sum(1 for v in vs if v not in ("YES", "NO"))
                if unk_n > 0 and yes_n == 0 and no_n == 0:
                    unknown += 1
                elif yes_n == len(vs):
                    all_pass += 1
                    problem_criteria.append((inst_id, rid, r["type"], "all-pass", r["criterion"][:80]))
                elif no_n == len(vs):
                    all_fail += 1
                    problem_criteria.append((inst_id, rid, r["type"], "all-fail", r["criterion"][:80]))
                else:
                    mixed += 1
        total = all_pass + all_fail + mixed + unknown
        emit(f"### tag={tag}")
        emit(f"- Total (instance,criterion) pairs: {total}")
        emit(f"- **mixed** (discriminating): {mixed}  ({mixed/total*100:.0f}%)")
        emit(f"- all-pass (trivial): {all_pass}")
        emit(f"- all-fail (trivial): {all_fail}")
        emit(f"- unknown (judge failed): {unknown}")
        emit("")
        if problem_criteria:
            emit("Sample trivial criteria (first 15):")
            emit("| instance | rid | type | pattern | criterion |")
            emit("|---|---|---|---|---|")
            for row in problem_criteria[:15]:
                emit("| " + " | ".join(str(x) for x in row) + " |")
        emit("")

    # ─── Final verdict ─────────────────────────────────────────────────

    emit("## Verdict summary")
    emit("")
    if "sft" in judged_sources:
        sft_zero = zero_var_per_tag["sft"]
        n = len(instances_by_id)
        if sft_zero <= n // 4:
            emit(f"- ✅ SFT variance OK: only {sft_zero}/{n} instances have std=0. "
                 "RL should produce non-trivial gradients on most prompts.")
        else:
            emit(f"- ⚠ SFT variance weak: {sft_zero}/{n} instances have std=0. "
                 "Over 25% of RL budget would be wasted. Consider tightening rubric "
                 "or adding harder scenarios.")

    if all(t in judged_sources for t in tiers):
        correct_rate = correct / len(instances_by_id)
        if correct_rate >= 0.7:
            emit(f"- ✅ Tier ranking solid: {correct}/{n} strict correct. "
                 "Rubric reliably measures quality direction.")
        elif correct_rate >= 0.5:
            emit(f"- ⚠ Tier ranking mixed: only {correct}/{n} strict correct. "
                 "Most rubrics point the right way but some are noisy.")
        else:
            emit(f"- ✗ Tier ranking broken: {correct}/{n} strict correct. "
                 "Rubric is not a reliable quality signal. Fix before RL.")

    emit("")

    # Write out
    text = "\n".join(out_lines)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Report written: {args.output}")
    else:
        print(text)


if __name__ == "__main__":
    main()
