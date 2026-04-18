"""Judge rollouts against instance rubrics using DeepSeek.

For each rollout:
  - Look up its instance to fetch rubric
  - Ask DeepSeek to evaluate all rubric criteria in one call (JSON output)
  - Aggregate: weighted pass rate (hard=1.5x, anti=1.5x, soft=1.0x)

Usage:
    python judge_rollouts.py \\
      --rollouts /path/to/rollouts_deepseek.jsonl \\
      --instances /path/to/v1.jsonl \\
      --output /path/to/judged_deepseek.jsonl \\
      [--limit N] [--start K]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

TYPE_WEIGHT = {"hard": 1.5, "anti": 1.5, "soft": 1.0}


# ─── Trajectory rendering ───────────────────────────────────────────────

def render_trajectory(rollout: dict) -> str:
    """Render the post-user-message part of trajectory for the judge.

    rollout.trajectory includes: conversation_prefix + user_message + agent
    ReAct turns. We show the user_message as anchor, then everything after.
    """
    trajectory = rollout["trajectory"]
    prefix_len = rollout.get("prefix_len", 0)
    # Everything from prefix_len onward (the user msg + what agent produced)
    slice_ = trajectory[prefix_len:]

    lines = []
    for m in slice_:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "user":
            lines.append(f"=== USER ===\n{content}")
        elif role == "assistant":
            text_parts = []
            if content:
                text_parts.append(content.strip())
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name", "?")
                args = fn.get("arguments", "")
                if isinstance(args, str) and len(args) > 400:
                    args = args[:400] + "..."
                text_parts.append(f"<tool_call>{name}({args})</tool_call>")
            lines.append(f"=== ASSISTANT ===\n" + "\n".join(text_parts) if text_parts else "=== ASSISTANT (empty) ===")
        elif role == "tool":
            tool_name = m.get("name", "?")
            content_show = content[:500] + ("..." if len(content) > 500 else "")
            lines.append(f"=== TOOL({tool_name}) ===\n{content_show}")
    return "\n\n".join(lines)


# ─── Prompt construction ───────────────────────────────────────────────

JUDGE_SYSTEM = """你是一个严格的 LeetCode 辅导 agent 回复评审员。你收到：
(1) 用户的场景真实意图（scenario_intent，用户本次真正想要什么）
(2) 场景的系统约束（L1 规则 / L2 画像）
(3) agent 的本轮轨迹（包括 tool call 和最终自然语言回复）
(4) 一组 yes/no 判据（rubric），每条独立评估

**统一判据语义（非常重要，不要搞反）**：
- YES = 这条判据**通过**（agent 做到了该做的，或没有发生不该发生的行为）
- NO  = 这条判据**失败**（agent 没做到，或发生了不该发生的行为）

判据类型规则：
- `hard` / `soft`：criterion 描述"agent 应该做 X"。做到 → YES；没做到 → NO
- `anti`：criterion 描述"若 agent 做了不该做的 X → 判否"这类否定表达。
  * 如果 agent **没有**发生该不良行为（该避免的事被避免了）→ YES
  * 如果 agent **发生了**该不良行为 → NO
  * 不要因为 criterion 写了"若...则判否"就默认 NO——先判断条件（X）是否发生

**其他规则**：
- 严格按 criterion 和 check_hint 描述执行，不要自行加戏或发明新判据
- 如果 agent 轨迹为空 / 报错 / 没有实质回复，大部分 hard / soft 判据应判 NO
- 输出必须是合法 JSON 对象，不要任何 markdown 围栏或解释文字之外的内容
"""

JUDGE_TEMPLATE = """\
## Scenario intent
{scenario_intent}

## System constraints
### L1 (LeetCode.md，被训模型必须遵守):
{L1}

### L2 (用户画像/偏好记忆):
{L2}

## User's message this turn
{user_message}

## Agent's full trajectory this turn
{trajectory}

## Rubric criteria
{rubric_block}

## 输出要求
必须输出合法 JSON，格式：
{{
  "r1": {{"verdict": "YES"|"NO", "reason": "1 句中文"}},
  "r2": {{"verdict": "YES"|"NO", "reason": "1 句中文"}},
  ...（共 {n_rubric} 项，key 必须与判据的 id 对应）
}}

只输出 JSON，不要任何其他文字、markdown 围栏、前后说明。
"""


def build_rubric_block(rubric: list) -> str:
    lines = []
    for r in rubric:
        hint = f"   （程序化提示：{r['check_hint']}）" if r.get("check_hint") else ""
        lines.append(f"[{r['id']}] ({r['type']}) {r['criterion']}{hint}")
    return "\n".join(lines)


# ─── DeepSeek call ──────────────────────────────────────────────────────

def call_deepseek(client, prompt: str, max_retries: int = 2) -> str:
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(1.5 * (attempt + 1))
            continue
    raise RuntimeError(f"deepseek judge failed after {max_retries+1} attempts: {last_err}")


def parse_verdicts(raw: str, rubric: list) -> dict:
    """Parse JSON verdicts. Return {rubric_id: {verdict, reason}}."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Try to strip markdown fences
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if fenced:
            obj = json.loads(fenced.group(1))
        else:
            raise
    verdicts = {}
    for r in rubric:
        rid = r["id"]
        item = obj.get(rid)
        if not item:
            verdicts[rid] = {"verdict": "UNKNOWN", "reason": "judge skipped this criterion"}
            continue
        v = str(item.get("verdict", "")).strip().upper()
        if v not in ("YES", "NO"):
            v = "UNKNOWN"
        verdicts[rid] = {"verdict": v, "reason": item.get("reason", "")}
    return verdicts


def aggregate_score(rubric: list, verdicts: dict) -> dict:
    """Weighted pass rate; hard/anti judged missing as failure (conservative)."""
    total_w = 0.0
    got_w = 0.0
    n_yes = n_no = n_unknown = 0
    per_type_hits = {"hard": [0, 0], "soft": [0, 0], "anti": [0, 0]}  # [hit, total]
    for r in rubric:
        w = TYPE_WEIGHT.get(r["type"], 1.0)
        total_w += w
        v = verdicts.get(r["id"], {}).get("verdict", "UNKNOWN")
        pt = per_type_hits.setdefault(r["type"], [0, 0])
        pt[1] += 1
        if v == "YES":
            got_w += w
            pt[0] += 1
            n_yes += 1
        elif v == "NO":
            n_no += 1
        else:
            n_unknown += 1
    score = got_w / total_w if total_w > 0 else 0.0
    return {
        "score": round(score, 3),
        "n_yes": n_yes, "n_no": n_no, "n_unknown": n_unknown,
        "per_type": {k: f"{v[0]}/{v[1]}" for k, v in per_type_hits.items() if v[1] > 0},
    }


# ─── Main ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, type=Path)
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: provide --api-key or set DEEPSEEK_API_KEY", file=sys.stderr)
        sys.exit(1)

    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL, timeout=60)

    # Load instances
    instances_by_id = {json.loads(l)["id"]: json.loads(l)
                       for l in open(args.instances)}
    for _id in list(instances_by_id.keys()):
        instances_by_id[_id] = json.loads(
            open(args.instances).read().splitlines()[list(instances_by_id.keys()).index(_id)]
        )
    # Simpler re-load:
    instances_by_id = {}
    with open(args.instances) as f:
        for line in f:
            d = json.loads(line)
            instances_by_id[d["id"]] = d

    rollouts = [json.loads(l) for l in open(args.rollouts)]
    rollouts = rollouts[args.start:]
    if args.limit is not None:
        rollouts = rollouts[:args.limit]

    logger.info(f"Judging {len(rollouts)} rollouts → {args.output}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.start > 0 and args.output.exists() else "w"

    n_ok, n_err = 0, 0
    t_start = time.time()
    with open(args.output, mode, encoding="utf-8") as f_out:
        for i, r in enumerate(rollouts):
            inst = instances_by_id.get(r["instance_id"])
            if inst is None:
                logger.warning(f"  [{i}] instance {r['instance_id']} not found; skipping")
                continue

            tag = r.get("tag", "?")
            idx = args.start + i
            prompt = JUDGE_TEMPLATE.format(
                scenario_intent=inst["scenario_intent"],
                L1=inst["system_additions"].get("L1_preferences") or "(no L1)",
                L2=inst["system_additions"].get("L2_user_memory") or "(no L2)",
                user_message=inst["user_message"],
                trajectory=render_trajectory(r),
                rubric_block=build_rubric_block(inst["rubric"]),
                n_rubric=len(inst["rubric"]),
            )

            t0 = time.time()
            try:
                raw = call_deepseek(client, prompt)
                verdicts = parse_verdicts(raw, inst["rubric"])
                agg = aggregate_score(inst["rubric"], verdicts)
                n_ok += 1
            except Exception as e:
                logger.warning(f"  [{idx}] {r['instance_id']} k={r.get('rollout_idx')} err: {e}")
                n_err += 1
                verdicts = {rr["id"]: {"verdict": "UNKNOWN", "reason": str(e)[:100]}
                            for rr in inst["rubric"]}
                agg = aggregate_score(inst["rubric"], verdicts)

            dt = time.time() - t0
            logger.info(
                f"[{idx+1}/{args.start + len(rollouts)}] {r['instance_id']} k={r.get('rollout_idx')} "
                f"tag={tag}  score={agg['score']:.3f}  Y/N/?={agg['n_yes']}/{agg['n_no']}/{agg['n_unknown']}  "
                f"{dt:.1f}s"
            )

            out = {
                "tag": tag,
                "instance_id": r["instance_id"],
                "label": r.get("label"),
                "rollout_idx": r.get("rollout_idx"),
                "score": agg["score"],
                "n_yes": agg["n_yes"],
                "n_no": agg["n_no"],
                "n_unknown": agg["n_unknown"],
                "per_type": agg["per_type"],
                "verdicts": verdicts,
                "rollout_error": r.get("error"),
            }
            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            f_out.flush()

    logger.info(f"Done. ok={n_ok} err={n_err} total={time.time()-t_start:.0f}s  →  {args.output}")


if __name__ == "__main__":
    main()
