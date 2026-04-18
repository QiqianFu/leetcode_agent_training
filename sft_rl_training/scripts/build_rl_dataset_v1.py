"""Convert RL instance jsonl (Exp-012 v1) into a GRPO-ready parquet dataset.

Output columns:
  - prompt: list[dict] — [system_with_L1_L2, ...conversation_prefix_normalized, user_message]
  - instance_id: str — key for env.reset() and reward_func to look up rubric/workspace

The env and reward functions load v1.jsonl once at module init and look up instances
by id. This keeps the parquet small and avoids schema trouble with nested dicts.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

LEETCODE_SRC = Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_system_message(system_additions: dict) -> dict:
    """Replicate lc.agent.Agent._build_system_prompt() by baking L1/L2 into the
    static SYSTEM_PROMPT. We must do it here (not dynamically) because TRL
    feeds the dataset prompt verbatim to the model during rollout.
    """
    # Import only after path inject; keeps this script self-contained.
    from lc.agent import SYSTEM_PROMPT

    parts = [SYSTEM_PROMPT]
    l1 = (system_additions or {}).get("L1_preferences", "").strip()
    l2 = (system_additions or {}).get("L2_user_memory", "").strip()
    if l1:
        parts.append(
            f"\n\n## 用户自定义指令 (LeetCode.md)\n以下是用户的自定义指令，你必须遵守：\n\n{l1}"
        )
    if l2:
        parts.append(
            f"\n\n## 用户偏好记忆\n以下是你之前记录的用户偏好，请参考：\n\n{l2}"
        )
    return {"role": "system", "content": "".join(parts)}


def normalize_prefix(prefix: list) -> list:
    """Mirror rollout_on_instances.run_one_rollout's normalization:
    1) function.arguments must be a JSON string
    2) tool role messages must carry tool_call_id matching preceding assistant
    """
    normalized = []
    last_ids: list[str] = []
    for idx, m in enumerate(prefix or []):
        m2 = dict(m)
        if m2.get("tool_calls"):
            fixed_tcs = []
            last_ids = []
            for tci, tc in enumerate(m2["tool_calls"]):
                tc2 = dict(tc)
                if not tc2.get("id"):
                    tc2["id"] = f"call_{idx}_{tci}"
                tc2.setdefault("type", "function")
                fn = dict(tc2.get("function") or {})
                args = fn.get("arguments")
                if not isinstance(args, str):
                    fn["arguments"] = json.dumps(args or {}, ensure_ascii=False)
                tc2["function"] = fn
                fixed_tcs.append(tc2)
                last_ids.append(tc2["id"])
            m2["tool_calls"] = fixed_tcs
        if m2.get("role") == "tool" and not m2.get("tool_call_id"):
            if last_ids:
                m2["tool_call_id"] = last_ids.pop(0)
            else:
                m2["tool_call_id"] = f"orphan_{idx}"
        normalized.append(m2)
    return normalized


def build_prompt(instance: dict) -> list:
    system_msg = build_system_message(instance.get("system_additions") or {})
    prefix = normalize_prefix(instance.get("conversation_prefix") or [])
    user_msg = {"role": "user", "content": instance["user_message"]}
    return [system_msg] + prefix + [user_msg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", default="/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.jsonl")
    ap.add_argument("--output", default="/shared/rsaas/qiqianf2/lc_agent_experiments/rl_instances/v1.parquet")
    args = ap.parse_args()

    rows = []
    with open(args.instances) as f:
        for line in f:
            inst = json.loads(line)
            rows.append({
                "prompt": build_prompt(inst),
                "instance_id": inst["id"],
            })

    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    logger.info(f"Wrote {len(df)} rows → {args.output}")

    # Sanity: print summary of row 0 and row 19
    for i in [0, len(df) - 1]:
        r = df.iloc[i]
        sys_head = r["prompt"][0]["content"][:80].replace("\n", " ")
        user_msg = r["prompt"][-1]["content"][:80]
        logger.info(f"  [{i}] id={r['instance_id']} prompt_msgs={len(r['prompt'])} "
                    f"sys_head='{sys_head}...' user='{user_msg}'")


if __name__ == "__main__":
    main()
