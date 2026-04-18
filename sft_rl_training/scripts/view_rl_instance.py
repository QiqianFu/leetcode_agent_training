"""Pretty-print a single RL instance from rl_instances/v1.jsonl.

Usage:
    python view_rl_instance.py <jsonl_path> <id_or_index>
    python view_rl_instance.py rl_instances/v1.jsonl rl_inst_0008
    python view_rl_instance.py rl_instances/v1.jsonl 7        # 0-indexed

Flags:
    --raw      dump full JSON instead of pretty view
    --prefix   also show conversation_prefix (skipped by default)
"""

import argparse
import json
import sys
from pathlib import Path


def load(path: Path, key: str) -> dict:
    """Look up instance by id string or by 0-based index."""
    records = [json.loads(l) for l in open(path)]
    if key.isdigit():
        return records[int(key)]
    for r in records:
        if r["id"] == key:
            return r
    raise KeyError(f"{key} not found in {path}")


def pretty(inst: dict, show_prefix: bool = False) -> None:
    bar = "=" * 78
    print(bar)
    print(f"{inst['id']}    {inst['label']}")
    print(bar)
    ax = inst["axes"]
    print(f"axes:  intent={ax['intent_clarity']}  stage={ax['session_stage']}  "
          f"prefs={ax['preference_strength']}  ws={ax['workspace_richness']}  "
          f"adv={ax['adversarial']}")
    print()

    sa = inst["system_additions"]
    if sa.get("L1_preferences"):
        print("── L1 (LeetCode.md, 强制遵守) ──")
        print(sa["L1_preferences"].rstrip())
        print()
    if sa.get("L2_user_memory"):
        print("── L2 (用户画像/偏好记忆) ──")
        print(sa["L2_user_memory"].rstrip())
        print()

    ws = inst["workspace_state"]
    solved = ws.get("solved_problems", [])
    if solved:
        print(f"── workspace: {len(solved)} 道已做过的题（L3 记忆存在）──")
        for p in solved:
            print(f"  #{p['id']:<4} {p['title']:35s} [{p['category']}]")
        print()
    cur = ws.get("current_problem")
    if cur:
        print(f"── 正在做的题 ──")
        print(f"  #{cur['id']} {cur['title']} ({cur['category']})")
        if cur.get("user_code"):
            print("  user_code:")
            for ln in cur["user_code"].splitlines():
                print(f"    {ln}")
        print()

    if show_prefix and inst.get("conversation_prefix"):
        print(f"── conversation_prefix ({len(inst['conversation_prefix'])} msgs) ──")
        for m in inst["conversation_prefix"]:
            role = m.get("role", "?")
            if role == "tool":
                name = m.get("name", "?")
                content = (m.get("content") or "")[:200]
                print(f"  [tool:{name}] {content}")
            elif role == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    fn = tc.get("function", {})
                    print(f"  [assistant→tool] {fn.get('name')}({fn.get('arguments', '')[:100]})")
                if m.get("content"):
                    print(f"  [assistant] {m['content'][:200]}")
            else:
                content = (m.get("content") or "")[:300]
                print(f"  [{role}] {content}")
        print()

    print(f"── 用户这一轮说的话 ──")
    print(f"  {inst['user_message']}")
    print()
    print(f"── scenario_intent（hidden from 被训模型，给 judge 看的）──")
    print(f"  {inst['scenario_intent']}")
    print()
    print("── rubric（judge 的打分判据）──")
    for r in inst["rubric"]:
        tag = {"hard": "[HARD]", "soft": "[soft]", "anti": "[ANTI]"}.get(r["type"], f"[{r['type']}]")
        print(f"  {tag} {r['criterion']}")
        if r.get("check_hint"):
            print(f"         ↳ {r['check_hint']}")
    print()
    if inst.get("notes_for_reviewer"):
        print("── notes ──")
        print(f"  {inst['notes_for_reviewer']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("key", type=str, help="id (rl_inst_0008) or 0-based index")
    ap.add_argument("--raw", action="store_true", help="dump raw JSON")
    ap.add_argument("--prefix", action="store_true", help="include conversation_prefix")
    args = ap.parse_args()

    inst = load(args.path, args.key)
    if args.raw:
        json.dump(inst, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        pretty(inst, show_prefix=args.prefix)


if __name__ == "__main__":
    main()
