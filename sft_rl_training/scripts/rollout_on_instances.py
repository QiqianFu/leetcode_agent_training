"""Roll out a model on each RL instance with per-rollout isolated workspace.

For each instance:
  - Create a fresh tmpdir
  - Pre-populate: L1 (LeetCode.md), L2 (user_memory.md), L3 (.memories/ + DB),
    current_problem's user_code file
  - Build agent with conversation_prefix injected, call agent.chat(user_message)
  - Repeat K times
  - Dump full message trajectory per rollout

Usage:
  python rollout_on_instances.py \\
    --instances /path/to/v1.jsonl \\
    --base-url https://api.deepseek.com \\
    --model-name deepseek-chat \\
    --api-key $DEEPSEEK_API_KEY \\
    --k 4 \\
    --output /path/to/rollouts_deepseek.jsonl \\
    [--limit N] [--start K] [--tag deepseek]

Run the script once per model. Output tag helps aggregate across models later.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

LEETCODE_SRC = Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Env / import guard ──────────────────────────────────────────────────
# lc.config reads env vars at import time, so set them BEFORE importing lc.

def configure_env(base_url: str, model_name: str, api_key: str):
    os.environ["DEEPSEEK_BASE_URL"] = base_url
    os.environ["DEEPSEEK_MODEL"] = model_name
    os.environ["DEEPSEEK_API_KEY"] = api_key


# ─── Headless patches ────────────────────────────────────────────────────

def apply_headless_patches():
    """Match the patches from train_grpo.py and generate_trajectories.py."""
    import lc.ui
    import lc.agent as lc_agent
    import lc.tool_impl.problems as lc_problems

    lc.ui.flush_stdin = lambda: None
    lc_agent.flush_stdin = lambda: None

    def _stub_arrow_select(choices, load_more=None):
        if not choices:
            return None
        return random.choice(choices)[1]

    lc.ui.arrow_select = _stub_arrow_select
    lc_problems.arrow_select = _stub_arrow_select


# ─── Workspace setup ─────────────────────────────────────────────────────

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slug(title: str) -> str:
    s = _SLUG_RE.sub("_", title.lower()).strip("_")
    return s or "untitled"


def setup_workspace(tmpdir: Path, instance: dict) -> Path:
    """Pre-populate tmpdir from instance.workspace_state. Returns DB path."""
    tmpdir.mkdir(parents=True, exist_ok=True)
    (tmpdir / ".memories").mkdir(exist_ok=True)

    sa = instance.get("system_additions", {})
    if sa.get("L1_preferences"):
        (tmpdir / "LeetCode.md").write_text(sa["L1_preferences"], encoding="utf-8")
    # L2 file is in tmpdir; we patch USER_MEMORY_PATH to point here
    user_mem_path = tmpdir / "user_memory.md"
    if sa.get("L2_user_memory"):
        user_mem_path.write_text(sa["L2_user_memory"], encoding="utf-8")

    db_path = tmpdir / "leetcode.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS problem_memories (
            problem_id    INTEGER PRIMARY KEY,
            title         TEXT NOT NULL,
            difficulty    TEXT,
            tags          TEXT,
            memory_file   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS session (
            key           TEXT PRIMARY KEY,
            value         TEXT
        );
    """)

    ws = instance.get("workspace_state", {})
    for p in ws.get("solved_problems", []):
        slug = _slug(p["title"])
        mem_rel = f".memories/{p['id']}_{slug}.md"
        (tmpdir / mem_rel).write_text(p.get("memory") or "", encoding="utf-8")
        conn.execute(
            """INSERT OR REPLACE INTO problem_memories
               (problem_id, title, difficulty, tags, memory_file)
               VALUES (?, ?, ?, ?, ?)""",
            (p["id"], p["title"], "", p.get("category", ""), mem_rel),
        )

    cur = ws.get("current_problem")
    if cur:
        cat = cur.get("category") or "misc"
        cat_dir = tmpdir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        slug = _slug(cur["title"])
        (cat_dir / f"{cur['id']}_{slug}.py").write_text(
            cur.get("user_code") or "", encoding="utf-8"
        )
        # Also insert into DB so check_problem works
        conn.execute(
            """INSERT OR IGNORE INTO problem_memories
               (problem_id, title, difficulty, tags, memory_file)
               VALUES (?, ?, ?, ?, ?)""",
            (cur["id"], cur["title"], "", cat, f".memories/{cur['id']}_{slug}.md"),
        )
    conn.commit()
    conn.close()
    return db_path


# ─── One rollout ─────────────────────────────────────────────────────────

def run_one_rollout(instance: dict, tmpdir: Path, tag: str, model_name: str,
                    timeout_s: int = 180) -> dict:
    """Execute one ReAct loop on this instance in tmpdir. Returns trajectory dict."""
    import lc.agent as lc_agent
    import lc.config as lc_config
    import lc.db as lc_db
    import lc.tool_impl.subagents as lc_subagents

    db_path = setup_workspace(tmpdir, instance)

    orig_cwd = os.getcwd()
    orig_db_path = lc_db.DB_PATH
    orig_config_db = lc_config.DB_PATH
    orig_user_mem = lc_config.USER_MEMORY_PATH
    orig_sub_user_mem = lc_subagents.USER_MEMORY_PATH
    orig_conn = getattr(lc_db._local, "conn", None)
    orig_llm_client = lc_agent._llm_client

    error_msg = None
    t0 = time.time()

    try:
        os.chdir(tmpdir)

        lc_db.DB_PATH = db_path
        lc_config.DB_PATH = db_path
        user_mem_path = tmpdir / "user_memory.md"
        lc_config.USER_MEMORY_PATH = user_mem_path
        lc_subagents.USER_MEMORY_PATH = user_mem_path
        lc_db._local.conn = None  # force reconnect to this DB

        # Reset LLM client so it picks up current env vars
        lc_agent._llm_client = None

        # Build Agent
        agent = lc_agent.Agent()

        # Pre-fill conversation history from instance prefix.
        # Note: agent.chat() prepends system prompt at call time via
        # _build_system_prompt(), which reads LeetCode.md and user_memory.md
        # from cwd / USER_MEMORY_PATH (which we've patched).
        prefix = instance.get("conversation_prefix") or []
        # Normalize prefix:
        #   1) function.arguments must be a JSON string (codex may emit dict)
        #   2) tool role messages must carry tool_call_id matching the preceding
        #      assistant tool_call id (OpenAI API is strict about this)
        normalized_prefix = []
        last_tool_call_ids: list[str] = []  # queue of expected tool_call_ids
        for idx, m in enumerate(prefix):
            m2 = dict(m)
            if m2.get("tool_calls"):
                fixed_tcs = []
                last_tool_call_ids = []
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
                    last_tool_call_ids.append(tc2["id"])
                m2["tool_calls"] = fixed_tcs
            if m2.get("role") == "tool" and not m2.get("tool_call_id"):
                # Pop matching id in order; fall back to a stub if queue empty
                if last_tool_call_ids:
                    m2["tool_call_id"] = last_tool_call_ids.pop(0)
                else:
                    m2["tool_call_id"] = f"orphan_{idx}"
            normalized_prefix.append(m2)
        agent.messages = normalized_prefix

        # Now run agent.chat() with the user_message. This triggers the full
        # ReAct loop including tool calls. agent.messages gets mutated.
        user_msg = instance["user_message"]

        # We wrap in a budget timer — if something hangs we abort.
        start = time.time()
        try:
            agent.chat(user_msg)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            error_msg = f"agent.chat raised: {type(e).__name__}: {e}"
            logger.warning("  %s", error_msg)

        elapsed = time.time() - start
        if elapsed > timeout_s:
            # Logged but we still return what we have
            error_msg = (error_msg or "") + f" | overrun {elapsed:.0f}s"

        # Extract final messages (everything added since prefix)
        trajectory = agent.messages

    finally:
        # Restore
        os.chdir(orig_cwd)
        lc_db.DB_PATH = orig_db_path
        lc_config.DB_PATH = orig_config_db
        lc_config.USER_MEMORY_PATH = orig_user_mem
        lc_subagents.USER_MEMORY_PATH = orig_sub_user_mem
        # Close the per-rollout DB conn we opened
        rollout_conn = getattr(lc_db._local, "conn", None)
        if rollout_conn is not None and rollout_conn is not orig_conn:
            try:
                rollout_conn.close()
            except Exception:
                pass
        lc_db._local.conn = orig_conn
        lc_agent._llm_client = orig_llm_client

    return {
        "tag": tag,
        "model": model_name,
        "instance_id": instance["id"],
        "label": instance["label"],
        "user_message": instance["user_message"],
        "trajectory": trajectory,
        "prefix_len": len(instance.get("conversation_prefix") or []),
        "elapsed_s": round(time.time() - t0, 2),
        "error": error_msg,
    }


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True, type=Path)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tag", default=None,
                    help="Label for this run (e.g. 'base','sft','deepseek'). "
                         "Defaults to model_name.")
    ap.add_argument("--limit", type=int, default=None, help="Limit to N instances")
    ap.add_argument("--start", type=int, default=0, help="Start index into instances")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: must provide --api-key or set DEEPSEEK_API_KEY env var",
              file=sys.stderr)
        sys.exit(1)

    configure_env(args.base_url, args.model_name, api_key)
    apply_headless_patches()

    instances = [json.loads(l) for l in open(args.instances)]
    instances = instances[args.start:]
    if args.limit is not None:
        instances = instances[:args.limit]

    tag = args.tag or args.model_name
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"rollout: {len(instances)} instances × K={args.k} × tag={tag}")
    logger.info(f"  base_url={args.base_url}  model={args.model_name}")
    logger.info(f"  output={args.output}")

    # Append mode so resumption via --start doesn't clobber earlier output
    mode = "a" if args.start > 0 and args.output.exists() else "w"
    t_start = time.time()
    n_ok, n_err = 0, 0

    with open(args.output, mode, encoding="utf-8") as f:
        for i, inst in enumerate(instances):
            idx = args.start + i
            logger.info(f"[{idx+1}/{args.start + len(instances)}] {inst['id']} — {inst['label']}")
            for k in range(args.k):
                # Fresh tmpdir per rollout (tool side-effects don't leak)
                random.seed(args.seed + idx * 1000 + k)
                tmp = Path(tempfile.mkdtemp(prefix=f"lc_roll_{inst['id']}_k{k}_"))
                try:
                    result = run_one_rollout(inst, tmp, tag, args.model_name)
                    result["rollout_idx"] = k
                    result["seed"] = args.seed + idx * 1000 + k
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    if result.get("error"):
                        n_err += 1
                        logger.info(f"  k={k} ERR: {result['error'][:120]}")
                    else:
                        n_ok += 1
                        n_msgs = len(result["trajectory"])
                        logger.info(f"  k={k} ok  {result['elapsed_s']:>6.1f}s  msgs={n_msgs}")
                finally:
                    shutil.rmtree(tmp, ignore_errors=True)

    total = time.time() - t_start
    logger.info(f"Done. ok={n_ok} err={n_err} total_time={total:.0f}s  →  {args.output}")


if __name__ == "__main__":
    main()
