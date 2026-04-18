"""
GRPO training for LeetCode Agent using TRL.

Uses environment_factory for per-trajectory isolation:
  - Each trajectory gets its own tmpdir workspace + SQLite DB
  - Tool side-effects (file writes, DB updates) don't leak across trajectories
  - Conversation messages are tracked and forwarded to sub-agent tools

TRL's GRPOTrainer multi-turn tool calling loop:
  Model generates tool_call → tool executes → result fed back → model continues
  This repeats up to max_tool_calling_iterations.
  After the trajectory completes, reward_func scores the full conversation.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOTrainer, GRPOConfig

# Add leetcode_agent to path for tool execution
LEETCODE_SRC = Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise RuntimeError(
        "DEEPSEEK_API_KEY must be set in the environment. "
        "Export it in the launcher script (e.g. run_grpo.sh) before invoking this file."
    )
os.environ["TRL_EXPERIMENTAL_SILENCE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Headless patches for leetcode_agent ───
# flush_stdin patches are always needed (no interactive terminal in training)
import lc.ui
lc.ui.flush_stdin = lambda: None
import lc.agent as _am
_am.flush_stdin = lambda: None

# arrow_select: patched with deterministic seed per environment instance (see LeetCodeEnvironment)
# We store the original so we can restore if needed, but in training we always override.
import lc.tool_impl.problems as _pm
import lc.tool_impl.subagents as _sub_mod
import lc.config as _lc_cfg_mod

from lc import db as _lc_db_module
from lc.tools import execute_tool

# OpenAI client for sub-agent tools (shared across environments — stateless HTTP client)
from openai import OpenAI
_lc_client = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ─── Instance lookup (loaded once per process) ───
# RL instances carry workspace_state + rubric. env.reset() and reward_func
# both need these, so we load from a path passed via env var.
_INSTANCES_PATH = os.environ.get("RL_INSTANCES_PATH", "")
_INSTANCES: dict[str, dict] = {}
if _INSTANCES_PATH and os.path.exists(_INSTANCES_PATH):
    with open(_INSTANCES_PATH) as _f:
        for _line in _f:
            _inst = json.loads(_line)
            _INSTANCES[_inst["id"]] = _inst
    logger.info(f"Loaded {len(_INSTANCES)} RL instances from {_INSTANCES_PATH}")

# Import rubric-judge helpers (Exp-013) — lazy so smoke tests don't need them
_JUDGE_MOD = None
def _get_judge_mod():
    global _JUDGE_MOD
    if _JUDGE_MOD is None:
        import judge_rollouts as _jr  # same scripts/ dir
        _JUDGE_MOD = _jr
    return _JUDGE_MOD


# ─── Workspace pre-fill (per-instance) ───
# Adapted from Exp-013 rollout_on_instances.setup_workspace: given an instance
# with workspace_state + system_additions, pre-populate a tmpdir so the agent
# sees a consistent starting state for tool calls (read_memory, find_problem_file, etc.)

import re as _re
_SLUG_RE = _re.compile(r"[^a-z0-9]+")


def _slug(title: str) -> str:
    s = _SLUG_RE.sub("_", title.lower()).strip("_")
    return s or "untitled"


def _populate_workspace(tmpdir: Path, conn: sqlite3.Connection, instance: dict) -> None:
    """Pre-fill tmpdir from instance.system_additions + workspace_state.
    Caller owns `conn` (the env's DB connection); we add rows then return.
    """
    sa = instance.get("system_additions") or {}
    if sa.get("L1_preferences"):
        (tmpdir / "LeetCode.md").write_text(sa["L1_preferences"], encoding="utf-8")
    if sa.get("L2_user_memory"):
        (tmpdir / "user_memory.md").write_text(sa["L2_user_memory"], encoding="utf-8")

    ws = instance.get("workspace_state") or {}
    for p in ws.get("solved_problems") or []:
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
        conn.execute(
            """INSERT OR IGNORE INTO problem_memories
               (problem_id, title, difficulty, tags, memory_file)
               VALUES (?, ?, ?, ?, ?)""",
            (cur["id"], cur["title"], "", cat, f".memories/{cur['id']}_{slug}.md"),
        )
    conn.commit()


# ─── LeetCode Environment for TRL environment_factory ───

class LeetCodeEnvironment:
    """
    Per-trajectory isolated environment for GRPO training.

    Each instance gets:
      - Its own tmpdir as workspace (cwd during tool execution)
      - Its own SQLite DB (no cross-trajectory pollution)
      - Its own conversation message history (for sub-agent tools)
      - Deterministic arrow_select seeded by prompt content
    """

    def __init__(self):
        self._tmpdir: str | None = None
        self._db_path: Path | None = None
        self._original_cwd: str | None = None
        self.messages: list[dict] = []
        self._seed: int = 42
        self._rng = random.Random(self._seed)

    # ─── reset (required by TRL) ───

    def reset(self, **kwargs) -> None:
        """
        Called by TRL before each generation. Receives the full dataset row.
        Sets up an isolated workspace and DB for this trajectory.

        If the dataset row has `instance_id` and that id is in _INSTANCES,
        the tmpdir is pre-populated with that instance's workspace_state
        (L1 LeetCode.md, L2 user_memory.md, L3 .memories/ + solved_problems DB,
        current_problem user_code). This mirrors Exp-013 rollout_on_instances.setup_workspace.
        """
        # Clean up previous trajectory if any
        self._cleanup()

        # Create isolated workspace
        self._tmpdir = tempfile.mkdtemp(prefix="lc_grpo_")
        workspace = Path(self._tmpdir)
        (workspace / ".memories").mkdir(exist_ok=True)

        # Create isolated DB
        self._db_path = workspace / "leetcode.db"
        conn = sqlite3.connect(str(self._db_path))
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
        conn.commit()

        # If this dataset row carries an instance_id, pre-populate the workspace
        # (L1/L2/L3 + current_problem). Conn still open here, we'll close after.
        instance_id = kwargs.get("instance_id")
        if instance_id and instance_id in _INSTANCES:
            _populate_workspace(workspace, conn, _INSTANCES[instance_id])

        conn.close()

        # Seed conversation history with the incoming prompt so sub-agent tools
        # (analyze_and_memorize / find_similar_problems / update_user_memory) can
        # see system + user context when _sub_agent_call reuses main_messages[:-1].
        prompt = kwargs.get("prompt") or []
        self.messages = [dict(m) for m in prompt]

        # Deterministic seed based on prompt content for reproducible arrow_select
        if prompt:
            prompt_str = json.dumps(prompt, ensure_ascii=False, sort_keys=True)
            self._seed = int(hashlib.md5(prompt_str.encode()).hexdigest()[:8], 16)
        else:
            self._seed = 42
        self._rng = random.Random(self._seed)

        return None  # no observation to append to prompt

    def _cleanup(self):
        """Remove temporary workspace. Safe to call during interpreter shutdown."""
        try:
            if self._tmpdir and os.path.exists(self._tmpdir):
                shutil.rmtree(self._tmpdir, ignore_errors=True)
        except Exception:
            # During interpreter teardown, os/shutil may already be torn down.
            pass
        self._tmpdir = None

    def __del__(self):
        try:
            self._cleanup()
        except Exception:
            pass

    # ─── Tool execution helper ───

    def _run_tool(self, name: str, **kwargs) -> str:
        """Execute a leetcode_agent tool within this environment's isolated context."""
        original_cwd = os.getcwd()
        # db.py imports DB_PATH at module level: `from lc.config import DB_PATH`
        # so we must patch db.DB_PATH directly (not lc.config.DB_PATH)
        original_db_path = _lc_db_module.DB_PATH
        original_db_conn = getattr(_lc_db_module._local, "conn", None)
        original_arrow_select = lc.ui.arrow_select
        # USER_MEMORY_PATH is bound into lc.tool_impl.subagents at import time.
        # We must redirect both the source (lc.config) and the consumer (subagents)
        # to a tmpdir-local path, else `tool_update_user_memory` writes to
        # ~/.leetcode_agent/user_memory.md — shared across all envs and the dev's real home.
        original_user_memory_sub = _sub_mod.USER_MEMORY_PATH
        original_user_memory_cfg = _lc_cfg_mod.USER_MEMORY_PATH

        try:
            # Switch to isolated workspace
            os.chdir(self._tmpdir)

            # Point DB to isolated path (patch the reference in db module directly)
            _lc_db_module.DB_PATH = self._db_path
            # Force reconnection to isolated DB
            _lc_db_module._local.conn = None

            # Redirect USER_MEMORY_PATH to tmpdir (see comment above)
            isolated_user_memory = Path(self._tmpdir) / "user_memory.md"
            _sub_mod.USER_MEMORY_PATH = isolated_user_memory
            _lc_cfg_mod.USER_MEMORY_PATH = isolated_user_memory

            # Patch arrow_select with deterministic RNG for this environment
            rng = self._rng

            def _deterministic_select(choices, load_more=None):
                if not choices:
                    return None
                return rng.choice(choices)[1]

            lc.ui.arrow_select = _deterministic_select
            _pm.arrow_select = _deterministic_select

            # Execute tool, passing conversation messages for sub-agent tools
            result = execute_tool(
                name,
                json.dumps(kwargs, ensure_ascii=False),
                _lc_client,
                self.messages,
            )

            # Track this tool call in conversation history so subsequent
            # sub-agent tools (analyze_and_memorize, etc.) have context
            self.messages.append({
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": {"name": name, "arguments": kwargs}}],
            })
            self.messages.append({
                "role": "tool",
                "name": name,
                "content": result,
            })

            return result
        except Exception as e:
            return json.dumps({"error": str(e)})
        finally:
            # Restore global state
            os.chdir(original_cwd)
            _lc_db_module.DB_PATH = original_db_path
            # Close the isolated connection before restoring
            isolated_conn = getattr(_lc_db_module._local, "conn", None)
            if isolated_conn is not None:
                try:
                    isolated_conn.close()
                except Exception:
                    pass
            _lc_db_module._local.conn = original_db_conn
            lc.ui.arrow_select = original_arrow_select
            _pm.arrow_select = original_arrow_select
            _sub_mod.USER_MEMORY_PATH = original_user_memory_sub
            _lc_cfg_mod.USER_MEMORY_PATH = original_user_memory_cfg

    # ─── Tools as methods (auto-registered by TRL) ───
    # TRL discovers all public non-underscore methods except reset().
    # transformers generates tool schemas from function signature + docstring.

    def check_problem(self, problem_id: int) -> str:
        """按题号查询题目信息。返回题目元信息和是否有记忆文件。

        Args:
            problem_id: 题目编号
        """
        return self._run_tool("check_problem", problem_id=problem_id)

    def read_solution(self, problem_id: int = None, file_path: str = None) -> str:
        """读取用户的解题代码文件。可传 file_path 或 problem_id。

        Args:
            problem_id: 题目编号（与 file_path 二选一）
            file_path: 解题文件路径
        """
        kwargs = {}
        if problem_id is not None:
            kwargs["problem_id"] = problem_id
        if file_path is not None:
            kwargs["file_path"] = file_path
        return self._run_tool("read_solution", **kwargs)

    def find_problem_file(self, problem_id: int) -> str:
        """在当前工作区内按题号查找本地解题文件。

        Args:
            problem_id: 题目编号
        """
        return self._run_tool("find_problem_file", problem_id=problem_id)

    def append_solution(self, file_path: str, content: str) -> str:
        """将参考解法追加到用户的解题文件末尾。

        Args:
            file_path: 解题文件路径
            content: 参考解法代码
        """
        return self._run_tool("append_solution", file_path=file_path, content=content)

    def pick_problem(self, tag: str = None, difficulty: str = None) -> str:
        """从 CodeTop 高频题库推荐题目供用户选择。

        Args:
            tag: 标签筛选，如 dp, graph
            difficulty: 难度筛选，如 Easy, Medium, Hard
        """
        kwargs = {}
        if tag:
            kwargs["tag"] = tag
        if difficulty:
            kwargs["difficulty"] = difficulty
        return self._run_tool("pick_problem", **kwargs)

    def search_problem(self, keyword: str) -> str:
        """用英文关键词搜索 LeetCode 题目。

        Args:
            keyword: 英文搜索关键词
        """
        return self._run_tool("search_problem", keyword=keyword)

    def start_problem(self, problem_id: int) -> str:
        """开始做指定题号的 LeetCode 题。

        Args:
            problem_id: 题目编号
        """
        return self._run_tool("start_problem", problem_id=problem_id)

    def read_memory(self, problem_id: int) -> str:
        """读取某道题的记忆文件内容。

        Args:
            problem_id: 题目编号
        """
        return self._run_tool("read_memory", problem_id=problem_id)

    def write_memory(self, problem_id: int, content: str, mode: str = "append") -> str:
        """写入或追加内容到某道题的记忆文件。

        Args:
            problem_id: 题目编号
            content: 要写入的内容
            mode: 写入模式，append 或 overwrite
        """
        return self._run_tool("write_memory", problem_id=problem_id, content=content, mode=mode)

    def web_search(self, query: str, max_results: int = 5) -> str:
        """搜索互联网获取信息。

        Args:
            query: 搜索关键词
            max_results: 返回结果数量
        """
        return self._run_tool("web_search", query=query, max_results=max_results)

    def update_user_memory(self) -> str:
        """更新用户偏好记忆。"""
        return self._run_tool("update_user_memory")

    def find_similar_problems(self, problem_id: int) -> str:
        """查找与当前题目算法思路相似的已做题目。

        Args:
            problem_id: 当前题目编号
        """
        return self._run_tool("find_similar_problems", problem_id=problem_id)

    def analyze_and_memorize(self, problem_id: int) -> str:
        """将当前题目的分析写入记忆文件。

        Args:
            problem_id: 题目编号
        """
        return self._run_tool("analyze_and_memorize", problem_id=problem_id)


# ─── Reward function ───

def reward_func(completions: list[list[dict]], **kwargs) -> list[float]:
    """Rubric-based reward (Exp-014).

    For each (completion, instance_id) pair:
      - Look up instance rubric from _INSTANCES
      - Ask DeepSeek to judge all rubric criteria (reuses judge_rollouts logic)
      - Aggregate: weighted pass rate in [0, 1] (hard/anti 1.5x, soft 1.0x)

    If _INSTANCES is empty or instance_id is missing, falls back to the
    placeholder signal (has-tool-call → 1.0 else 0.0) so pipeline still runs.

    TRL passes dataset columns as kwargs (lists aligned with completions).
    """
    instance_ids = kwargs.get("instance_id") or []

    # Fallback path: no instance lookup available → placeholder signal
    if not _INSTANCES or not instance_ids:
        rewards = []
        for messages in completions:
            has_tc = any(
                m.get("role") == "assistant" and m.get("tool_calls")
                for m in messages
            )
            rewards.append(1.0 if has_tc else 0.0)
        return rewards

    jr = _get_judge_mod()
    rewards = []
    for messages, inst_id in zip(completions, instance_ids):
        inst = _INSTANCES.get(inst_id)
        if inst is None:
            logger.warning(f"reward_func: instance_id={inst_id!r} not found; reward=0")
            rewards.append(0.0)
            continue

        rubric = inst["rubric"]
        # TRL's `completions` for a trajectory is the post-prompt message list
        # (assistant + tool turns). render_trajectory uses prefix_len to slice,
        # so we pass prefix_len=0 to render the whole completion.
        fake_rollout = {"trajectory": messages, "prefix_len": 0}
        prompt = jr.JUDGE_TEMPLATE.format(
            scenario_intent=inst["scenario_intent"],
            L1=(inst["system_additions"].get("L1_preferences") or "(no L1)"),
            L2=(inst["system_additions"].get("L2_user_memory") or "(no L2)"),
            user_message=inst["user_message"],
            trajectory=jr.render_trajectory(fake_rollout),
            rubric_block=jr.build_rubric_block(rubric),
            n_rubric=len(rubric),
        )
        try:
            raw = jr.call_deepseek(_lc_client, prompt)
            verdicts = jr.parse_verdicts(raw, rubric)
            agg = jr.aggregate_score(rubric, verdicts)
            score = float(agg["score"])
        except Exception as e:
            logger.warning(f"reward_func judge failed for {inst_id}: {e}; reward=0")
            score = 0.0
        rewards.append(score)

    return rewards


# ─── Build dataset ───

def build_dataset(data_path: str) -> Dataset:
    """Load prompts from parquet.

    Supports two parquet schemas:
      (old) `prompt` as JSON string — from Exp-002 rl_prompts.parquet
      (new) `prompt` as list[dict] + `instance_id` — from Exp-014 v1.parquet

    Any non-`prompt` columns are passed through verbatim (TRL forwards them
    as kwargs to env.reset() and reward_func).
    """
    import pandas as pd
    import numpy as np

    def _to_py(val):
        """Coerce numpy array of dicts / JSON string / list to a plain list of dicts."""
        if isinstance(val, np.ndarray):
            val = val.tolist()
        if isinstance(val, str):
            val = json.loads(val)
        return [dict(m) for m in val]

    df = pd.read_parquet(data_path)
    records = []
    for _, row in df.iterrows():
        rec = {"prompt": _to_py(row["prompt"])}
        for col in df.columns:
            if col == "prompt":
                continue
            rec[col] = row[col]
        records.append(rec)

    return Dataset.from_list(records)


# ─── Main ───

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_merged_model")
    parser.add_argument("--data_path", default="/shared/rsaas/qiqianf2/lc_agent_experiments/rl_prompts.parquet")
    parser.add_argument("--output_dir", default="/shared/rsaas/qiqianf2/lc_agent_experiments/grpo_trl_exp001")
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="Override num_train_epochs with a fixed step count (for dry-runs). -1 = disabled")
    parser.add_argument("--limit_prompts", type=int, default=-1,
                        help="Truncate dataset to first N prompts (for dry-runs). -1 = disabled")
    parser.add_argument("--num_generations", type=int, default=4,
                        help="GRPO group size")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.04,
                        help="KL penalty coefficient (TRL default 0.04). Larger = more conservative.")
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--max_completion_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=int, default=5)
    args = parser.parse_args()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build dataset
    dataset = build_dataset(args.data_path)
    logger.info(f"Dataset: {len(dataset)} prompts")
    if args.limit_prompts > 0:
        dataset = dataset.select(range(min(args.limit_prompts, len(dataset))))
        logger.info(f"Truncated dataset to {len(dataset)} prompts (dry-run)")

    # Split train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    # LoRA config
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=64,
        lora_alpha=128,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )

    # GRPO config
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=3,
        report_to="none",
        # GRPO specific
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        # Tool calling
        max_tool_calling_iterations=5,
        # Generation
        generation_kwargs={
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
        },
    )

    # Initialize trainer with environment_factory for per-trajectory isolation
    trainer = GRPOTrainer(
        model=args.model_path,
        reward_funcs=reward_func,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        environment_factory=LeetCodeEnvironment,
    )

    logger.info("Starting GRPO training...")
    trainer.train()
    trainer.save_model()
    logger.info(f"Training complete! Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
