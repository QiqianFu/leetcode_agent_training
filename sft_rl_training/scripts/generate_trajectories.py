"""
Generate SFT training trajectories by running leetcode_agent in headless mode
with DeepSeek self-play (DeepSeek-User + DeepSeek-Agent).

Usage:
    # Run a few test trajectories
    python generate_trajectories.py --n 5 --mode structured --output test_traj.jsonl

    # Full generation
    python generate_trajectories.py --n 400 --mode structured --output structured.jsonl
    python generate_trajectories.py --n 100 --mode free --free_intents free_intents.txt --output free.jsonl

    # Clarify mode: user opens vaguely, agent must ask back, user then specifies.
    # Rest of the loop (code writing → ending → analyze_and_memorize) is unchanged.
    python generate_trajectories.py --n 70 --mode clarify --output clarify_trajectories.jsonl
"""
from __future__ import annotations

import json
import os
import random
import sys
import time
import logging
import argparse
from pathlib import Path

# ─── Setup: patch leetcode_agent imports ───
LEETCODE_SRC = Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))

if not os.environ.get("DEEPSEEK_API_KEY"):
    raise RuntimeError("DEEPSEEK_API_KEY not set — source project-root .env first (cp .env.example .env)")
os.environ["MAX_AGENT_HISTORY_MESSAGES"] = "500"

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USER_SIM_CLIENT = OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

# ─── Monkey-patch for headless mode ───

def _headless_arrow_select(choices, load_more=None):
    """Auto-select a random choice instead of showing terminal UI."""
    if not choices:
        return None
    _, value = random.choice(choices)
    logger.info(f"  [headless] auto-selected: {value}")
    return value

import lc.ui as _ui_module
_ui_module.arrow_select = _headless_arrow_select
_ui_module.flush_stdin = lambda: None
import lc.ui
lc.ui.arrow_select = _headless_arrow_select
lc.ui.flush_stdin = lambda: None

from lc import db
from lc.agent import Agent, SYSTEM_PROMPT, DEEPSEEK_MODEL
from lc.tool_defs import TOOLS as TOOL_SCHEMAS

import lc.tool_impl.problems as _problems_module
_problems_module.arrow_select = _headless_arrow_select

import lc.agent as _agent_module
_agent_module.flush_stdin = lambda: None

# Patch _call_model_once: non-streaming to avoid Rich Live blocking
def _headless_call_model_once(self, messages):
    import time as _time
    t0 = _time.time()
    messages = self._sanitize_messages(messages)
    resp = self.client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        tools=TOOL_SCHEMAS,
        stream=False,
        temperature=0.3,
        max_tokens=4096,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    tool_calls = []
    if choice.message.tool_calls:
        tool_calls = [
            {"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
            for tc in choice.message.tool_calls
        ]
    usage = {}
    if resp.usage:
        usage = {
            "prompt": resp.usage.prompt_tokens,
            "completion": resp.usage.completion_tokens,
            "total": resp.usage.total_tokens,
        }
    elapsed = _time.time() - t0
    logger.info(f"  [LLM] {elapsed:.1f}s, {usage.get('total', '?')} tokens, tools={[t['name'] for t in tool_calls] or 'none'}")
    return content, tool_calls, usage

Agent._call_model_once = _headless_call_model_once


# ─── Initialize DB for workspace ───

def init_workspace_db(workspace_dir: Path):
    db.init_db()
    categories = ["dp", "greedy", "binary_search", "two_pointers", "dfs_bfs",
                  "sorting", "stack_queue", "tree", "graph", "design", "math_bit", "string"]
    for cat in categories:
        cat_dir = workspace_dir / cat
        if not cat_dir.is_dir():
            continue
        for py_file in cat_dir.glob("*.py"):
            parts = py_file.stem.split("_", 1)
            if parts[0].isdigit():
                pid = int(parts[0])
                title = parts[1].replace("_", " ").title() if len(parts) > 1 else ""
                memory_file = str(workspace_dir / ".memories" / f"{py_file.stem}.md")
                if db.get_memory(pid) is None:
                    db.upsert_memory(pid, title, memory_file, "", "")
                    logger.info(f"  Registered existing problem: {pid} {title}")


def get_existing_problems(workspace_dir: Path) -> list[int]:
    ids = []
    for py_file in workspace_dir.rglob("*.py"):
        if py_file.parent.name.startswith(".") or "src" in str(py_file):
            continue
        parts = py_file.stem.split("_", 1)
        if parts[0].isdigit():
            ids.append(int(parts[0]))
    return sorted(set(ids))


# ─── Step 1: User opening message ───

TOPICS_ZH = ["动态规划", "树", "图", "数组", "链表", "栈", "堆", "二分查找",
             "滑动窗口", "回溯", "DFS", "BFS", "贪心", "双指针", "字符串", "设计"]

# Common LeetCode IDs for "我要做第x题" — random from a wide pool
COMMON_PROBLEM_IDS = list(range(1, 800))  # Wide range, truly random


def generate_step1_message(variant: str) -> str:
    if variant == "random":
        resp = USER_SIM_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": "你是一个 LeetCode 用户，你想让刷题助手帮你随机推荐一道题。"
                           "用一句简短自然的中文表达这个意图。不要加引号，直接输出这句话。"
                           "例如：来一道题吧、帮我选一道题做、今天做什么好、开始刷题、推荐一道题",
            }],
            temperature=1.0,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip().strip('"\'')
    elif variant == "specific":
        pid = random.choice(COMMON_PROBLEM_IDS)
        resp = USER_SIM_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"你是一个 LeetCode 用户，你想做第 {pid} 题。"
                           f"用一句简短自然的中文表达这个意图，必须包含题号 {pid}。"
                           f"不要加引号，直接输出这句话。"
                           f"例如：我要做第{pid}题、开始做{pid}题、来做LeetCode {pid}",
            }],
            temperature=1.0,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip().strip('"\'')
    elif variant == "topic":
        topic = random.choice(TOPICS_ZH)
        resp = USER_SIM_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "user",
                "content": f"你是一个 LeetCode 用户，你想做一道{topic}类型的题。"
                           f"用一句简短自然的中文表达这个意图。不要加引号，直接输出这句话。"
                           f"例如：来一道{topic}的题、我想练习{topic}、给我推荐一道{topic}方向的题",
            }],
            temperature=1.0,
            max_tokens=50,
        )
        return resp.choices[0].message.content.strip().strip('"\'')
    else:
        raise ValueError(f"Unknown variant: {variant}")


# ─── Clarify mode: vague opener + concrete follow-up ───

# Deliberately vague openers — chosen so that a well-behaved agent should ask back
# rather than blindly calling pick_problem / search_problem. Mixed with LLM-generated
# variants for diversity.
AMBIGUOUS_SEEDS = [
    # Empirically validated: these consistently trigger agent clarification.
    "你看看怎么办",
    "给我点建议",
    "帮我看看",
    "帮帮我",
    "帮我规划一下",
    "嗯…你说呢",
    "我不知道做啥",
    "我想进步",
    "我想练练",
    "你帮我想想",
    "现在咋办",
    "怎么整",
    "我有点迷茫",
    "你觉得呢",
]


def generate_vague_first_message() -> str:
    """Generate an intentionally vague opener. 70% seed, 30% LLM for diversity.

    Avoid 'arrange/recommend' verbs — DeepSeek-agent reliably interprets them as
    "give me a problem" and directly calls pick_problem, which defeats the purpose.
    """
    if random.random() < 0.7:
        return random.choice(AMBIGUOUS_SEEDS)
    resp = USER_SIM_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": (
                "你是一个刚打开 LeetCode 刷题助手的用户，心里有点困惑、不太清楚自己想做什么。"
                "你想让助手反过来问你才好讲清楚需求。"
                "请生成一句极简短、极模糊的中文开场白，要求：\n"
                "1. 长度最多 10 个字\n"
                "2. 不要出现 '推荐'、'安排'、'来一道'、'给我题'、'开始刷题' 这类动词——"
                "这些词会让助手以为你想直接做题\n"
                "3. 最好是寻求建议/犹豫/求助/表达困惑的感觉\n"
                "4. 不要加引号，不要解释\n"
                "参考风格：你看看怎么办、给我点建议、帮帮我、我不知道做啥、嗯…你说呢、怎么整"
            ),
        }],
        temperature=1.2,
        max_tokens=30,
    )
    text = resp.choices[0].message.content.strip().strip('"\'')
    # Guardrails: reject outputs that look action-oriented or too long.
    bad_markers = [
        "第", "题号", "动态规划", "贪心", "二分", "DP", "dp",
        "开始刷题", "来一道", "推荐", "安排", "给我题", "给我一道", "找一道",
    ]
    if any(m in text for m in bad_markers) or len(text) > 30:
        return random.choice(AMBIGUOUS_SEEDS)
    return text


def generate_concrete_followup(assistant_clarify_reply: str) -> str:
    """Given agent's clarification question, ask DeepSeek-User to give a concrete
    answer. Bias toward variety (topic / specific id / random pick / difficulty)."""
    # Pre-seed a concrete intent to prevent DeepSeek from drifting back to vague.
    variant = random.choices(
        ["random", "specific", "topic", "difficulty"],
        weights=[1, 1, 2, 1],
        k=1,
    )[0]
    if variant == "random":
        hint = "你真的没啥偏好，让助手随便推荐一道题就行"
    elif variant == "specific":
        pid = random.choice(COMMON_PROBLEM_IDS)
        hint = f"你想做第 {pid} 题（回复里必须包含题号 {pid}）"
    elif variant == "topic":
        topic = random.choice(TOPICS_ZH)
        hint = f"你想练「{topic}」方向的题"
    else:
        difficulty = random.choice(["easy", "medium", "hard", "简单", "中等", "困难"])
        hint = f"你想做一道 {difficulty} 难度的题"

    resp = USER_SIM_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": (
                f"你在和 LeetCode 刷题助手对话。你上一句话说得很模糊，助手刚刚反过来问你：\n\n"
                f"【助手问】{assistant_clarify_reply}\n\n"
                f"你的真实需求是：{hint}。\n"
                f"请自然地回答助手这个问题，把需求讲清楚。要求：\n"
                f"1. 一句话，简短口语，像真人聊天\n"
                f"2. 不要加引号，不要解释\n"
                f"3. 如果助手问的内容和你的需求不完全对齐，也要把你的需求讲出来，不要被问题带偏"
            ),
        }],
        temperature=0.9,
        max_tokens=100,
    )
    return resp.choices[0].message.content.strip().strip('"\''), variant


# ─── Step 2: Simulate user writing code ───

def simulate_user_code(solution_file: Path, quality: str) -> str:
    if not solution_file.exists():
        return ""

    original = solution_file.read_text(encoding="utf-8")

    quality_instructions = {
        "correct": "写一个完整且正确的解法。",
        "buggy": "写一个看起来合理但有一个小 bug 的解法（比如边界条件错误、off-by-one、忘记处理空输入等）。",
        "partial": "只写一半代码，留一个 TODO 或 pass，表示用户做到一半放弃了。",
    }

    resp = USER_SIM_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": f"你是一个正在做 LeetCode 的用户。以下是题目文件：\n\n```python\n{original}\n```\n\n"
                       f"请{quality_instructions[quality]}\n\n"
                       f"只输出完整的 Python 文件内容（包括原有注释和题目描述），不要加任何解释。",
        }],
        temperature=0.7,
        max_tokens=2048,
    )
    code = resp.choices[0].message.content.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])

    solution_file.write_text(code, encoding="utf-8")
    logger.info(f"  [user-sim] wrote {quality} code to {solution_file.name}")
    return code


# ─── Step 2.5 (optional): User asks "is this right?" ───

def generate_midstep_question() -> str:
    """DeepSeek generates a natural mid-step question from the user."""
    resp = USER_SIM_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{
            "role": "user",
            "content": "你是一个正在做 LeetCode 的用户，你刚写完代码想让助手帮你看看。"
                       "用一句简短自然的中文表达这个意图。不要加引号，直接输出这句话。"
                       "例如：这样写对吗、你帮我看看这个思路对不对、我写了一版你看看有没有问题、这个方法行吗",
        }],
        temperature=1.0,
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip().strip('"\'')


# ─── Step 3: Ending message ───

def generate_ending_message(quality: str) -> str:
    """DeepSeek dynamically generates a natural ending message."""
    if quality in ("correct", "buggy"):
        prompt = ("你是一个刚做完 LeetCode 题目的用户，想让助手总结一下。"
                  "用一句简短自然的中文表达。不要加引号，直接输出这句话。"
                  "例如：我做完了、好了你看看吧、搞定了帮我总结下、ok做完了、完成了看看我的代码")
    else:  # partial
        prompt = ("你是一个做 LeetCode 做到一半不想继续的用户，想让助手帮忙看看。"
                  "用一句简短自然的中文表达。不要加引号，直接输出这句话。"
                  "例如：我只能做这么多了、大概就这样了你看看、做不下去了、算了不会做了你帮我看看")

    resp = USER_SIM_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=50,
    )
    return resp.choices[0].message.content.strip().strip('"\'')


# ─── Find the solution file created by start_problem ───

def find_latest_solution_file(workspace_dir: Path, before_files: set[str]) -> Path | None:
    for py_file in workspace_dir.rglob("*.py"):
        if str(py_file) not in before_files and py_file.parent.name != "src":
            return py_file
    return None


# ─── Check if trajectory has L3 memory written ───

def trajectory_has_l3_memory(messages: list[dict]) -> bool:
    """Check if analyze_and_memorize was called and succeeded."""
    for msg in messages:
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if '"l3_written": true' in content or '"l3_written":true' in content:
                return True
    return False


# ─── Core: run one structured trajectory ───

def run_structured_trajectory(workspace_dir: Path, existing_ids: list[int], traj_id: int) -> dict | None:
    """Generate one structured trajectory.

    Flow:
      Step 1: User asks for a problem (random/specific/topic)
              → Agent: pick/search → start_problem → find_similar_problems
      Step 2: User "writes code" (simulated by DeepSeek)
      Step 2.5 (1/3 chance): User asks "is this right?" → Agent reads & gives feedback
      Step 3: User ending message (DeepSeek generated)
              → Agent: read_solution → feedback → analyze_and_memorize
    """
    logger.info(f"=== Trajectory {traj_id} (structured) ===")

    step1_variant = random.choices(["random", "specific", "topic"], weights=[1, 1, 1], k=1)[0]
    code_quality = random.choices(["correct", "buggy", "partial"], weights=[1, 1, 1], k=1)[0]
    has_midstep = random.random() < 1/3  # 1/3 chance of mid-step question

    logger.info(f"  step1={step1_variant}, code={code_quality}, midstep={has_midstep}")

    before_files = {str(f) for f in workspace_dir.rglob("*.py") if f.parent.name != "src"}

    agent = Agent()

    # Step 1: User opens
    msg1 = generate_step1_message(step1_variant)
    logger.info(f"  [user] {msg1}")

    try:
        agent.chat(msg1)
    except Exception as e:
        logger.warning(f"  Step 1 failed: {e}")
        return None

    # Find newly created solution file
    solution_file = find_latest_solution_file(workspace_dir, before_files)

    # Step 2: Simulate user writing code
    if solution_file:
        simulate_user_code(solution_file, code_quality)
    else:
        logger.info("  No solution file found, skipping code simulation")

    # Step 2.5 (optional): User asks "is this right?"
    if has_midstep and solution_file:
        mid_msg = generate_midstep_question()
        logger.info(f"  [user] (midstep) {mid_msg}")
        try:
            agent.chat(mid_msg)
        except Exception as e:
            logger.warning(f"  Midstep failed: {e}")

    # Step 3: User ending message
    msg_end = generate_ending_message(code_quality)
    logger.info(f"  [user] {msg_end}")

    try:
        agent.chat(msg_end)
    except Exception as e:
        logger.warning(f"  Step 3 failed: {e}")

    # Check if L3 memory was written
    has_l3 = trajectory_has_l3_memory(agent.messages)

    trajectory = {
        "id": f"structured_{traj_id:04d}",
        "type": "structured",
        "config": {
            "step1": step1_variant,
            "code_quality": code_quality,
            "has_midstep": has_midstep,
        },
        "has_l3_memory": has_l3,
        "messages": agent.messages,
        "tools": TOOL_SCHEMAS,
    }
    return trajectory


# ─── Core: run one clarify trajectory ───

def _new_assistant_made_tool_call(messages: list[dict], before_len: int) -> bool:
    """Whether any assistant message appended after `before_len` contains tool_calls."""
    for msg in messages[before_len:]:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            return True
    return False


def _last_assistant_content(messages: list[dict]) -> str:
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


def run_clarify_trajectory(workspace_dir: Path, existing_ids: list[int], traj_id: int) -> dict | None:
    """Generate one trajectory where the first user turn is deliberately vague.

    Goal: teach the model that when user intent is ambiguous, the right action is
    to ask back (natural-language clarification) BEFORE calling any tool.

    Flow:
      Step 0:   User says something vague ("帮我看看" / "给我点建议" / ...).
      Check:    Agent must NOT call a tool this turn. If it did, trajectory is
                dropped (it's the opposite of what we want to teach).
      Step 1:   DeepSeek-User gives a concrete follow-up based on agent's question.
                → Agent: pick/search → start_problem → find_similar_problems
      Step 2:   User "writes code" (simulated by DeepSeek)
      Step 2.5: (1/3 chance) User mid-step question
      Step 3:   User ending → Agent: read_solution → analyze_and_memorize
    """
    logger.info(f"=== Trajectory {traj_id} (clarify) ===")

    code_quality = random.choices(["correct", "buggy", "partial"], weights=[1, 1, 1], k=1)[0]
    has_midstep = random.random() < 1/3
    logger.info(f"  code={code_quality}, midstep={has_midstep}")

    before_files = {str(f) for f in workspace_dir.rglob("*.py") if f.parent.name != "src"}

    agent = Agent()

    # Step 0: vague opener
    vague_msg = generate_vague_first_message()
    logger.info(f"  [user] (vague) {vague_msg}")

    msgs_before = len(agent.messages)
    try:
        agent.chat(vague_msg)
    except Exception as e:
        logger.warning(f"  Step 0 failed: {e}")
        return None

    # Key filter: agent must have asked for clarification, not called a tool.
    if _new_assistant_made_tool_call(agent.messages, msgs_before):
        logger.info("  ✗ Agent tool-called on vague input — drop (not a clarify example)")
        return None

    clarify_reply = _last_assistant_content(agent.messages)
    if not clarify_reply or len(clarify_reply) < 5:
        logger.info("  ✗ Empty / too-short clarify reply — drop")
        return None

    # Step 1: concrete follow-up
    concrete_msg, variant = generate_concrete_followup(clarify_reply)
    logger.info(f"  [user] (concrete/{variant}) {concrete_msg}")

    try:
        agent.chat(concrete_msg)
    except Exception as e:
        logger.warning(f"  Step 1 failed: {e}")
        return None

    solution_file = find_latest_solution_file(workspace_dir, before_files)

    # Step 2: simulate code
    if solution_file:
        simulate_user_code(solution_file, code_quality)
    else:
        logger.info("  No solution file found, skipping code simulation")

    # Step 2.5 (optional)
    if has_midstep and solution_file:
        mid_msg = generate_midstep_question()
        logger.info(f"  [user] (midstep) {mid_msg}")
        try:
            agent.chat(mid_msg)
        except Exception as e:
            logger.warning(f"  Midstep failed: {e}")

    # Step 3: ending
    msg_end = generate_ending_message(code_quality)
    logger.info(f"  [user] {msg_end}")

    try:
        agent.chat(msg_end)
    except Exception as e:
        logger.warning(f"  Step 3 failed: {e}")

    has_l3 = trajectory_has_l3_memory(agent.messages)

    trajectory = {
        "id": f"clarify_{traj_id:04d}",
        "type": "clarify",
        "config": {
            "concrete_variant": variant,
            "code_quality": code_quality,
            "has_midstep": has_midstep,
            "vague_first_msg": vague_msg,
            "clarify_reply_preview": clarify_reply[:300],
        },
        "has_l3_memory": has_l3,
        "messages": agent.messages,
        "tools": TOOL_SCHEMAS,
    }
    return trajectory


# ─── Core: run one free trajectory ───

def run_free_trajectory(workspace_dir: Path, first_message: str, traj_id: int, max_turns: int = 5) -> dict | None:
    logger.info(f"=== Trajectory {traj_id} (free) ===")
    logger.info(f"  [user] {first_message}")

    agent = Agent()

    try:
        agent.chat(first_message)
    except Exception as e:
        logger.warning(f"  Turn 1 failed: {e}")
        return None

    for turn in range(2, max_turns + 1):
        last_assistant_msg = ""
        for msg in reversed(agent.messages):
            if msg.get("role") == "assistant" and msg.get("content"):
                last_assistant_msg = msg["content"]
                break

        if not last_assistant_msg:
            break

        resp = USER_SIM_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "你正在和一个 LeetCode 刷题助手对话。你是用户，你在做 LeetCode 题目。"
                           "根据助手的回复，生成你作为用户的下一句话。保持自然简短。"
                           "如果觉得对话该结束了，回复'[END]'。",
            }, {
                "role": "user",
                "content": f"助手说：{last_assistant_msg}\n\n你的回复：",
            }],
            temperature=0.8,
            max_tokens=200,
        )
        user_reply = resp.choices[0].message.content.strip()

        if "[END]" in user_reply or not user_reply:
            logger.info(f"  [user-sim] conversation ended at turn {turn}")
            break

        logger.info(f"  [user] (turn {turn}) {user_reply[:80]}...")

        try:
            agent.chat(user_reply)
        except Exception as e:
            logger.warning(f"  Turn {turn} failed: {e}")
            break

    has_l3 = trajectory_has_l3_memory(agent.messages)

    trajectory = {
        "id": f"free_{traj_id:04d}",
        "type": "free",
        "first_message": first_message,
        "has_l3_memory": has_l3,
        "messages": agent.messages,
        "tools": TOOL_SCHEMAS,
    }
    return trajectory


# ─── Main ───

def main():
    parser = argparse.ArgumentParser(description="Generate SFT trajectories")
    parser.add_argument("--n", type=int, default=5, help="Number of trajectories")
    parser.add_argument("--mode", choices=["structured", "free", "clarify"], default="structured")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")
    parser.add_argument("--fail_output", type=str, default=None, help="Output for failed (no L3) trajectories")
    parser.add_argument("--free_intents", type=str, default=None,
                        help="File with one intent per line (for free mode)")
    parser.add_argument("--workspace", type=str,
                        default=str(Path(__file__).resolve().parents[1] / "workspace"),
                        help="Workspace directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    workspace_dir = Path(args.workspace).resolve()

    os.chdir(workspace_dir)
    logger.info(f"Workspace: {workspace_dir}")

    init_workspace_db(workspace_dir)

    existing_ids = get_existing_problems(workspace_dir)
    logger.info(f"Existing problems: {len(existing_ids)} ({existing_ids[:10]}...)")

    output_dir = Path("/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories")
    output_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = output_dir / "fail_cases"
    fail_dir.mkdir(exist_ok=True)

    output_path = args.output or str(output_dir / f"{args.mode}_trajectories.jsonl")
    fail_path = args.fail_output or str(fail_dir / f"{args.mode}_no_l3.jsonl")

    success_count = 0
    fail_count = 0

    def save_trajectory(traj):
        nonlocal success_count, fail_count
        if traj["has_l3_memory"]:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            success_count += 1
            logger.info(f"  ✓ Saved (L3 OK) trajectory {traj['id']} ({len(traj['messages'])} msgs)")
        else:
            with open(fail_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(traj, ensure_ascii=False) + "\n")
            fail_count += 1
            logger.info(f"  ✗ Saved (NO L3) trajectory {traj['id']} -> fail_cases/")

    if args.mode == "structured":
        for i in range(args.n):
            traj = run_structured_trajectory(workspace_dir, existing_ids, i)
            if traj:
                save_trajectory(traj)
            time.sleep(0.5)

    elif args.mode == "clarify":
        for i in range(args.n):
            traj = run_clarify_trajectory(workspace_dir, existing_ids, i)
            if traj:
                save_trajectory(traj)
            time.sleep(0.5)

    elif args.mode == "free":
        if args.free_intents:
            intents = Path(args.free_intents).read_text().strip().splitlines()
        else:
            intents = ["来一道题吧"] * args.n

        for i, intent in enumerate(intents[:args.n]):
            traj = run_free_trajectory(workspace_dir, intent.strip(), i)
            if traj:
                save_trajectory(traj)
            time.sleep(0.5)

    logger.info(f"\nDone! Success (with L3): {success_count}, Failed (no L3): {fail_count}")
    logger.info(f"Output: {output_path}")
    if fail_count > 0:
        logger.info(f"Fail cases: {fail_path}")


if __name__ == "__main__":
    main()
