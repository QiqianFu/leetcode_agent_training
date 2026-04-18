"""
Generate SFT training data for the "no tool call" behavior.

Self-distillation: sample from Qwen3-1.7B (base Instruct) with the full
leetcode_agent system prompt + tool schemas, filter out any response that
contains a <tool_call>, keep only pure-text replies that match the target
behavior for each category.

6 categories × ~25 samples, 50% 1-turn / 50% 2-turn.

Output JSONL format is compatible with existing `all_trajectories.jsonl`:
  {id, type, category, turns, messages, tools}

Usage:
    # Dry run (3 samples per category, no save)
    python generate_no_tool_data.py --n_per_category 3 --dry_run

    # Real run
    python generate_no_tool_data.py --n_per_category 25 \\
        --output /shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/no_tool_data.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path

import torch
from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─── Setup: make leetcode_agent importable for SYSTEM_PROMPT + TOOLS ───
LEETCODE_SRC = Path(__file__).resolve().parents[2] / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))

from lc.agent import SYSTEM_PROMPT
from lc.tool_defs import TOOLS as TOOL_SCHEMAS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Constants ───

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-REDACTED")
DEEPSEEK_CLIENT = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

QWEN3_MODEL_ID = os.environ.get("QWEN3_MODEL_ID", "Qwen/Qwen3-1.7B")
QWEN3_CACHE_DIR = os.environ.get("QWEN3_CACHE_DIR", "/shared/rsaas/qiqianf2/hf_models")

EXISTING_TRAJ_PATH = Path(
    "/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/all_trajectories.jsonl"
)

# ─── Category definitions ───

CATEGORIES = {
    "greeting_or_feature": {
        "desc": "用户只是打招呼，或问这个助手能做什么、怎么用、支持哪些操作。期望行为：纯文字回复，介绍功能或简短问候，不调用任何工具。",
        "seeds": [
            "你好",
            "hi",
            "在吗",
            "你能做什么",
            "这个工具怎么用",
            "你支持哪些功能",
            "你是啥",
            "怎么开始",
        ],
        "allow_two_turn": True,
    },
    "status_sharing": {
        "desc": "用户分享状态、情绪、日常（没有做题相关的明确请求）。期望行为：共情回复，可选轻度引导，不调用任何工具。",
        "seeds": [
            "今天好累",
            "昨天面试挂了",
            "刷题好烦",
            "最近状态不好",
            "人麻了",
            "工作太忙没时间刷题",
            "复习得心力交瘁",
            "今天心情不错",
        ],
        "allow_two_turn": True,
    },
    "ambiguous_intent": {
        "desc": "用户想做点什么但表达很模糊，没有说清楚具体是什么方向/题号/类型。期望行为：先用自然的一句话反问澄清需求，不调用任何工具。",
        "seeds": [
            "帮我看看",
            "随便来点",
            "给我点建议",
            "我想进步",
            "你看看怎么办",
            "有什么推荐的",
            "帮我安排一下",
            "来点刺激的",
        ],
        # 2-turn will produce contradictory behavior (user clarifies → should call tool).
        "allow_two_turn": False,
    },
    "mid_problem_chat": {
        "desc": "用户在做题过程中和助手纯讨论（解题思路、时间复杂度、概念澄清等），不需要助手调用工具。期望行为：基于已开始的题目上下文，直接用文字讨论/讲解。",
        "seeds": [
            "这题该怎么想",
            "时间复杂度是多少",
            "这种类型的题有什么通用套路吗",
            "为什么要用 DP 做",
            "我这个思路对吗",
            "我没太懂题意",
            "这题贪心为什么不行",
            "hash 表这里怎么用",
        ],
        "allow_two_turn": True,
        "needs_context": True,
    },
    "complaint_or_giveup": {
        "desc": "用户抱怨题目太难、做不下去、想放弃。期望行为：共情 + 可选轻度鼓励/建议，不调用任何工具。",
        "seeds": [
            "这题太难了",
            "我放弃了",
            "做不下去",
            "不想做了",
            "算了不会做了",
            "真的搞不动",
            "我是不是不适合刷题",
            "这题搞崩了",
        ],
        "allow_two_turn": True,
    },
    "off_topic": {
        "desc": "用户说的话和 LeetCode 刷题关系不大（闲聊、推荐、问其他话题等）。期望行为：礼貌简短回应，自然往刷题方向引导或不强行引导，不调用任何工具。",
        "seeds": [
            "今天天气不错",
            "你推荐本书吧",
            "周末去哪玩好",
            "Python 和 C++ 哪个好",
            "你觉得刷题有用吗",
            "大厂真的那么卷吗",
            "考研和找工作哪个好",
            "你吃饭了吗",
        ],
        "allow_two_turn": True,
    },
}

# ─── Qwen3 model (lazy loaded) ───

_QWEN_MODEL = None
_QWEN_TOKENIZER = None


def load_qwen():
    global _QWEN_MODEL, _QWEN_TOKENIZER
    if _QWEN_MODEL is not None:
        return _QWEN_MODEL, _QWEN_TOKENIZER

    logger.info("Loading Qwen3-1.7B (id=%s, cache=%s) ...", QWEN3_MODEL_ID, QWEN3_CACHE_DIR)
    _QWEN_TOKENIZER = AutoTokenizer.from_pretrained(
        QWEN3_MODEL_ID, cache_dir=QWEN3_CACHE_DIR, trust_remote_code=True,
    )
    _QWEN_MODEL = AutoModelForCausalLM.from_pretrained(
        QWEN3_MODEL_ID,
        cache_dir=QWEN3_CACHE_DIR,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    _QWEN_MODEL.eval()
    logger.info("Qwen3 ready (device=%s)", _QWEN_MODEL.device)
    return _QWEN_MODEL, _QWEN_TOKENIZER


def sample_qwen(messages: list[dict], k: int = 3, temperature: float = 0.8, max_new_tokens: int = 512) -> list[str]:
    """Sample k completions from Qwen3 given the chat messages + tool schemas."""
    model, tokenizer = load_qwen()

    # enable_thinking=False: skip <think> blocks (SFT'd model produces clean content).
    # The template silently ignores unknown kwargs on older versions, but Qwen3 supports it.
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tools=TOOL_SCHEMAS,
            add_generation_prompt=True,
            tokenize=False,
        )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=k,
            pad_token_id=tokenizer.eos_token_id,
        )

    completions = []
    for out in outputs:
        new_tokens = out[input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        completions.append(text)
    return completions


# ─── Filtering ───

TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOOL_CALL_OPEN_RE = re.compile(r"<tool_call\b")  # catches truncated outputs too


def has_tool_call(text: str) -> bool:
    return bool(TOOL_CALL_OPEN_RE.search(text))


def strip_qwen_thinking(text: str) -> str:
    """Qwen3 may emit <think>...</think> blocks; strip them for the final stored content."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def passes_filter(text: str, min_len: int = 5, max_len: int = 800) -> tuple[bool, str]:
    """Return (ok, reason_if_not_ok)."""
    if has_tool_call(text):
        return False, "has_tool_call"
    cleaned = strip_qwen_thinking(text)
    if not cleaned:
        return False, "empty_after_strip"
    if len(cleaned) < min_len:
        return False, f"too_short({len(cleaned)})"
    if len(cleaned) > max_len:
        return False, f"too_long({len(cleaned)})"
    # Cheap repetition check
    lines = cleaned.splitlines()
    if len(lines) > 3 and len(set(lines)) / len(lines) < 0.5:
        return False, "repetitive"
    return True, ""


# ─── DeepSeek helpers ───

def generate_scenario_pool(category: str, n: int) -> list[str]:
    """Ask DeepSeek to produce n varied first-turn user messages for this category."""
    cfg = CATEGORIES[category]
    seeds_str = "\n".join(f"- {s}" for s in cfg["seeds"])
    prompt = (
        f"你需要为 LeetCode 刷题助手的训练数据生成 {n} 条多样化的用户首句话，类别说明如下：\n\n"
        f"【类别】{category}\n"
        f"【说明】{cfg['desc']}\n\n"
        f"【参考例子（不要照抄，只是展示风格和分布）】\n{seeds_str}\n\n"
        f"要求：\n"
        f"1. 严格属于这个类别，不要混入别的类型（比如不要突然说'来一道 DP 题'这种明确做题请求）\n"
        f"2. 语言自然，像真人对话（简短 / 随意 / 口语都可以）\n"
        f"3. 覆盖不同语气：礼貌、口语、疲惫、好奇、抱怨等\n"
        f"4. 长度不要太齐整，从几个字到一两句话都有\n"
        f"5. 中英文混杂的也可以（比如 'hi 在吗'、'今天好 tired'）\n\n"
        f"直接输出 {n} 条，每行一条，不要编号、不要引号、不要额外解释。"
    )
    resp = DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=2000,
    )
    raw = resp.choices[0].message.content.strip()
    lines = [l.strip().strip('"\'-•·').strip() for l in raw.splitlines() if l.strip()]
    # Deduplicate + strip numbering artifacts
    out = []
    seen = set()
    for l in lines:
        l = re.sub(r"^\d+[\.\)]\s*", "", l).strip()
        if l and l not in seen:
            out.append(l)
            seen.add(l)
    return out


def generate_followup_user(category: str, prior_messages: list[dict]) -> str:
    """Given a user↔assistant exchange, ask DeepSeek to produce a natural user follow-up
    that stays within the same category's spirit (so the 2nd turn is also no-tool)."""
    cfg = CATEGORIES[category]
    # Only pass the last two turns to DeepSeek for brevity
    relevant = [m for m in prior_messages if m.get("role") in ("user", "assistant")][-2:]
    transcript = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}：{m.get('content','')}" for m in relevant
    )
    prompt = (
        f"你在模拟一个正在和 LeetCode 刷题助手聊天的用户。当前话题类别：{category}。\n"
        f"类别说明：{cfg['desc']}\n\n"
        f"已有对话：\n{transcript}\n\n"
        f"请生成这个用户的下一句话。要求：\n"
        f"1. 仍然保持在「{category}」这个类别里，不要突然切换成明确的做题请求\n"
        f"2. 对助手的上一句话做自然延续（追问、吐槽、回应、接话都可以）\n"
        f"3. 简短自然，一两句话，不要引号\n"
        f"4. 不要生成助手回复，只生成用户这一句"
    )
    resp = DEEPSEEK_CLIENT.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip().strip('"\'')


def llm_judge(category: str, messages: list[dict]) -> tuple[bool, str]:
    """Optional: ask DeepSeek whether the assistant's replies are appropriate for this category.
    Returns (accept, brief_reason). Cheap single call."""
    cfg = CATEGORIES[category]
    transcript = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}：{m.get('content','')}"
        for m in messages if m.get("role") in ("user", "assistant")
    )
    prompt = (
        f"你是数据质量审核员。类别：{category}\n"
        f"类别期望行为：{cfg['desc']}\n\n"
        f"对话：\n{transcript}\n\n"
        f"判断助手的每一次回复是否：\n"
        f"(a) 符合类别期望行为\n"
        f"(b) 语言自然、没有明显重复、没有胡言乱语\n"
        f"(c) 没有调用任何工具（必须纯文字）\n\n"
        f"只输出 JSON：{{\"accept\": true/false, \"reason\": \"一句话理由\"}}"
    )
    try:
        resp = DEEPSEEK_CLIENT.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content.strip()
        data = json.loads(raw)
        return bool(data.get("accept", False)), str(data.get("reason", ""))[:200]
    except Exception as e:
        logger.warning("llm_judge failed: %s", e)
        return True, "judge_failed_default_accept"


# ─── Prior-context prefixes (for mid_problem_chat) ───

def load_problem_contexts(n: int = 30) -> list[list[dict]]:
    """Load n prior-message prefixes from existing trajectories, each ending right after
    the assistant describes the started problem (content-only, no pending tool_calls)."""
    if not EXISTING_TRAJ_PATH.exists():
        logger.warning("No existing trajectories found at %s", EXISTING_TRAJ_PATH)
        return []

    prefixes = []
    with open(EXISTING_TRAJ_PATH, encoding="utf-8") as f:
        for line in f:
            if len(prefixes) >= n:
                break
            try:
                traj = json.loads(line)
            except json.JSONDecodeError:
                continue
            msgs = traj.get("messages", [])
            prefix = _extract_post_start_prefix(msgs)
            if prefix is not None:
                prefixes.append(prefix)

    random.shuffle(prefixes)
    logger.info("Loaded %d problem-context prefixes", len(prefixes))
    return prefixes


def _extract_post_start_prefix(messages: list[dict]) -> list[dict] | None:
    """Extract messages up through first assistant-only-text turn after start_problem succeeded."""
    seen_start = False
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            content = msg.get("content", "")
            if '"status": "started"' in content or '"status":"started"' in content:
                seen_start = True
        if seen_start and msg.get("role") == "assistant" and not msg.get("tool_calls"):
            # Include everything up to and including this assistant content turn
            return messages[: i + 1]
    return None


# ─── Core generation ───

def generate_one_sample(
    category: str,
    scenario: str,
    turns: int,
    context_prefix: list[dict] | None = None,
    k_samples: int = 3,
    use_judge: bool = True,
) -> dict | None:
    """Generate a single (possibly multi-turn) no-tool-call sample.
    Returns None if all samples failed filter."""
    cfg = CATEGORIES[category]
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}

    # Start building the conversation
    conv: list[dict] = []
    if context_prefix:
        conv.extend(context_prefix)
    conv.append({"role": "user", "content": scenario})

    # ── Turn 1 ──
    sampling_input = [system_msg] + conv if not context_prefix else conv  # context already includes whatever is needed
    # Prefix trajectories don't start with a system message, so always prepend
    sampling_input = [system_msg] + conv
    candidates = sample_qwen(sampling_input, k=k_samples, temperature=0.8)
    accepted_turn1 = None
    reject_reasons = []
    for c in candidates:
        ok, reason = passes_filter(c)
        if ok:
            accepted_turn1 = strip_qwen_thinking(c)
            break
        reject_reasons.append(reason)
    if accepted_turn1 is None:
        logger.info("  [reject turn1] %s | %s", scenario[:40], reject_reasons)
        return None
    conv.append({"role": "assistant", "content": accepted_turn1})

    # ── Turn 2 (optional) ──
    if turns >= 2:
        followup = generate_followup_user(category, conv)
        conv.append({"role": "user", "content": followup})
        sampling_input = [system_msg] + conv
        candidates2 = sample_qwen(sampling_input, k=k_samples, temperature=0.8)
        accepted_turn2 = None
        reject_reasons2 = []
        for c in candidates2:
            ok, reason = passes_filter(c)
            if ok:
                accepted_turn2 = strip_qwen_thinking(c)
                break
            reject_reasons2.append(reason)
        if accepted_turn2 is None:
            logger.info("  [reject turn2] %s | %s", followup[:40], reject_reasons2)
            return None
        conv.append({"role": "assistant", "content": accepted_turn2})

    # ── Optional LLM judge on the whole thing ──
    if use_judge:
        ok, reason = llm_judge(category, conv)
        if not ok:
            logger.info("  [judge reject] %s | %s", scenario[:40], reason)
            return None

    return {
        "category": category,
        "turns": turns,
        "scenario": scenario,
        "messages": conv,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_category", type=int, default=25,
                        help="Target number of samples per category")
    parser.add_argument("--two_turn_ratio", type=float, default=0.5,
                        help="Fraction of samples that should be 2-turn")
    parser.add_argument("--k_samples", type=int, default=3,
                        help="Over-sample K completions per scenario")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip LLM-as-judge filter (saves DeepSeek calls)")
    parser.add_argument("--output", type=str,
                        default="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/no_tool_data.jsonl")
    parser.add_argument("--scenario_cache", type=str,
                        default="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/no_tool_scenarios.json")
    parser.add_argument("--reject_log", type=str,
                        default="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/no_tool_rejects.jsonl")
    parser.add_argument("--dry_run", action="store_true",
                        help="Do not write output files; just print")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Subset of categories to run (default: all)")
    parser.add_argument("--prep_scenarios_only", action="store_true",
                        help="Only build/cache the scenario pool via DeepSeek, then exit (no GPU sampling)")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_path = Path(args.output)
    cache_path = Path(args.scenario_cache)
    if not args.dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: scenario pool (cached) ──
    if cache_path.exists():
        logger.info("Loading cached scenarios from %s", cache_path)
        scenario_pool = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        logger.info("Generating scenario pool via DeepSeek...")
        scenario_pool = {}
        for cat in CATEGORIES:
            # Over-generate so later we have buffer for rejects; aim for 2x target
            n_needed = max(args.n_per_category * 2, 50)
            scenarios = generate_scenario_pool(cat, n_needed)
            scenario_pool[cat] = scenarios
            logger.info("  %s: %d scenarios generated", cat, len(scenarios))
            time.sleep(0.5)
        if not args.dry_run:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(scenario_pool, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.prep_scenarios_only:
        logger.info("Scenario prep done. Exiting (--prep_scenarios_only).")
        return

    # ── Eager-load Qwen3 so model/tokenizer errors fail fast (not per-scenario) ──
    load_qwen()

    # ── Stage 2: load context prefixes for mid_problem_chat ──
    problem_contexts = load_problem_contexts(n=40)
    if not problem_contexts and "mid_problem_chat" in (args.categories or CATEGORIES.keys()):
        logger.warning("No problem contexts available — mid_problem_chat will be skipped")

    # ── Stage 3: sample + filter ──
    target_cats = args.categories or list(CATEGORIES.keys())
    n_target = args.n_per_category
    use_judge = not args.no_judge

    stats = {c: {"accepted": 0, "rejected": 0, "turn1_only": 0, "turn2": 0} for c in target_cats}
    sample_id = 0

    for cat in target_cats:
        if cat not in CATEGORIES:
            logger.warning("Unknown category %s, skipping", cat)
            continue
        cfg = CATEGORIES[cat]
        allow_2t = cfg.get("allow_two_turn", True)
        needs_ctx = cfg.get("needs_context", False)
        if needs_ctx and not problem_contexts:
            logger.warning("Skipping %s (no context prefixes)", cat)
            continue

        scenarios = scenario_pool.get(cat, [])
        random.shuffle(scenarios)
        scenario_iter = iter(scenarios)

        accepted = 0
        rejected = 0
        # Decide how many should be 2-turn
        n_two = int(n_target * args.two_turn_ratio) if allow_2t else 0
        n_one = n_target - n_two
        # Keep separate counters to hit the ratio
        got_one = 0
        got_two = 0

        logger.info("=== Category: %s (target=%d, 1t=%d, 2t=%d) ===", cat, n_target, n_one, n_two)

        while got_one < n_one or got_two < n_two:
            try:
                scenario = next(scenario_iter)
            except StopIteration:
                logger.warning("  Ran out of scenarios for %s (accepted=%d)", cat, accepted)
                break

            # Pick how many turns this sample should be
            need_two = got_two < n_two
            need_one = got_one < n_one
            if need_two and need_one:
                turns = random.choice([1, 2])
            elif need_two:
                turns = 2
            else:
                turns = 1

            context_prefix = random.choice(problem_contexts) if needs_ctx else None

            try:
                sample = generate_one_sample(
                    category=cat,
                    scenario=scenario,
                    turns=turns,
                    context_prefix=context_prefix,
                    k_samples=args.k_samples,
                    use_judge=use_judge,
                )
            except Exception as e:
                logger.exception("  generation error: %s", e)
                rejected += 1
                continue

            if sample is None:
                rejected += 1
                # Try the other turn count once before giving up on this scenario
                continue

            sample["id"] = f"no_tool_{sample_id:04d}"
            sample["type"] = "no_tool"
            sample["tools"] = TOOL_SCHEMAS
            sample_id += 1
            accepted += 1
            if turns == 1:
                got_one += 1
                stats[cat]["turn1_only"] += 1
            else:
                got_two += 1
                stats[cat]["turn2"] += 1

            if not args.dry_run:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
            logger.info(
                "  ✓ %s [%dt] %s → %s",
                sample["id"],
                turns,
                scenario[:30],
                sample["messages"][-1]["content"][:60].replace("\n", " "),
            )

        stats[cat]["accepted"] = accepted
        stats[cat]["rejected"] = rejected

    # ── Report ──
    logger.info("\n=== Final stats ===")
    total_acc = 0
    for cat, s in stats.items():
        logger.info(
            "  %s: accepted=%d (1t=%d, 2t=%d), rejected=%d",
            cat, s["accepted"], s["turn1_only"], s["turn2"], s["rejected"],
        )
        total_acc += s["accepted"]
    logger.info("Total accepted: %d", total_acc)
    if not args.dry_run:
        logger.info("Output: %s", output_path)


if __name__ == "__main__":
    main()
