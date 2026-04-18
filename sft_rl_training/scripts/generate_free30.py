"""Generate 30 free-form trajectories with DeepSeek self-play (5 turns each)."""
import json, os, sys, random, time, logging
from pathlib import Path

LEETCODE_SRC = Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"
sys.path.insert(0, str(LEETCODE_SRC))
assert os.environ.get("DEEPSEEK_API_KEY"), "DEEPSEEK_API_KEY not set — source project-root .env first"
os.environ["MAX_AGENT_HISTORY_MESSAGES"] = "500"

from openai import OpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

USER_SIM = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url="https://api.deepseek.com")

# --- Headless patches ---
def _headless_select(choices, load_more=None):
    if not choices: return None
    _, v = random.choice(choices)
    logger.info(f"  [headless] auto-selected: {v}")
    return v

import lc.ui
lc.ui.arrow_select = _headless_select
lc.ui.flush_stdin = lambda: None

from lc import db
from lc.agent import Agent, DEEPSEEK_MODEL
from lc.tool_defs import TOOLS as TOOL_SCHEMAS

import lc.tool_impl.problems as _pm
_pm.arrow_select = _headless_select
import lc.agent as _am
_am.flush_stdin = lambda: None

def _headless_call(self, messages):
    t0 = time.time()
    messages = self._sanitize_messages(messages)
    resp = self.client.chat.completions.create(
        model=DEEPSEEK_MODEL, messages=messages, tools=TOOL_SCHEMAS,
        stream=False, temperature=0.3, max_tokens=4096,
    )
    choice = resp.choices[0]
    content = choice.message.content or ""
    tool_calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                  for tc in (choice.message.tool_calls or [])]
    usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens,
             "total": resp.usage.total_tokens} if resp.usage else {}
    logger.info(f"  [LLM] {time.time()-t0:.1f}s, {usage.get('total','?')} tok, tools={[t['name'] for t in tool_calls] or 'none'}")
    return content, tool_calls, usage

Agent._call_model_once = _headless_call

# --- Main ---
USER_SYSTEM_PROMPT = (
    "你是一个正在使用 LeetCode 刷题 App 的用户。这个 App 会在你下达指令后返回一些内容帮助你刷题。"
    "你可以让它推荐题目、开始做题、查看之前做过的题、复习笔记等等。"
    "请自然地使用这个 App，像一个真实用户一样对话。保持简短自然。"
    "每次只说一句话。"
)

def generate_user_message(conversation_so_far: list[dict]) -> str:
    messages = [{"role": "system", "content": USER_SYSTEM_PROMPT}]
    for msg in conversation_so_far:
        messages.append(msg)
    messages.append({"role": "user", "content": "请说出你作为用户的下一句话："})
    resp = USER_SIM.chat.completions.create(
        model="deepseek-chat", messages=messages, temperature=0.9, max_tokens=100,
    )
    return resp.choices[0].message.content.strip().strip('"\'')

def run_one(traj_id: int, workspace_dir: Path) -> dict:
    logger.info(f"=== Free trajectory {traj_id} ===")
    agent = Agent()
    
    # DeepSeek-User generates first message with no prior context
    user_msg = generate_user_message([])
    logger.info(f"  [user] {user_msg}")

    # Track conversation from user-sim perspective
    user_sim_history = [{"role": "assistant", "content": user_msg}]

    for turn in range(5):
        try:
            agent.chat(user_msg)
        except Exception as e:
            logger.warning(f"  Turn {turn} agent failed: {e}")
            break

        # Get agent's last text response
        last_response = ""
        for m in reversed(agent.messages):
            if m.get("role") == "assistant" and m.get("content"):
                last_response = m["content"]
                break

        if turn < 4:  # Generate next user message for turns 0-3 (total 5 user messages)
            user_sim_history.append({"role": "user", "content": f"[App回复] {last_response[:300]}"})
            user_msg = generate_user_message(user_sim_history)
            user_sim_history.append({"role": "assistant", "content": user_msg})
            logger.info(f"  [user] (turn {turn+2}) {user_msg}")

    return {
        "id": f"free_{traj_id:04d}",
        "type": "free",
        "messages": agent.messages,
        "tools": TOOL_SCHEMAS,
    }

if __name__ == "__main__":
    random.seed(3333)
    workspace = Path(__file__).resolve().parents[1] / "workspace_b"  # Use workspace_b
    os.chdir(workspace)
    db.init_db()
    
    output_path = "/shared/rsaas/qiqianf2/lc_agent_experiments/sft_trajectories/free_30.jsonl"
    
    for i in range(30):
        traj = run_one(i, workspace)
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(traj, ensure_ascii=False) + "\n")
        logger.info(f"  Saved free_{i:04d} ({len(traj['messages'])} msgs)")
        time.sleep(0.5)
    
    logger.info(f"\nDone! 30 trajectories -> {output_path}")
