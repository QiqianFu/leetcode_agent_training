"""
Quick eval: test if SFT model can generate correct tool calls.
Loads LoRA adapter on Qwen3-1.7B, runs a few test prompts, checks tool call format.
"""
import json
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_PATH = "Qwen/Qwen3-1.7B"
ADAPTER_PATH = "/shared/rsaas/qiqianf2/lc_agent_experiments/sft_real_exp002"
CACHE_DIR = "/shared/rsaas/qiqianf2/hf_models"

# Load the 13 tools (same as training data)
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".." / "leetcode_agent" / "src"))
from lc.tool_defs import TOOLS

# Test prompts: each should trigger a specific tool
TEST_CASES = [
    {
        "user": "来一道题吧",
        "expected_tool": "pick_problem",
        "desc": "随机推荐题目",
    },
    {
        "user": "我要做第146题",
        "expected_tool": "start_problem",
        "desc": "指定题号开题 (也可能先 check_problem)",
    },
    {
        "user": "来一道动态规划的题",
        "expected_tool": "search_problem",
        "desc": "按方向搜索",
    },
    {
        "user": "帮我看看第53题的笔记",
        "expected_tool": "read_memory",
        "desc": "读取记忆 (也可能先 check_problem)",
    },
    {
        "user": "我做完了，帮我看看代码",
        "expected_tool": "read_solution",
        "desc": "读取代码",
    },
    {
        "user": "帮我记一下：这道题的关键是用哈希表",
        "expected_tool": "write_memory",
        "desc": "写笔记",
    },
    {
        "user": "什么是单调栈",
        "expected_tool": "web_search",
        "desc": "知识搜索",
    },
    {
        "user": "我喜欢用迭代而不是递归",
        "expected_tool": "update_user_memory",
        "desc": "用户偏好",
    },
]


def load_model():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True, padding_side="left",
    )

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, cache_dir=CACHE_DIR, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map="auto",
    )

    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print(f"Model loaded on {model.device}")
    return model, tokenizer


def generate_response(model, tokenizer, user_message):
    messages = [
        {"role": "user", "content": user_message},
    ]

    text = tokenizer.apply_chat_template(
        messages, tools=TOOLS, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Decode only the generated part
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return response


def extract_tool_call(response: str) -> dict | None:
    """Try to extract tool call from model response."""
    # Qwen3 tool call format: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    import re
    match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            return {"raw": match.group(1), "parse_error": True}

    # Also check for function call patterns
    if "tool_call" in response.lower() or '"name"' in response:
        return {"raw": response[:200], "partial": True}

    return None


def main():
    model, tokenizer = load_model()

    print("\n" + "=" * 70)
    print("EVALUATION: SFT Model Tool Call Generation")
    print("=" * 70)

    correct = 0
    total = len(TEST_CASES)

    for i, tc in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}/{total}: {tc['desc']} ---")
        print(f"  User: {tc['user']}")
        print(f"  Expected tool: {tc['expected_tool']}")

        response = generate_response(model, tokenizer, tc["user"])

        # Truncate for display
        display_resp = response[:500].replace('\n', '\n  ')
        print(f"  Response:\n  {display_resp}")

        tool_call = extract_tool_call(response)
        if tool_call:
            tool_name = tool_call.get("name", "???")
            is_correct = tool_name == tc["expected_tool"]
            # Also accept alternative valid tools
            alternatives = {
                "start_problem": ["check_problem", "start_problem"],
                "read_memory": ["check_problem", "read_memory"],
            }
            if not is_correct and tc["expected_tool"] in alternatives:
                is_correct = tool_name in alternatives[tc["expected_tool"]]

            status = "PASS" if is_correct else "WRONG TOOL"
            if is_correct:
                correct += 1
            print(f"  Tool call: {tool_name} {'✓' if is_correct else '✗'} ({status})")
            if "arguments" in tool_call:
                print(f"  Arguments: {json.dumps(tool_call['arguments'], ensure_ascii=False)}")
        else:
            print(f"  Tool call: NONE (no tool call detected) ✗")
            # Check if model generated text response instead
            if response.strip() and "<tool_call>" not in response:
                print(f"  (Model gave text response instead of tool call)")

    print(f"\n{'=' * 70}")
    print(f"Result: {correct}/{total} correct tool calls ({correct/total*100:.0f}%)")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
