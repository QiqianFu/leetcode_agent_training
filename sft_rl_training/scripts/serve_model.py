"""
Lightweight OpenAI-compatible API server for the SFT model.

Serves the merged Qwen3-1.7B model with tool-calling support,
compatible with the leetcode_agent's OpenAI client.

Usage:
    conda run -n qwen_RL python sft_rl_training/scripts/serve_model.py \
        --model_path /shared/rsaas/qiqianf2/lc_agent_experiments/sft_merged_model \
        --port 8234

Then in another terminal:
    DEEPSEEK_BASE_URL=http://localhost:8234/v1 DEEPSEEK_MODEL=qwen3-sft lc
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
import logging
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI()

# Global model/tokenizer (loaded at startup)
model = None
tokenizer = None
MODEL_NAME = "qwen3-sft"


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatRequest(BaseModel):
    model: str = MODEL_NAME
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: str | None = None
    stream: bool = False
    temperature: float = 0.3
    max_tokens: int = 4096
    top_p: float = 0.95
    stream_options: dict | None = None


def generate_response(request: ChatRequest) -> tuple[str, dict]:
    """Generate a response from the model."""
    # Apply chat template with tools
    chat_kwargs = {}
    if request.tools:
        chat_kwargs["tools"] = request.tools

    text = tokenizer.apply_chat_template(
        request.messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            temperature=max(request.temperature, 0.01),
            top_p=request.top_p,
            do_sample=request.temperature > 0,
        )

    new_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

    usage = {
        "prompt_tokens": input_len,
        "completion_tokens": len(new_tokens),
        "total_tokens": input_len + len(new_tokens),
    }

    return response_text, usage


def parse_tool_calls(text: str) -> tuple[str | None, list[dict] | None]:
    """
    Parse tool calls from model output.

    Qwen3 outputs tool calls in this format:
    <tool_call>
    {"name": "tool_name", "arguments": {"arg1": "val1"}}
    </tool_call>
    """
    import re

    tool_call_pattern = re.compile(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
    )
    matches = tool_call_pattern.findall(text)

    if not matches:
        return text, None

    tool_calls = []
    for i, match in enumerate(matches):
        try:
            parsed = json.loads(match)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": parsed.get("name", ""),
                    "arguments": json.dumps(
                        parsed.get("arguments", {}), ensure_ascii=False
                    ),
                },
            })
        except json.JSONDecodeError:
            logger.warning("Failed to parse tool call: %s", match)
            continue

    if tool_calls:
        # Remove tool_call tags from content
        content = tool_call_pattern.sub("", text).strip()
        return content or None, tool_calls

    return text, None


def build_chat_completion(
    request_id: str,
    content: str | None,
    tool_calls: list[dict] | None,
    usage: dict,
    model_name: str,
) -> dict:
    """Build an OpenAI-compatible chat completion response."""
    message = {"role": "assistant"}
    if content:
        message["content"] = content
    if tool_calls:
        message["tool_calls"] = tool_calls
        # When there are tool calls, content should be null per OpenAI spec
        if not content:
            message["content"] = None

    return {
        "id": request_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if tool_calls else "stop",
            }
        ],
        "usage": usage,
    }


def build_stream_chunks(
    request_id: str,
    content: str | None,
    tool_calls: list[dict] | None,
    usage: dict,
    model_name: str,
    include_usage: bool = False,
) -> list[str]:
    """Build SSE stream chunks compatible with OpenAI streaming format."""
    chunks = []

    # First chunk: role
    delta_first = {"role": "assistant", "content": ""}
    chunks.append(
        json.dumps(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [{"index": 0, "delta": delta_first, "finish_reason": None}],
            }
        )
    )

    # Content chunks (send in ~20-char pieces for streaming feel)
    if content:
        chunk_size = 20
        for i in range(0, len(content), chunk_size):
            piece = content[i : i + chunk_size]
            chunks.append(
                json.dumps(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {"content": piece},
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )

    # Tool call chunks
    if tool_calls:
        for tc in tool_calls:
            # Send tool call start
            chunks.append(
                json.dumps(
                    {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": tc["id"],
                                            "type": "function",
                                            "function": {
                                                "name": tc["function"]["name"],
                                                "arguments": tc["function"][
                                                    "arguments"
                                                ],
                                            },
                                        }
                                    ]
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
            )

    # Final chunk with finish_reason
    finish_reason = "tool_calls" if tool_calls else "stop"
    final = {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }
    if include_usage:
        final["usage"] = usage
    chunks.append(json.dumps(final))

    return chunks


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    logger.info(
        "Request: %d messages, tools=%s, stream=%s",
        len(request.messages),
        bool(request.tools),
        request.stream,
    )

    # Generate
    response_text, usage = generate_response(request)

    # Parse tool calls
    content, tool_calls = parse_tool_calls(response_text)

    logger.info(
        "Response: %d tokens, tool_calls=%s",
        usage["completion_tokens"],
        [tc["function"]["name"] for tc in tool_calls] if tool_calls else None,
    )

    if not request.stream:
        return build_chat_completion(
            request_id, content, tool_calls, usage, MODEL_NAME
        )

    # Streaming response
    include_usage = (
        request.stream_options.get("include_usage", False)
        if request.stream_options
        else False
    )
    chunks = build_stream_chunks(
        request_id, content, tool_calls, usage, MODEL_NAME, include_usage
    )

    async def stream_generator():
        for chunk in chunks:
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "owned_by": "local",
            }
        ],
    }


def main():
    global model, tokenizer, MODEL_NAME

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/shared/rsaas/qiqianf2/lc_agent_experiments/sft_merged_model",
    )
    parser.add_argument(
        "--adapter_path", default=None,
        help="Optional PEFT LoRA adapter to apply on top of model_path",
    )
    parser.add_argument(
        "--tokenizer_path", default=None,
        help="Override tokenizer load path (defaults to adapter_path if set, "
             "else model_path). Useful when adapter dir has updated tokenizer.",
    )
    parser.add_argument("--port", type=int, default=8234)
    parser.add_argument("--model_name", default="qwen3-sft")
    args = parser.parse_args()

    MODEL_NAME = args.model_name

    tokenizer_src = args.tokenizer_path or args.adapter_path or args.model_path
    logger.info("Loading tokenizer from %s ...", tokenizer_src)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, trust_remote_code=True)

    logger.info("Loading base model from %s ...", args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.adapter_path:
        from peft import PeftModel
        logger.info("Applying LoRA adapter from %s ...", args.adapter_path)
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    model.eval()
    logger.info("Model loaded! Serving on port %d", args.port)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
