"""
SFT training script for LeetCode Agent on Qwen3-1.5B.

Key features:
- LoRA fine-tuning (memory efficient for 2x A40)
- Loss masking: only compute loss on assistant turns, mask user/tool/system turns
- Qwen3 chat template with tool calling support
- DeepSpeed ZeRO-2 for multi-GPU training
"""
from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─── Arguments ───

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        default="Qwen/Qwen3-1.7B",
        metadata={"help": "Path to pretrained model or model identifier from HuggingFace"},
    )
    cache_dir: Optional[str] = field(
        default="/shared/rsaas/qiqianf2/hf_models",
        metadata={"help": "Where to store the pretrained models downloaded from HuggingFace"},
    )
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA"})
    lora_r: int = field(default=64, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})


@dataclass
class DataArguments:
    data_path: str = field(
        default="",
        metadata={"help": "Path to the training data JSONL file"},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length for tokenization"},
    )


# ─── Data processing ───

def load_and_tokenize_data(
    data_path: str,
    tokenizer,
    max_seq_length: int,
) -> Dataset:
    """Load JSONL data and tokenize with assistant-only loss masking.

    Each sample has 'messages' (multi-turn conversation) and 'tools' (tool schemas).
    We use the tokenizer's chat template to format the conversation,
    then mask loss on all tokens except assistant responses.
    """
    IGNORE_INDEX = -100

    with open(data_path, "r", encoding="utf-8") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    logger.info("Loaded %d samples from %s", len(raw_data), data_path)

    all_input_ids = []
    all_labels = []
    skipped = 0

    for sample in raw_data:
        messages = sample["messages"]
        tools = sample.get("tools")

        # Use tokenizer's chat template to get full text
        # Qwen3 supports tools in apply_chat_template
        try:
            full_text = tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception as e:
            logger.warning("Failed to apply chat template for %s: %s", sample.get("id", "?"), e)
            skipped += 1
            continue

        full_ids = tokenizer.encode(full_text, add_special_tokens=False)

        if len(full_ids) > max_seq_length:
            full_ids = full_ids[:max_seq_length]

        # Build labels: mask everything except assistant content
        # Strategy: tokenize prefix up to each assistant turn, mark those tokens
        labels = [IGNORE_INDEX] * len(full_ids)

        # Find assistant segments by tokenizing incrementally
        prefix_so_far = ""
        for i, msg in enumerate(messages):
            if msg["role"] != "assistant":
                # Tokenize conversation up to and including this non-assistant message
                try:
                    prefix_text = tokenizer.apply_chat_template(
                        messages[:i + 1],
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=False if i < len(messages) - 1 else False,
                    )
                except Exception:
                    continue
                prefix_so_far = prefix_text
            else:
                # This is an assistant message - the tokens between previous prefix and current prefix should be unmasked
                try:
                    # Prefix including this assistant message
                    current_prefix = tokenizer.apply_chat_template(
                        messages[:i + 1],
                        tools=tools,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                except Exception:
                    continue

                prev_ids = tokenizer.encode(prefix_so_far, add_special_tokens=False) if prefix_so_far else []
                curr_ids = tokenizer.encode(current_prefix, add_special_tokens=False)

                # The assistant tokens are from len(prev_ids) to len(curr_ids)
                start_idx = len(prev_ids)
                end_idx = min(len(curr_ids), len(full_ids))

                for j in range(start_idx, end_idx):
                    labels[j] = full_ids[j]

                prefix_so_far = current_prefix

        # Shift labels for causal LM: labels[i] = input_ids[i+1]
        # Actually, HuggingFace Trainer handles the shift internally in CausalLM
        # So we just align labels with input_ids directly

        # Skip if no valid labels
        valid_labels = sum(1 for l in labels if l != IGNORE_INDEX)
        if valid_labels == 0:
            skipped += 1
            continue

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    logger.info("Processed %d samples, skipped %d", len(all_input_ids), skipped)

    # Pad to same length within batch (handled by data collator, but we store raw)
    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "labels": all_labels,
    })
    return dataset


class SFTDataCollator:
    """Pads input_ids and labels to the same length within a batch."""

    def __init__(self, tokenizer, max_seq_length: int):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __call__(self, features):
        max_len = min(
            max(len(f["input_ids"]) for f in features),
            self.max_seq_length,
        )

        input_ids_batch = []
        labels_batch = []
        attention_mask_batch = []

        for f in features:
            ids = f["input_ids"][:max_len]
            labs = f["labels"][:max_len]
            pad_len = max_len - len(ids)

            input_ids_batch.append(ids + [self.pad_token_id] * pad_len)
            labels_batch.append(labs + [-100] * pad_len)
            attention_mask_batch.append([1] * len(ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask_batch, dtype=torch.long),
        }


# ─── Main ───

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))

    if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(sys.argv[1])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # ─── Load tokenizer ───
    logger.info("Loading tokenizer from %s", model_args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ─── Load and tokenize data ───
    logger.info("Loading data from %s", data_args.data_path)
    dataset = load_and_tokenize_data(
        data_args.data_path,
        tokenizer,
        data_args.max_seq_length,
    )
    logger.info("Dataset size: %d samples", len(dataset))

    # Train/eval split
    if len(dataset) > 50:
        split = dataset.train_test_split(test_size=0.05, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]
    else:
        train_dataset = dataset
        eval_dataset = None

    # ─── Load model ───
    logger.info("Loading model from %s", model_args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )

    # ─── Apply LoRA ───
    if model_args.use_lora:
        logger.info("Applying LoRA (r=%d, alpha=%d)", model_args.lora_r, model_args.lora_alpha)
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            lora_dropout=model_args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    # Enable gradient checkpointing compatibility with LoRA
    if training_args.gradient_checkpointing and model_args.use_lora:
        model.enable_input_require_grads()

    # ─── Data collator ───
    collator = SFTDataCollator(tokenizer, data_args.max_seq_length)

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # ─── Train ───
    logger.info("Starting training...")
    train_result = trainer.train()
    trainer.save_model()
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ─── Eval ───
    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        eval_metrics["eval_samples"] = len(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete! Model saved to %s", training_args.output_dir)


if __name__ == "__main__":
    main()
