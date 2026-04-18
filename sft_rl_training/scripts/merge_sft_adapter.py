"""Merge a PEFT LoRA adapter into its base model and save a single merged model.

For Exp-014 we use this to create `sft_exp009_merged/` as the starting point
for GRPO training (TRL's GRPOTrainer adds a fresh LoRA on top of a merged model).
"""
from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter_path", required=True,
                    help="PEFT adapter dir (contains adapter_config.json)")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--cache_dir", default="/shared/rsaas/qiqianf2/hf_models")
    args = ap.parse_args()

    out = Path(args.output_dir)
    if out.exists():
        logger.warning(f"Output dir {out} already exists; will overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading base model: {args.base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    logger.info(f"Loading tokenizer: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, cache_dir=args.cache_dir, trust_remote_code=True
    )

    logger.info(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base, args.adapter_path)

    logger.info("Merging adapter into base weights (merge_and_unload)...")
    merged = model.merge_and_unload()

    logger.info(f"Saving merged model → {out}")
    merged.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)
    logger.info("Done.")


if __name__ == "__main__":
    main()
