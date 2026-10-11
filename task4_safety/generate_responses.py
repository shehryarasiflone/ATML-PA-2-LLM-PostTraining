from __future__ import annotations

import argparse
import gc
import json
import logging
import warnings
from pathlib import Path

# Silence 8-bit quantized matrix multiplication warnings
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

import pandas as pd
import torch

from common.data import load_yaml, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = specs[policy_name]

    print(f"\n=======================================================")
    print(f" Generating Responses for Policy: {policy_name}")
    print(f" Adapter: {adapter}")
    print(f"=======================================================\n")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    model.eval()

    df = load_xstest(cfg)
    records = []
    total = len(df)

    for start in range(0, total, batch_size):
        chunk = df.iloc[start : start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        gen = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=int(cfg.get("safety_max_new_tokens", 256)),
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        for (_, row), response, n_tok in zip(chunk.iterrows(), gen["responses"], gen["response_lengths"]):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),
            })

        processed = min(start + batch_size, total)
        if processed % 50 == 0 or processed == total:
            print(f"Generated {processed:03d} / {total:03d} responses for {policy_name}...")

        del gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clear_gpu(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policy", choices=["sft", "dpo", "ppo", "grpo", "all"], default="all")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    outdir.mkdir(parents=True, exist_ok=True)

    policies_to_run = (
        ["sft", "dpo", "ppo", "grpo"] if args.policy == "all" else [args.policy]
    )

    for pol in policies_to_run:
        out_file = outdir / f"generated_{pol}.jsonl"
        records = generate_for_policy(cfg, pol, batch_size=args.batch_size)
        with out_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
        print(f"Saved {len(records)} responses to: {out_file}")

    print("\n--- Response Generation Complete ---")


if __name__ == "__main__":
    main()