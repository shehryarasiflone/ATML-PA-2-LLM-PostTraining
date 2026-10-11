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

import numpy as np
import pandas as pd
import torch

from common.data import load_yaml, read_jsonl, repo_path
from common.generation import batch_generate
from common.models import clear_gpu, load_policy, load_tokenizer
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward, extract_designated_final


def policy_specs(cfg):
    return {
        "sft": None,
        "rlvr": cfg["policies"]["rlvr"],
        "rlaif": cfg["policies"]["rlaif"],
    }


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def generate_math_responses(cfg, policy_name: str, rows: list[dict], batch_size: int = 4) -> list[dict]:
    specs = policy_specs(cfg)
    adapter = specs[policy_name]

    print(f"\nGenerating responses for: {policy_name.upper()} (Adapter: {adapter})")
    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=adapter, trainable=False)
    model.eval()

    records = []
    total = len(rows)
    max_tokens = int(cfg.get("math_max_new_tokens", 512))

    for start in range(0, total, batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [
            r.get("messages")
            if "messages" in r and r["messages"]
            else [{"role": "user", "content": str(r["question"])}]
            for r in chunk
        ]

        gen = batch_generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_prompt_length=int(cfg.get("max_prompt_length", 256)),
            max_new_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
        )

        for r, resp, n_tok in zip(chunk, gen["responses"], gen["response_lengths"]):
            gold = str(r["gold_final"])
            acc = exact_reward(resp, gold)
            has_fmt = extract_designated_final(resp) is not None
            records.append({
                "source_index": r.get("source_index", -1),
                "question": r["question"],
                "gold_final": gold,
                "response": resp,
                "response_tokens": int(n_tok),
                "exact_accuracy": float(acc),
                "format_compliance": float(has_fmt),
            })

        processed = min(start + batch_size, total)
        if processed % 50 == 0 or processed == total:
            print(f"Generated {processed:03d} / {total:03d} responses...")

        del gen
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    clear_gpu(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return records


def run_pairwise_eval(judge: PairwiseAIJudge, rows: list[dict], resp_map: dict[str, list[dict]]):
    n = len(rows)
    policy_pairs = [("rlvr", "sft"), ("rlaif", "sft"), ("rlvr", "rlaif")]
    pairwise_stats = {}

    print("\nRunning pairwise AI judging...")
    for p_a, p_b in policy_pairs:
        wins_a = 0
        wins_b = 0
        ties = 0

        for i in range(n):
            problem = rows[i]["question"]
            resp_a = resp_map[p_a][i]["response"]
            resp_b = resp_map[p_b][i]["response"]

            pref = judge.compare(problem, resp_a, resp_b)
            if pref == "A":
                wins_a += 1
            elif pref == "B":
                wins_b += 1
            else:
                ties += 1

        pairwise_stats[f"{p_a}_vs_{p_b}"] = {
            f"{p_a}_win_rate": float(wins_a / n),
            f"{p_b}_win_rate": float(wins_b / n),
            "tie_rate": float(ties / n),
        }
        print(f"  {p_a.upper()} vs {p_b.upper()}: Win={wins_a/n*100:.1f}%, Loss={wins_b/n*100:.1f}%, Tie={ties/n*100:.1f}%")

    return pairwise_stats


def evaluate_dataset(config_path: str, dataset: str, batch_size: int = 4):
    cfg = load_yaml(config_path)
    rows = read_jsonl(repo_path(dataset_path(cfg, dataset)))
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / "pairwise_judge_cache.json"

    print(f"\n=======================================================")
    print(f" Evaluating Dataset: {dataset.upper()} ({len(rows)} prompts)")
    print(f"=======================================================")

    policies = ["sft", "rlvr", "rlaif"]
    policy_responses = {}
    summary_metrics = {}

    # Stage 1: Generate completions policy-by-policy
    for pol in policies:
        cached_file = results_dir / f"{dataset}_{pol}_responses.jsonl"
        if cached_file.exists():
            print(f"Loading cached responses for {pol.upper()} from {cached_file.name}")
            records = read_jsonl(cached_file)
        else:
            records = generate_math_responses(cfg, pol, rows, batch_size=batch_size)
            with cached_file.open("w", encoding="utf-8") as f:
                for r in records:
                    f.write(json.dumps(r) + "\n")

        policy_responses[pol] = records

        # Compute summary stats
        accs = [r["exact_accuracy"] for r in records]
        fmts = [r["format_compliance"] for r in records]
        lens = [r["response_tokens"] for r in records]
        q75, q25 = np.percentile(lens, [75, 25]) if lens else (0, 0)

        summary_metrics[pol] = {
            "exact_accuracy": float(np.mean(accs)),
            "format_compliance_rate": float(np.mean(fmts)),
            "mean_length": float(np.mean(lens)),
            "std_length": float(np.std(lens)),
            "length_iqr": float(q75 - q25),
        }

    # Stage 2: Pairwise judging across the responses
    judge = PairwiseAIJudge(cfg, cache_path)
    pairwise_stats = run_pairwise_eval(judge, rows, policy_responses)
    judge.close()

    output_payload = {
        "dataset": dataset,
        "n_samples": len(rows),
        "policy_metrics": summary_metrics,
        "pairwise_preferences": pairwise_stats,
    }

    out_file = results_dir / f"{dataset}_eval.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nSaved {dataset.upper()} evaluation to: {out_file}\n")
    return output_payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    datasets = ["gsm", "transfer"] if args.dataset == "both" else [args.dataset]
    for ds in datasets:
        evaluate_dataset(args.config, ds, batch_size=args.batch_size)


if __name__ == "__main__":
    main()