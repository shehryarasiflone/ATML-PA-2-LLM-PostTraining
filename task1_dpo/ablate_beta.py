from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from common.data import load_yaml, repo_path
from common.models import clear_gpu
from task1_dpo.evaluate import evaluate_dpo
from task1_dpo.train import run_training


def run_beta_ablation(config_path: str):
    cfg = load_yaml(config_path)
    betas = cfg.get("betas", [0.03, 0.10, 0.30])
    max_examples = int(cfg.get("short_ablation_examples", 600))
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Starting DPO Beta Regularization Sweep ===")
    print(f"Beta values: {betas}")
    print(f"Dataset limit per fork: {max_examples} examples\n")

    all_metrics = []

    for beta in betas:
        run_name = f"beta_{beta:.2f}".replace(".", "_")
        output_path = f"outputs/task1_dpo/{run_name}"
        
        print(f"\n==========================================")
        print(f" Training condition: Beta = {beta} ({run_name})")
        print(f"==========================================")

        # 1. Train matched DPO fork from clean initialization
        run_training(
            config_path=config_path,
            run_name=run_name,
            output_path=output_path,
            beta=beta,
            max_examples=max_examples,
        )

        # 2. Release training memory before evaluation
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"\nEvaluating condition: Beta = {beta}...")
        
        # 3. Evaluate under common protocol
        metrics = evaluate_dpo(
            config_path=config_path,
            adapter_path=output_path,
            run_name=run_name,
            beta=beta,
        )

        all_metrics.append(metrics)

        # 4. Clean GPU memory between conditions
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save summary JSON for report tables
    summary_path = results_dir / "beta_ablation_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    # Print comparative Markdown table
    print("\n" + "=" * 80)
    print("                 DPO BETA ABLATION RESULTS SUMMARY")
    print("=" * 80)
    print(f"| {'Beta':^6} | {'Held-out Loss':^14} | {'Pref Acc (%)':^13} | {'Sampled KL':^12} | {'Reward Score':^18} | {'Length Mean ± Std':^21} |")
    print(f"|{'-'*8}|{'-'*16}|{'-'*15}|{'-'*14}|{'-'*20}|{'-'*23}|")
    for m in all_metrics:
        b_val = m["beta"]
        loss_val = m["dpo_loss"]
        acc_val = m["preference_accuracy"] * 100.0
        kl_val = m["sampled_kl"]
        rew_val = f"{m['mean_reward']:.3f} ± {m['std_reward']:.3f}"
        len_val = f"{m['response_length_mean']:.1f} ± {m['response_length_std']:.1f}"
        print(f"| {b_val:^6.2f} | {loss_val:^14.4f} | {acc_val:^13.2f} | {kl_val:^12.4f} | {rew_val:^18} | {len_val:^21} |")
    print("=" * 80)
    print(f"Full metrics saved to: {summary_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    args = ap.parse_args()
    run_beta_ablation(args.config)


if __name__ == "__main__":
    main()