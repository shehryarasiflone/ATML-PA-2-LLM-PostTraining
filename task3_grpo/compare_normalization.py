from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path
from task3_grpo.continue_train import run_grpo
from task3_grpo.evaluate import evaluate_grpo


def run_normalization_comparison(config_path: str):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    conditions = ["grpo", "dr_grpo"]
    comparison_results = {}

    for loss_type in conditions:
        run_name = f"norm_{loss_type}"
        out_adapter = f"outputs/task3_grpo/{run_name}"

        print(f"\n>>> Running Normalization Fork: {loss_type} (Updates: {fork_updates})")
        run_grpo(
            config_path=config_path,
            output=out_adapter,
            updates=fork_updates,
            loss_type=loss_type,
            run_name=run_name,
        )

        print(f"\n>>> Evaluating Fork: {run_name}")
        eval_metrics = evaluate_grpo(
            config_path=config_path,
            adapter_path=out_adapter,
            run_name=run_name,
            batch_size=4,
        )

        # Read training log to extract length-conditioned gradient statistics
        log_file = results_dir / f"{run_name}_train_log.json"
        grad_norms = []
        lengths = []
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                history = json.load(f)
            grad_norms = [step["policy_grad_norm"] for step in history]
            lengths = [step["response_length"] for step in history]

        comparison_results[loss_type] = {
            "loss_type": loss_type,
            "eval_metrics": eval_metrics,
            "mean_train_grad_norm": float(sum(grad_norms) / max(1, len(grad_norms))),
            "mean_train_length": float(sum(lengths) / max(1, len(lengths))),
        }

    out_file = results_dir / "normalization_comparison.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2)

    print("\n=======================================================")
    print(" Normalization Comparison Summary")
    print("=======================================================")
    print(f"{'Condition':<12} | {'Reward':<16} | {'Sampled KL':<12} | {'Held-out Length':<16} | {'Train Grad Norm':<16}")
    print("-" * 76)
    for c_type, d in comparison_results.items():
        ev = d["eval_metrics"]
        print(
            f"{c_type:<12} | "
            f"{ev['mean_reward']:<6.4f} ± {ev['std_reward']:<6.2f} | "
            f"{ev['sampled_kl']:<12.4f} | "
            f"{ev['response_length_mean']:<6.1f} ± {ev['response_length_std']:<6.1f} | "
            f"{d['mean_train_grad_norm']:<16.4f}"
        )
    print("=======================================================\n")
    print(f"Saved comparison to: {out_file}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    run_normalization_comparison(args.config)


if __name__ == "__main__":
    main()