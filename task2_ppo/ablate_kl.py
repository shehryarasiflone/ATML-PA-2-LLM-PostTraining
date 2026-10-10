from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path

# Silence 8-bit quantized matrix multiplication warnings
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

import numpy as np

from common.data import load_yaml, repo_path
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo


def run_kl_ablation(config_path: str):
    cfg = load_yaml(config_path)
    fork_updates = int(cfg.get("fork_updates", 8))
    kl_values = [float(k) for k in cfg.get("kl_values", [0.0, 0.10, 0.20])]
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f" Starting KL-Pressure Ablation Study")
    print(f" Fork Updates: {fork_updates} | Conditions: {kl_values}")
    print(f"=======================================================\n")

    summary_records = {}

    for beta_kl in kl_values:
        run_name = f"kl_beta_{beta_kl:.2f}"
        out_adapter = f"outputs/task2_ppo/{run_name}"

        print(f"\n>>> Running Matched Fork: {run_name} (Updates: {fork_updates}, KL Beta: {beta_kl})")
        run_ppo(
            config_path=config_path,
            output=out_adapter,
            updates=fork_updates,
            clip_epsilon=float(cfg.get("clip_epsilon", 0.20)),
            kl_beta=beta_kl,
            run_name=run_name,
        )

        print(f"\n>>> Evaluating Fork: {run_name}")
        eval_metrics = evaluate_ppo(
            config_path=config_path,
            adapter_path=out_adapter,
            run_name=run_name,
            batch_size=2,
        )

        # Load training log to retrieve mean entropy during optimization
        log_file = results_dir / f"{run_name}_train_log.json"
        entropies = []
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                history = json.load(f)
            entropies = [step["entropy"] for step in history]

        summary_records[str(beta_kl)] = {
            "kl_beta": beta_kl,
            "mean_reward": eval_metrics["mean_reward"],
            "std_reward": eval_metrics["std_reward"],
            "sampled_kl": eval_metrics["sampled_kl"],
            "response_length_mean": eval_metrics["response_length_mean"],
            "response_length_std": eval_metrics["response_length_std"],
            "response_length_iqr": eval_metrics["response_length_iqr"],
            "eos_termination_rate": eval_metrics["eos_termination_rate"],
            "training_mean_entropy": float(np.mean(entropies)) if entropies else 0.0,
        }

    summary_file = results_dir / "kl_ablation_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)

    print("\n=======================================================")
    print(" KL-Pressure Ablation Summary")
    print("=======================================================")
    print(f"{'KL Beta':<10} | {'Reward':<16} | {'Sampled KL':<14} | {'Resp Length':<16} | {'Entropy':<10}")
    print("-" * 72)
    for k_str, d in summary_records.items():
        print(
            f"{d['kl_beta']:<10.2f} | "
            f"{d['mean_reward']:<6.4f} ± {d['std_reward']:<6.2f} | "
            f"{d['sampled_kl']:<14.4f} | "
            f"{d['response_length_mean']:<6.1f} ± {d['response_length_std']:<6.1f} | "
            f"{d['training_mean_entropy']:<10.4f}"
        )
    print("=======================================================\n")
    print(f"Saved complete ablation summary to: {summary_file}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    args = ap.parse_args()
    run_kl_ablation(args.config)


if __name__ == "__main__":
    main()