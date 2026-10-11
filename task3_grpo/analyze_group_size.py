from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def load_k8_cache(path):
    rows = read_jsonl(repo_path(path))
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)

    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt: dict, k: int):
    """Partition the cached generations into K-sized groups keeping total generations = 192."""
    groups = []
    for pid, completions in by_prompt.items():
        # completions has length 8
        for i in range(0, 8, k):
            subgroup = completions[i : i + k]
            if len(subgroup) == k:
                groups.append({"prompt_id": pid, "completions": subgroup})
    return groups


def analyze_group_metrics(groups, tol=1e-5):
    """Calculate informative rate, reward std, and relative advantage variance."""
    stds = []
    informative_flags = []
    all_advantages = []

    for grp in groups:
        rewards = np.array([c["reward"] for c in grp["completions"]], dtype=np.float64)
        mean_r = np.mean(rewards)
        std_r = np.std(rewards, ddof=0)
        stds.append(std_r)

        is_info = std_r > tol
        informative_flags.append(float(is_info))

        # Relative advantages
        clamped_std = max(std_r, 1e-6)
        advs = (rewards - mean_r) / clamped_std
        all_advantages.extend(advs.tolist())

    return {
        "num_groups": len(groups),
        "informative_rate": float(np.mean(informative_flags)),
        "uninformative_rate": float(1.0 - np.mean(informative_flags)),
        "mean_within_group_std": float(np.mean(stds)),
        "relative_signal_variance": float(np.var(all_advantages)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    group_sizes = [int(k) for k in cfg.get("group_sizes", [2, 4, 8])]
    print(f"\n=======================================================")
    print(f" GRPO Group-Size Informativeness Study")
    print(f" Total Cached Prompts: {len(by_prompt)} (Total generations: 192)")
    print(f" Group sizes: {group_sizes}")
    print(f"=======================================================\n")

    # Define prompt difficulty by overall prompt reward mean across all 8 generations
    prompt_mean_rewards = {
        pid: np.mean([c["reward"] for c in comps])
        for pid, comps in by_prompt.items()
    }
    median_reward = float(np.median(list(prompt_mean_rewards.values())))
    print(f"Prompt Difficulty Median Split Threshold: {median_reward:.4f}")

    results = {}

    for k in group_sizes:
        all_groups = regroup_equal_generation_budget(by_prompt, k)
        overall_metrics = analyze_group_metrics(all_groups)

        # Difficulty subsets
        difficult_groups = [g for g in all_groups if prompt_mean_rewards[g["prompt_id"]] <= median_reward]
        easy_groups = [g for g in all_groups if prompt_mean_rewards[g["prompt_id"]] > median_reward]

        diff_metrics = analyze_group_metrics(difficult_groups)
        easy_metrics = analyze_group_metrics(easy_groups)

        results[f"K={k}"] = {
            "group_size": k,
            "overall": overall_metrics,
            "difficult_prompts (<= median)": diff_metrics,
            "easy_prompts (> median)": easy_metrics,
        }

    out_file = results_dir / "group_size_analysis.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'Group Size (K)':<15} | {'Informative Rate':<18} | {'Within-Group Std':<18} | {'Signal Variance':<16}")
    print("-" * 72)
    for k_key, data in results.items():
        ov = data["overall"]
        print(f"{k_key:<15} | {ov['informative_rate'] * 100:<17.2f}% | {ov['mean_within_group_std']:<18.4f} | {ov['relative_signal_variance']:<16.4f}")

    print("\n--- By Prompt Difficulty ---")
    for k_key, data in results.items():
        d_m = data["difficult_prompts (<= median)"]
        e_m = data["easy_prompts (> median)"]
        print(f"[{k_key}] Difficult (Hard): Info% = {d_m['informative_rate']*100:.1f}%, Std = {d_m['mean_within_group_std']:.4f}")
        print(f"[{k_key}] Easy:            Info% = {e_m['informative_rate']*100:.1f}%, Std = {e_m['mean_within_group_std']:.4f}")

    print(f"\nSaved analysis to: {out_file}\n")


if __name__ == "__main__":
    main()