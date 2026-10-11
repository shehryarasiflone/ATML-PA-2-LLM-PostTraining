from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

from common.data import load_yaml, read_jsonl, repo_path
from task5_feedback.rlaif import PairwiseAIJudge
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}


def load_diagnostic_groups(path):
    rows = read_jsonl(repo_path(path))
    by_problem = defaultdict(dict)
    for row in rows:
        by_problem[str(row["problem_id"])][row["variant_type"]] = row
    for pid, variants in by_problem.items():
        missing = EXPECTED_VARIANTS - set(variants)
        if missing:
            raise ValueError(f"Problem {pid} missing variants: {sorted(missing)}")
    return by_problem


def evaluate_pair(judge: PairwiseAIJudge, problem: str, gold_final: str, var_a: dict, var_b: dict):
    resp_a = var_a["response"]
    resp_b = var_b["response"]

    # 1. RLVR Preference
    r_a = exact_reward(resp_a, gold_final)
    r_b = exact_reward(resp_b, gold_final)
    if r_a > r_b:
        rlvr_pref = "A"
    elif r_b > r_a:
        rlvr_pref = "B"
    else:
        rlvr_pref = "TIE"

    # 2. RLAIF Preference
    rlaif_pref = judge.compare(problem, resp_a, resp_b)

    return {
        "rlvr_pref": rlvr_pref,
        "rlaif_pref": rlaif_pref,
        "reward_a": r_a,
        "reward_b": r_b,
    }


def aggregate_pair_stats(results: list[dict]):
    n = len(results)
    if n == 0:
        return {}

    rlvr_a = sum(1 for r in results if r["rlvr_pref"] == "A") / n
    rlvr_b = sum(1 for r in results if r["rlvr_pref"] == "B") / n
    rlvr_tie = sum(1 for r in results if r["rlvr_pref"] == "TIE") / n

    rlaif_a = sum(1 for r in results if r["rlaif_pref"] == "A") / n
    rlaif_b = sum(1 for r in results if r["rlaif_pref"] == "B") / n
    rlaif_tie = sum(1 for r in results if r["rlaif_pref"] == "TIE") / n

    return {
        "n_problems": n,
        "rlvr": {"pref_A": rlvr_a, "pref_B": rlvr_b, "tie": rlvr_tie},
        "rlaif": {"pref_A": rlaif_a, "pref_B": rlaif_b, "tie": rlaif_tie},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_path = results_dir / "pairwise_judge_cache.json"

    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    print(f"\n=======================================================")
    print(f" Task 5: Controlled Feedback Diagnostics")
    print(f" Loaded {len(groups)} diagnostic problems (5 variants each)")
    print(f"=======================================================\n")

    judge = PairwiseAIJudge(cfg, cache_path)

    pairs_to_test = {
        "reasoning_sensitivity": ("clean_correct", "corrupt_reasoning_correct_final"),
        "outcome_sensitivity": ("clean_correct", "good_reasoning_wrong_final"),
        "filler_susceptibility": ("clean_correct", "persuasive_filler_correct"),
        "distractor_robustness": ("clean_correct", "gold_distractor_wrong_final"),
        "process_vs_outcome": ("good_reasoning_wrong_final", "corrupt_reasoning_correct_final"),
    }

    diagnostic_summary = {}

    for pair_name, (key_a, key_b) in pairs_to_test.items():
        pair_results = []
        for pid, variants in groups.items():
            problem = variants[key_a]["question"]
            gold_final = str(variants[key_a]["gold_final"])
            res = evaluate_pair(judge, problem, gold_final, variants[key_a], variants[key_b])
            pair_results.append(res)

        stats = aggregate_pair_stats(pair_results)
        stats["candidate_A"] = key_a
        stats["candidate_B"] = key_b
        diagnostic_summary[pair_name] = stats

    out_file = results_dir / "perturbation_scores.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(diagnostic_summary, f, indent=2)

    # Print summary table
    print("\n" + "=" * 80)
    print(" CONTROLLED REWARD PERTURBATION DIAGNOSTICS")
    print("=" * 80)
    print(f"{'Diagnostic Test':<24} | {'RLVR (A / B / Tie)':<24} | {'RLAIF (A / B / Tie)':<24}")
    print("-" * 80)

    for name, s in diagnostic_summary.items():
        rl = s["rlvr"]
        ai = s["rlaif"]
        rl_str = f"{rl['pref_A']*100:.1f}% / {rl['pref_B']*100:.1f}% / {rl['tie']*100:.1f}%"
        ai_str = f"{ai['pref_A']*100:.1f}% / {ai['pref_B']*100:.1f}% / {ai['tie']*100:.1f}%"
        print(f"{name:<24} | {rl_str:<24} | {ai_str:<24}")

    print("=" * 80)
    print(f"Results saved to: {out_file}\n")
    judge.close()


if __name__ == "__main__":
    main()