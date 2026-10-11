from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.data import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg["results_dir"]) / "task5_feedback"

    gsm_file = results_dir / "gsm_eval.json"
    trans_file = results_dir / "transfer_eval.json"
    diag_file = results_dir / "perturbation_scores.json"

    if not gsm_file.exists() or not trans_file.exists() or not diag_file.exists():
        raise FileNotFoundError(
            "Please ensure score_perturbations.py and evaluate_math.py have completed.\n"
            f"Missing one of: {gsm_file.name}, {trans_file.name}, {diag_file.name}"
        )

    with gsm_file.open("r", encoding="utf-8") as f:
        gsm = json.load(f)
    with trans_file.open("r", encoding="utf-8") as f:
        trans = json.load(f)
    with diag_file.open("r", encoding="utf-8") as f:
        diag = json.load(f)

    # Compile Cross-Paradigm Comparison Table
    policies = ["sft", "rlvr", "rlaif"]
    rows = []

    for pol in policies:
        g_m = gsm["policy_metrics"][pol]
        t_m = trans["policy_metrics"][pol]
        transfer_delta = t_m["exact_accuracy"] - g_m["exact_accuracy"]

        rows.append({
            "Policy": pol.upper(),
            "GSM Acc": f"{g_m['exact_accuracy']*100:.2f}%",
            "GSM Fmt": f"{g_m['format_compliance_rate']*100:.1f}%",
            "Transfer Acc": f"{t_m['exact_accuracy']*100:.2f}%",
            "Transfer Drop": f"{transfer_delta*100:+.2f}%",
            "GSM Length": f"{g_m['mean_length']:.1f} ± {g_m['std_length']:.1f}",
        })

    print("\n" + "=" * 90)
    print(" TASK 5: RLVR vs. RLAIF CROSS-PARADIGM SUMMARY")
    print("=" * 90)
    print(f"{'Policy':<8} | {'GSM8K Acc':<12} | {'Format %':<10} | {'Transfer Acc':<14} | {'Transfer Delta':<16} | {'Length':<16}")
    print("-" * 90)
    for r in rows:
        print(f"{r['Policy']:<8} | {r['GSM Acc']:<12} | {r['GSM Fmt']:<10} | {r['Transfer Acc']:<14} | {r['Transfer Drop']:<16} | {r['GSM Length']:<16}")
    print("=" * 90)

    # Pairwise AI judge summary
    print("\nPairwise AI Judge Preference Rates (GSM8K):")
    pw = gsm["pairwise_preferences"]
    for pair, d in pw.items():
        parts = [f"{k}: {v*100:.1f}%" for k, v in d.items()]
        print(f"  • {pair.upper()}: " + " | ".join(parts))

    # Diagnostic highlights
    print("\nControlled Reward Perturbation Diagnostics:")
    print(f"{'Diagnostic Test':<24} | {'RLVR (A / B / Tie)':<24} | {'RLAIF (A / B / Tie)':<24}")
    print("-" * 76)
    for name, s in diag.items():
        rl = s["rlvr"]
        ai = s["rlaif"]
        rl_str = f"{rl['pref_A']*100:.1f}% / {rl['pref_B']*100:.1f}% / {rl['tie']*100:.1f}%"
        ai_str = f"{ai['pref_A']*100:.1f}% / {ai['pref_B']*100:.1f}% / {ai['tie']*100:.1f}%"
        print(f"{name:<24} | {rl_str:<24} | {ai_str:<24}")
    print("-" * 76)

    # Save consolidated comparison payload
    combined = {
        "gsm_eval": gsm,
        "transfer_eval": trans,
        "perturbation_scores": diag,
        "cross_paradigm_table": rows,
    }
    out_file = results_dir / "feedback_comparison_summary.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nSaved consolidated summary to: {out_file}\n")


if __name__ == "__main__":
    main()