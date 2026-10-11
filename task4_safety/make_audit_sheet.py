from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].unique()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))

    rows = read_jsonl(src)
    per_class = int(cfg.get("manual_audit_per_class", 30))
    seed = int(cfg.get("seed", 42))

    ids = fixed_audit_ids(rows, per_class, seed)

    # 1. Save standard minimal manual audit ids file
    id_df = pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)})
    id_df.to_csv(outdir / "manual_audit_ids.csv", index=False)
    print(f"Wrote fixed audit IDs ({len(ids)} rows) to: {outdir / 'manual_audit_ids.csv'}")

    # 2. Save comprehensive audit sheets containing prompts and responses across all 4 policies
    audit_id_set = set(ids)
    comprehensive_records = []

    for pol in ["sft", "dpo", "ppo", "grpo"]:
        judged_file = outdir / f"judged_{pol}.jsonl"
        gen_file = outdir / f"generated_{pol}.jsonl"
        target_file = judged_file if judged_file.exists() else gen_file
        if not target_file.exists():
            continue
        p_rows = read_jsonl(target_file)
        for r in p_rows:
            if r["xstest_id"] in audit_id_set:
                comprehensive_records.append({
                    "xstest_id": r["xstest_id"],
                    "policy": pol,
                    "benchmark_class": r["benchmark_class"],
                    "type": r["type"],
                    "prompt": r["prompt"],
                    "response": r["response"],
                    "ai_judge_label": r.get("judge_label", ""),
                    "manual_label": "",  # To be audited
                })

    sheet_df = pd.DataFrame(comprehensive_records)
    sheet_path = outdir / "manual_audit_sheet.csv"
    sheet_df.to_csv(sheet_path, index=False)
    print(f"Wrote comprehensive audit sheet with responses to: {sheet_path}")


if __name__ == "__main__":
    main()