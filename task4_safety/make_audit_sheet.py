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
    ap.add_argument("--policy", default="sft", help="Policy to audit (default: sft)")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    outdir = repo_path(cfg["results_dir"]) / "task4_safety"
    src = outdir / f"generated_{args.policy}.jsonl"
    if not src.exists():
        raise FileNotFoundError(f"Missing responses file: {src}")

    rows = read_jsonl(src)
    per_class = int(cfg.get("manual_audit_per_class", 30))
    seed = int(cfg.get("seed", 42))

    ids = fixed_audit_ids(rows, per_class, seed)
    audit_id_set = set(ids)

    # 1. Save minimal ID list
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(
        outdir / "manual_audit_ids.csv", index=False
    )
    print(f"Wrote fixed audit IDs ({len(ids)} rows) to: {outdir / 'manual_audit_ids.csv'}")

    # 2. Build the 60-sample audit sheet
    audit_records = []
    judged_file = outdir / f"judged_{args.policy}.jsonl"
    source_records = read_jsonl(judged_file) if judged_file.exists() else rows

    for r in source_records:
        if r["xstest_id"] in audit_id_set:
            audit_records.append({
                "xstest_id": r["xstest_id"],
                "policy": args.policy,
                "benchmark_class": r["benchmark_class"],
                "type": r["type"],
                "prompt": r["prompt"],
                "response": r["response"],
                "manual_label": "",  # Empty for blind human audit
                "ai_judge_label": r.get("judge_label", ""),
            })

    sheet_df = pd.DataFrame(audit_records)
    sheet_path = outdir / "manual_audit_sheet.csv"
    sheet_df.to_csv(sheet_path, index=False)
    print(f"Wrote 60-sample manual audit sheet to: {sheet_path}")


if __name__ == "__main__":
    main()