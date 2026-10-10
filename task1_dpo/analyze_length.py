from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import batch_generate
from common.metrics import word_count, word_limit_compliance
from common.models import clear_gpu, load_policy, load_tokenizer
from task1_dpo.evaluate import evaluate_preference_loss
from task1_dpo.train import _get_prompt_token_count, run_training


@torch.no_grad()
def evaluate_word_limits(
    policy,
    tokenizer,
    cfg: dict,
    wl_rows: list[dict],
    batch_size: int = 4,
) -> dict:
    """Generate completions on word-limit prompts and evaluate strict compliance."""
    prompts = [prompt_messages(r) for r in wl_rows]
    max_prompt_len = int(cfg["max_sequence_length"])
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))
    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))

    compliances = []
    word_counts = []
    sample_outputs = []

    total = len(prompts)
    for i in range(0, total, batch_size):
        b_prompts = prompts[i : i + batch_size]
        b_rows = wl_rows[i : i + batch_size]

        gen = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=b_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

        for row, resp in zip(b_rows, gen["responses"]):
            p_msgs = prompt_messages(row)
            user_text = p_msgs[-1]["content"] if p_msgs else ""
            comp = word_limit_compliance(user_text, resp)
            wc = word_count(resp)
            if comp is not None:
                compliances.append(comp)
            word_counts.append(wc)

            if len(sample_outputs) < 5:
                sample_outputs.append({
                    "prompt": user_text,
                    "response": resp,
                    "word_count": wc,
                    "compliant": bool(comp) if comp is not None else None,
                })

        del gen
        torch.cuda.empty_cache()

    return {
        "compliance_rate": float(np.mean(compliances)) if compliances else 0.0,
        "mean_word_count": float(np.mean(word_counts)) if word_counts else 0.0,
        "std_word_count": float(np.std(word_counts)) if word_counts else 0.0,
        "sample_responses": sample_outputs,
    }


def evaluate_model_on_strata_and_limits(
    adapter_path: str,
    cfg: dict,
    tokenizer,
    stratified_rows: list[dict],
    wl_rows: list[dict],
) -> dict:
    """Evaluate one model adapter on all 3 length strata and the word-limit dataset."""
    print(f"\nLoading policy adapter from: {adapter_path}")
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    beta = float(cfg.get("beta", 0.10))

    strata_names = ["preferred_longer", "length_matched", "rejected_longer"]
    strata_results = {}

    print("Evaluating stratified preference accuracy...")
    for s in strata_names:
        s_rows = [r for r in stratified_rows if r.get("length_stratum") == s]
        if not s_rows:
            continue
        metrics = evaluate_preference_loss(
            policy=policy,
            rows=s_rows,
            tokenizer=tokenizer,
            cfg=cfg,
            beta=beta,
            batch_size=2,
        )
        strata_results[s] = {
            "count": len(s_rows),
            "preference_accuracy": metrics["preference_accuracy"],
            "dpo_loss": metrics["dpo_loss"],
            "policy_margin": metrics["mean_policy_margin"],
        }
        print(f"  [{s}] (n={len(s_rows)}): Acc = {metrics['preference_accuracy']*100:.2f}%, Loss = {metrics['dpo_loss']:.4f}")

    # Overall stratified evaluation
    overall_metrics = evaluate_preference_loss(
        policy=policy,
        rows=stratified_rows,
        tokenizer=tokenizer,
        cfg=cfg,
        beta=beta,
        batch_size=2,
    )
    strata_results["overall"] = {
        "count": len(stratified_rows),
        "preference_accuracy": overall_metrics["preference_accuracy"],
        "dpo_loss": overall_metrics["dpo_loss"],
        "policy_margin": overall_metrics["mean_policy_margin"],
    }
    print(f"  [overall] (n={len(stratified_rows)}): Acc = {overall_metrics['preference_accuracy']*100:.2f}%, Loss = {overall_metrics['dpo_loss']:.4f}")

    print("Evaluating word-limit compliance...")
    wl_results = evaluate_word_limits(policy, tokenizer, cfg, wl_rows, batch_size=4)
    print(f"  Compliance Rate: {wl_results['compliance_rate']*100:.2f}%, Mean Words: {wl_results['mean_word_count']:.1f}")

    clear_gpu(policy)

    return {
        "adapter_path": adapter_path,
        "strata": strata_results,
        "word_limits": wl_results,
    }


def run_length_study(config_path: str, skip_train: bool = False):
    cfg = load_yaml(config_path)
    std_adapter = cfg.get("standard_output", "outputs/task1_dpo/standard")
    length_adapter = cfg.get("length_output", "outputs/task1_dpo/length_balanced")
    results_dir = repo_path(cfg.get("results_dir", "results/task1_dpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Train length-balanced model if needed
    length_ckpt = repo_path(length_adapter)
    if not skip_train and not (length_ckpt / "adapter_config.json").exists():
        print("=== Training Length-Balanced DPO Model ===")
        run_training(
            config_path=config_path,
            run_name="length_balanced",
            dataset_path=cfg["paths"]["dpo_length_train"],
            output_path=length_adapter,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        print(f"Found existing length-balanced checkpoint at: {length_adapter}")

    # 2. Load evaluation datasets
    tokenizer = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])

    stratified_path = cfg["paths"]["dpo_length_eval"]
    raw_stratified_rows = read_jsonl(stratified_path)
    stratified_rows = [
        r for r in raw_stratified_rows
        if _get_prompt_token_count(tokenizer, prompt_messages_from_preference(r)) < max_len
    ]
    print(f"Loaded {len(stratified_rows)} valid stratified eval pairs.")

    wl_path = cfg["paths"]["word_limit_prompts"]
    wl_rows = read_jsonl(wl_path)
    print(f"Loaded {len(wl_rows)} word-limit prompts.")

    # 3. Evaluate Standard DPO
    print("\n" + "=" * 50)
    print(" Evaluating Condition A: Standard DPO")
    print("=" * 50)
    std_results = evaluate_model_on_strata_and_limits(
        adapter_path=std_adapter,
        cfg=cfg,
        tokenizer=tokenizer,
        stratified_rows=stratified_rows,
        wl_rows=wl_rows,
    )

    # 4. Evaluate Length-Balanced DPO
    print("\n" + "=" * 50)
    print(" Evaluating Condition B: Length-Balanced DPO")
    print("=" * 50)
    length_results = evaluate_model_on_strata_and_limits(
        adapter_path=length_adapter,
        cfg=cfg,
        tokenizer=tokenizer,
        stratified_rows=stratified_rows,
        wl_rows=wl_rows,
    )

    # 5. Save structured results
    combined_results = {
        "standard_dpo": std_results,
        "length_balanced_dpo": length_results,
    }
    summary_path = results_dir / "length_study_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    # 6. Comparative Presentation Tables
    print("\n" + "=" * 76)
    print("             LENGTH STRATIFICATION ACCURACY COMPARISON (%)")
    print("=" * 76)
    print(f"| {'Stratum':^22} | {'Standard DPO':^22} | {'Length-Balanced DPO':^23} |")
    print(f"|{'-'*24}|{'-'*24}|{'-'*25}|")
    for s in ["preferred_longer", "length_matched", "rejected_longer", "overall"]:
        std_acc = std_results["strata"][s]["preference_accuracy"] * 100.0
        bal_acc = length_results["strata"][s]["preference_accuracy"] * 100.0
        label = s.replace("_", " ").title()
        print(f"| {label:^22} | {std_acc:^22.2f} | {bal_acc:^23.2f} |")
    print("=" * 76)

    print("\n" + "=" * 76)
    print("             WORD-LIMIT INSTRUCTION COMPLIANCE & LENGTH")
    print("=" * 76)
    print(f"| {'Metric':^24} | {'Standard DPO':^22} | {'Length-Balanced DPO':^23} |")
    print(f"|{'-'*26}|{'-'*24}|{'-'*25}|")
    std_wl = std_results["word_limits"]
    bal_wl = length_results["word_limits"]
    print(f"| {'Compliance Rate (%)':^24} | {std_wl['compliance_rate']*100:^22.2f} | {bal_wl['compliance_rate']*100:^23.2f} |")
    print(f"| {'Mean Response Words':^24} | {std_wl['mean_word_count']:^22.1f} | {bal_wl['mean_word_count']:^23.1f} |")
    print(f"| {'Std Response Words':^24} | {std_wl['std_word_count']:^22.1f} | {bal_wl['std_word_count']:^23.1f} |")
    print("=" * 76)
    print(f"Saved full study results to: {summary_path}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    run_length_study(args.config, skip_train=args.skip_train)


if __name__ == "__main__":
    main()