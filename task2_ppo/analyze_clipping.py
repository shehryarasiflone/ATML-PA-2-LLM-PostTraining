from __future__ import annotations

import argparse
import json
import logging
import time
import warnings
from pathlib import Path

# Silence 8-bit quantized matrix multiplication warnings
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

import numpy as np
import torch
import torch.nn.functional as F

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import response_token_logprobs
from common.metrics import masked_mean
from common.models import clear_gpu, load_policy, load_tokenizer
from task2_ppo.continue_train import run_ppo
from task2_ppo.evaluate import evaluate_ppo
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


@torch.no_grad()
def analyze_cached_rollouts(cfg: dict, policy_adapter: str | None = None) -> dict:
    """Evaluate clipping statistics on the 32 fixed cached rollouts across epsilon values."""
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    eval_prompts = {p["prompt_id"]: p for p in read_jsonl(cfg["paths"]["rl_prompt_eval"])}
    tokenizer = load_tokenizer(cfg["base_model"])

    # Load policy to evaluate current token log-probabilities
    adapter = policy_adapter or cfg.get("output", "outputs/task2_ppo/standard")
    policy = load_policy(cfg, adapter_path=adapter, trainable=False)
    policy.eval()

    clip_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    gamma = float(cfg.get("gamma", 1.0))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    beta_kl = float(cfg.get("kl_beta", 0.10))

    print(f"\n=======================================================")
    print(f" Analyzing Cached Rollouts: {len(rows)} items")
    print(f" Policy Adapter: {adapter}")
    print(f" Epsilon values: {clip_values}")
    print(f"=======================================================\n")

    all_old_logp = []
    all_new_logp = []
    all_advantages = []
    all_masks = []

    for idx, row in enumerate(rows):
        prompt_id = row["prompt_id"]
        prompt_data = eval_prompts[prompt_id]
        messages = prompt_messages(prompt_data)

        # Reconstruct token sequence using chat template
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_width = len(prompt_ids)

        resp_text = row["response"]
        full_text = prompt_text + resp_text
        full_enc = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)

        input_ids = full_enc["input_ids"].cuda()
        attention_mask = full_enc["attention_mask"].cuda()
        resp_ids = input_ids[:, prompt_width:]
        resp_len = resp_ids.shape[1]

        # Forward pass to obtain current policy log-probabilities
        new_logp, _ = response_token_logprobs(
            model=policy,
            sequences=input_ids,
            attention_mask=attention_mask,
            prompt_width=prompt_width,
            response_ids=resp_ids,
        )
        new_logp = new_logp.squeeze(0).cpu()

        # Cached tensors
        old_logp = row["old_logprobs"][:resp_len].float()
        ref_logp = row["ref_logprobs"][:resp_len].float()
        values = row["values"][:resp_len].float()
        eff_reward = float(row.get("effective_terminal_reward", row.get("raw_terminal_reward", 0.0)))

        # Alignment check
        T = min(new_logp.shape[0], old_logp.shape[0])
        new_logp = new_logp[:T]
        old_logp = old_logp[:T]
        ref_logp = ref_logp[:T]
        values = values[:T]
        mask = torch.ones(1, T, dtype=torch.float32)

        # Compute token rewards and GAE advantages
        t_reward = torch.tensor([eff_reward], dtype=torch.float32)
        tok_rewards = shaped_rewards(
            task_reward=t_reward,
            policy_logp=old_logp.unsqueeze(0),
            ref_logp=ref_logp.unsqueeze(0),
            response_mask=mask,
            beta_kl=beta_kl,
        )
        adv, _ = compute_gae(
            rewards=tok_rewards,
            values=values.unsqueeze(0),
            mask=mask,
            gamma=gamma,
            lam=gae_lam,
        )
        norm_adv = normalize_advantages(adv, mask).squeeze(0)

        all_old_logp.append(old_logp)
        all_new_logp.append(new_logp)
        all_advantages.append(norm_adv)
        all_masks.append(mask.squeeze(0))

    cat_old_logp = torch.cat(all_old_logp)
    cat_new_logp = torch.cat(all_new_logp)
    cat_adv = torch.cat(all_advantages)

    log_ratio = (cat_new_logp.float() - cat_old_logp.float()).clamp(-10.0, 10.0)
    ratios = torch.exp(log_ratio)

    results_by_eps = {}
    print(f"{'Epsilon':<10} | {'Surrogate Loss':<16} | {'Clip Fraction':<16} | {'Affected Fraction':<18}")
    print("-" * 68)

    for eps in clip_values:
        surr1 = ratios * cat_adv
        surr2 = ratios.clamp(1.0 - eps, 1.0 + eps) * cat_adv
        obj = torch.minimum(surr1, surr2)
        surr_loss = -obj.mean().item()

        # Ratio strictly outside clipping boundary
        clipped_mask = (ratios < (1.0 - eps)) | (ratios > (1.0 + eps))
        clip_frac = clipped_mask.float().mean().item()

        # Clipping active in the pessimistic min objective
        affected_mask = ((ratios > (1.0 + eps)) & (cat_adv > 0)) | ((ratios < (1.0 - eps)) & (cat_adv < 0))
        affected_frac = affected_mask.float().mean().item()

        results_by_eps[str(eps)] = {
            "surrogate_loss": surr_loss,
            "clip_fraction": clip_frac,
            "affected_token_fraction": affected_frac,
            "ratio_mean": ratios.mean().item(),
            "ratio_std": ratios.std().item(),
            "ratio_min": ratios.min().item(),
            "ratio_max": ratios.max().item(),
        }

        print(f"{eps:<10.2f} | {surr_loss:<16.4f} | {clip_frac * 100:<15.2f}% | {affected_frac * 100:<17.2f}%")

    clear_gpu(policy)
    return results_by_eps


def run_clipping_forks(cfg: dict, config_path: str):
    """Run short 8-update PPO continuation forks for each epsilon from the common midpoint."""
    fork_updates = int(cfg.get("fork_updates", 8))
    clip_values = [float(e) for e in cfg.get("clip_values", [0.05, 0.20, 0.50])]
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    fork_results = {}

    for eps in clip_values:
        run_name = f"clip_eps_{eps:.2f}"
        out_adapter = f"outputs/task2_ppo/{run_name}"

        print(f"\n>>> Running Matched Fork: {run_name} (Updates: {fork_updates}, Epsilon: {eps})")
        run_ppo(
            config_path=config_path,
            output=out_adapter,
            updates=fork_updates,
            clip_epsilon=eps,
            kl_beta=float(cfg.get("kl_beta", 0.10)),
            run_name=run_name,
        )

        print(f"\n>>> Evaluating Fork: {run_name}")
        eval_metrics = evaluate_ppo(
            config_path=config_path,
            adapter_path=out_adapter,
            run_name=run_name,
            batch_size=2,
        )

        # Load training log to compute optimization stability statistics
        log_file = results_dir / f"{run_name}_train_log.json"
        grad_norms = []
        p_losses = []
        if log_file.exists():
            with log_file.open("r", encoding="utf-8") as f:
                history = json.load(f)
            grad_norms = [step["policy_grad_norm"] for step in history]
            p_losses = [step["policy_loss"] for step in history]

        fork_results[str(eps)] = {
            "eval_metrics": eval_metrics,
            "mean_policy_grad_norm": float(np.mean(grad_norms)) if grad_norms else 0.0,
            "std_policy_grad_norm": float(np.std(grad_norms)) if grad_norms else 0.0,
            "mean_policy_loss": float(np.mean(p_losses)) if p_losses else 0.0,
            "std_policy_loss": float(np.std(p_losses)) if p_losses else 0.0,
        }

    return fork_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--skip-forks", action="store_true", help="Only evaluate cached rollouts without running forks")
    ap.add_argument("--forks-only", action="store_true", help="Only run forks without cached rollout analysis")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if not args.forks_only:
        cached_stats = analyze_cached_rollouts(cfg)
        summary["cached_rollouts"] = cached_stats
        cached_file = results_dir / "clipping_cached_analysis.json"
        with cached_file.open("w", encoding="utf-8") as f:
            json.dump(cached_stats, f, indent=2)
        print(f"\nSaved cached rollout analysis to: {cached_file}")

    if not args.skip_forks:
        fork_stats = run_clipping_forks(cfg, args.config)
        summary["fork_results"] = fork_stats
        forks_file = results_dir / "clipping_forks_summary.json"
        with forks_file.open("w", encoding="utf-8") as f:
            json.dump(fork_stats, f, indent=2)
        print(f"\nSaved clipping forks summary to: {forks_file}")

    print("\n--- Clipping Study Complete ---")


if __name__ == "__main__":
    main()