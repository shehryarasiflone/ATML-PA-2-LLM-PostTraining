from __future__ import annotations

import argparse
import gc
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
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.logging_utils import set_seed
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import (
    compute_group_stats,
    group_relative_advantages,
    grpo_policy_loss,
    mask_truncated_sequences,
)


def safe_clip_grad_norm(parameters, max_norm: float = 1.0) -> float:
    """Clip gradients safely in float32 to prevent float16 norm overflow."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    for p in params:
        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)

    total_norm_sq = torch.zeros(1, dtype=torch.float32, device=params[0].device)
    for p in params:
        p_norm = torch.norm(p.grad.detach().float(), 2)
        total_norm_sq += p_norm ** 2
    total_norm = torch.sqrt(total_norm_sq).item()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1.0:
        for p in params:
            p.grad.detach().mul_(clip_coef)

    return total_norm


def sanitize_model_weights(model):
    """Ensure no NaNs or Infs persist in model parameters."""
    for p in model.parameters():
        if p.requires_grad and (torch.isnan(p.data).any() or torch.isinf(p.data).any()):
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
):
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)

    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task3_grpo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    optimizer = bundle["optimizer"]
    prompt_rows = bundle["prompt_rows"]

    total_updates = int(cfg["updates"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    num_generations = int(cfg.get("num_generations", 4))
    clip_eps = float(cfg.get("clip_epsilon", 0.20))
    kl_beta = float(cfg.get("kl_beta", 0.10))
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_comp_len = int(cfg.get("max_completion_length", 512))
    mask_truncated = bool(cfg.get("mask_truncated_completions", True))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    print(f"\n=======================================================")
    print(f" Starting GRPO Continuation: {run_name}")
    print(f" Updates: {total_updates} | K={num_generations} | Loss: {loss_type} | KL Beta: {kl_beta}")
    print(f"=======================================================\n")

    history = []
    prompt_idx = 0

    for update_idx in range(1, total_updates + 1):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---------------------------------------------------------
        # 1. Rollout Generation: K completions per prompt
        # ---------------------------------------------------------
        batch_slice = [
            prompt_rows[(prompt_idx + i) % len(prompt_rows)]
            for i in range(prompts_per_update)
        ]
        prompt_idx += prompts_per_update

        # Expand each prompt K times
        expanded_prompts = []
        group_ids_list = []
        for g_i, p_row in enumerate(batch_slice):
            msg = prompt_messages(p_row)
            for _ in range(num_generations):
                expanded_prompts.append(msg)
                group_ids_list.append(g_i)

        group_ids = torch.tensor(group_ids_list, device="cuda" if torch.cuda.is_available() else "cpu")

        policy.eval()
        with torch.no_grad():
            gen = batch_generate(
                model=policy,
                tokenizer=tokenizer,
                prompts=expanded_prompts,
                max_prompt_length=max_prompt_len,
                max_new_tokens=max_comp_len,
                temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
                top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
                do_sample=True,
            )

            sequences = gen["sequences"].detach().clone()
            attention_mask = gen["attention_mask"].detach().clone()
            response_ids = gen["response_ids"].detach().clone()
            response_mask = gen["response_mask"].detach().clone()
            prompt_width = gen["prompt_width"]

            # Compute old policy log-probs
            old_logp, _ = response_token_logprobs(
                model=policy,
                sequences=sequences,
                attention_mask=attention_mask,
                prompt_width=prompt_width,
                response_ids=response_ids,
            )

            # Compute frozen reference log-probs
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    model=policy,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    prompt_width=prompt_width,
                    response_ids=response_ids,
                )

            # Mask truncated sequences that hit max length without EOS
            if mask_truncated:
                truncated = [not t for t in gen["terminated_with_eos"]]
                effective_mask = mask_truncated_sequences(response_mask, truncated)
            else:
                effective_mask = response_mask

            # Score responses with Reward Model
            rewards = score_reward_pairs(
                rm_model=reward_model,
                rm_tokenizer=reward_tokenizer,
                prompts=expanded_prompts,
                responses=gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1280)),
            )

            # Within-group relative advantages and group stats
            advantages = group_relative_advantages(rewards, group_ids)
            grp_stats = compute_group_stats(rewards, group_ids)

        # ---------------------------------------------------------
        # 2. Optimization Step
        # ---------------------------------------------------------
        policy.train()
        new_logp, _ = response_token_logprobs(
            model=policy,
            sequences=sequences,
            attention_mask=attention_mask,
            prompt_width=prompt_width,
            response_ids=response_ids,
        )

        loss, diag = grpo_policy_loss(
            new_logp=new_logp,
            old_logp=old_logp.detach(),
            seq_adv=advantages.detach(),
            token_mask=effective_mask,
            ref_logp=ref_logp.detach(),
            eps=clip_eps,
            beta=kl_beta,
            loss_type=loss_type,
            max_completion_length=max_comp_len,
        )

        optimizer.zero_grad()
        loss.backward()
        pol_norm = safe_clip_grad_norm(trainable_parameters(policy), max_grad_norm)
        optimizer.step()
        sanitize_model_weights(policy)

        # Memory cleanup
        del sequences, attention_mask, response_ids, response_mask, effective_mask
        del old_logp, ref_logp, new_logp, rewards, advantages
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - start_time
        peak_vram = (
            torch.cuda.max_memory_allocated() / (1024 * 1024)
            if torch.cuda.is_available()
            else 0.0
        )
        resp_len = float(sum(gen["response_lengths"])) / max(1, len(gen["response_lengths"]))
        mean_reward = float(rewards.mean().item()) if "rewards" in locals() else float(grp_stats["mean_within_group_std"])

        step_record = {
            "update": update_idx,
            "loss": loss.item(),
            "policy_term": diag["policy_term"].item(),
            "sampled_kl": diag["sampled_kl"].item(),
            "clip_fraction": diag["clip_fraction"].item(),
            "entropy": diag["sample_entropy"].item(),
            "policy_grad_norm": pol_norm,
            "mean_reward": mean_reward,
            "within_group_reward_std": grp_stats["mean_within_group_std"],
            "uninformative_group_fraction": grp_stats["uninformative_fraction"],
            "response_length": resp_len,
            "wall_clock_time": elapsed,
            "peak_vram_mb": peak_vram,
        }
        history.append(step_record)

        print(
            f"Update {update_idx:02d}/{total_updates:02d} | "
            f"Loss: {loss.item():+.4f} | "
            f"KL: {diag['sampled_kl'].item():.4f} | "
            f"GrpStd: {grp_stats['mean_within_group_std']:.3f} | "
            f"Uninf%: {grp_stats['uninformative_fraction'] * 100:.1f}% | "
            f"GradNorm: {pol_norm:.3f} | "
            f"Len: {resp_len:.1f} | "
            f"VRAM: {peak_vram:.0f}MB | "
            f"Time: {elapsed:.2f}s"
        )

    # Save policy adapter checkpoint
    print(f"\nSaving final GRPO checkpoint to: {out}")
    policy.save_pretrained(out)

    log_file = results_dir / f"{run_name}_train_log.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved update metrics to: {log_file}\n")

    clear_gpu(policy, reward_model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name)


if __name__ == "__main__":
    main()