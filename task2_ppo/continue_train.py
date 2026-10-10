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
from common.metrics import masked_mean
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    token_values,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


def safe_clip_grad_norm(parameters, max_norm: float = 1.0) -> float:
    """Clip gradients safely in float32 to prevent float16 norm overflow to inf/nan."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return 0.0

    # Sanitize wild values before norm calculation
    for p in params:
        p.grad.data = torch.nan_to_num(p.grad.data, nan=0.0, posinf=1.0, neginf=-1.0)

    # Compute Euclidean norm in float32
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
    """Defensive sanitation to ensure no NaNs or Infs persist in model parameters."""
    for p in model.parameters():
        if p.requires_grad and (torch.isnan(p.data).any() or torch.isinf(p.data).any()):
            p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)


def prepare_ppo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "lora_head"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
        eps=1e-5,
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
        eps=1e-5,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
    }


def run_ppo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    clip_epsilon: float | None = None,
    kl_beta: float | None = None,
    run_name: str = "standard",
):
    bundle = prepare_ppo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)

    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    results_dir = repo_path(cfg.get("results_dir", "results/task2_ppo"))
    results_dir.mkdir(parents=True, exist_ok=True)

    policy = bundle["policy"]
    value_model = bundle["value_model"]
    reward_model = bundle["reward_model"]
    reward_tokenizer = bundle["reward_tokenizer"]
    tokenizer = bundle["tokenizer"]
    policy_optimizer = bundle["policy_optimizer"]
    value_optimizer = bundle["value_optimizer"]
    prompt_rows = bundle["prompt_rows"]

    total_updates = int(cfg["updates"])
    prompts_per_update = int(cfg.get("prompts_per_update", 1))
    ppo_epochs = int(cfg.get("ppo_epochs", 2))
    clip_eps = float(cfg["clip_epsilon"])
    beta_kl = float(cfg["kl_beta"])
    gamma = float(cfg.get("gamma", 1.0))
    gae_lam = float(cfg.get("gae_lambda", 0.95))
    value_coef = float(cfg.get("value_coef", 0.50))
    missing_eos_pen = float(cfg.get("missing_eos_penalty", 1.0))
    max_grad_norm = float(cfg.get("max_grad_norm", 1.0))

    print(f"\n=======================================================")
    print(f" Starting PPO Continuation: {run_name}")
    print(f" Updates: {total_updates} | Epsilon: {clip_eps} | KL Beta: {beta_kl}")
    print(f"=======================================================\n")

    history = []
    prompt_idx = 0

    for update_idx in range(1, total_updates + 1):
        start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # ---------------------------------------------------------
        # 1. Rollout Collection
        # ---------------------------------------------------------
        batch_slice = [
            prompt_rows[(prompt_idx + i) % len(prompt_rows)]
            for i in range(prompts_per_update)
        ]
        prompt_idx += prompts_per_update
        batch_prompts = [prompt_messages(r) for r in batch_slice]

        policy.eval()
        value_model.eval()

        with torch.no_grad():
            gen = batch_generate(
                model=policy,
                tokenizer=tokenizer,
                prompts=batch_prompts,
                max_prompt_length=int(cfg["max_prompt_length"]),
                max_new_tokens=int(cfg["max_response_length"]),
                temperature=float(cfg.get("generation", {}).get("temperature", 0.7)),
                top_p=float(cfg.get("generation", {}).get("top_p", 0.9)),
                do_sample=True,
            )

            if gen["response_ids"].shape[1] == 0:
                print(f"Update {update_idx:02d}: Empty rollout, continuing.")
                continue

            sequences = gen["sequences"].detach().clone()
            attention_mask = gen["attention_mask"].detach().clone()
            response_ids = gen["response_ids"].detach().clone()
            response_mask = gen["response_mask"].detach().clone()
            prompt_width = gen["prompt_width"]

            # Policy Log-probs
            old_logp, _ = response_token_logprobs(
                model=policy,
                sequences=sequences,
                attention_mask=attention_mask,
                prompt_width=prompt_width,
                response_ids=response_ids,
            )

            # Reference Log-probs
            with reference_mode(policy):
                ref_logp, _ = response_token_logprobs(
                    model=policy,
                    sequences=sequences,
                    attention_mask=attention_mask,
                    prompt_width=prompt_width,
                    response_ids=response_ids,
                )

            # Value Estimates
            full_values = token_values(value_model, sequences, attention_mask)
            old_values = full_values[:, prompt_width - 1 : -1][:, : response_ids.shape[1]]

            # Reward Model Scoring
            raw_rewards = score_reward_pairs(
                rm_model=reward_model,
                rm_tokenizer=reward_tokenizer,
                prompts=batch_prompts,
                responses=gen["responses"],
                max_length=int(cfg.get("reward_max_length", 1280)),
            )

            # Terminal Reward shaping + EOS penalty
            terminal_rewards = raw_rewards.clone()
            for b_i, terminated in enumerate(gen["terminated_with_eos"]):
                if not terminated:
                    terminal_rewards[b_i] -= missing_eos_pen

            terminal_rewards = terminal_rewards.clamp(-15.0, 15.0)

            token_rewards = shaped_rewards(
                task_reward=terminal_rewards,
                policy_logp=old_logp,
                ref_logp=ref_logp,
                response_mask=response_mask,
                beta_kl=beta_kl,
            )

            # GAE in float32
            advantages, returns = compute_gae(
                rewards=token_rewards,
                values=old_values,
                mask=response_mask,
                gamma=gamma,
                lam=gae_lam,
            )
            norm_advantages = normalize_advantages(advantages, response_mask)

        # ---------------------------------------------------------
        # 2. Optimization Epochs
        # ---------------------------------------------------------
        policy.train()
        value_model.train()

        update_p_loss, update_v_loss = 0.0, 0.0
        update_clip_frac, update_entropy = 0.0, 0.0
        update_ratio_mean = 0.0
        pol_norm, val_norm = 0.0, 0.0

        for _ in range(ppo_epochs):
            # Forward pass: Policy
            new_logp, logits = response_token_logprobs(
                model=policy,
                sequences=sequences,
                attention_mask=attention_mask,
                prompt_width=prompt_width,
                response_ids=response_ids,
            )

            p_loss, ratio, clip_frac = ppo_policy_loss(
                new_logp=new_logp,
                old_logp=old_logp.detach(),
                advantage=norm_advantages.detach(),
                mask=response_mask,
                eps=clip_eps,
            )

            # Entropy
            probs = F.softmax(logits.float(), dim=-1)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            plogp = torch.where(probs > 0, probs * log_probs, torch.zeros_like(probs))
            token_entropy = -plogp.sum(dim=-1)
            entropy = masked_mean(token_entropy, response_mask)

            # Forward pass: Value Model
            pred_full_values = token_values(value_model, sequences, attention_mask)
            pred_values = pred_full_values[:, prompt_width - 1 : -1][:, : response_ids.shape[1]]
            v_loss = value_mse_loss(pred_values, returns.detach(), response_mask)

            # Policy Optimization with Safe Clipping
            policy_optimizer.zero_grad()
            p_loss.backward()
            pol_norm = safe_clip_grad_norm(trainable_parameters(policy), max_grad_norm)
            policy_optimizer.step()
            sanitize_model_weights(policy)

            # Value Optimization with Safe Clipping
            value_optimizer.zero_grad()
            (value_coef * v_loss).backward()
            val_norm = safe_clip_grad_norm(value_model.parameters(), max_grad_norm)
            value_optimizer.step()
            sanitize_model_weights(value_model)

            update_p_loss += p_loss.item() / ppo_epochs
            update_v_loss += v_loss.item() / ppo_epochs
            update_clip_frac += clip_frac.item() / ppo_epochs
            update_entropy += entropy.item() / ppo_epochs
            update_ratio_mean += masked_mean(ratio, response_mask).item() / ppo_epochs

        # Free rollout memory
        del sequences, attention_mask, response_ids, response_mask, old_logp, ref_logp
        del old_values, raw_rewards, token_rewards, advantages, returns, norm_advantages
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
        raw_r_mean = float(terminal_rewards.mean().item())

        step_record = {
            "update": update_idx,
            "policy_loss": update_p_loss,
            "value_loss": update_v_loss,
            "clip_fraction": update_clip_frac,
            "ratio_mean": update_ratio_mean,
            "entropy": update_entropy,
            "policy_grad_norm": pol_norm,
            "value_grad_norm": val_norm,
            "raw_reward": raw_r_mean,
            "response_length": resp_len,
            "wall_clock_time": elapsed,
            "peak_vram_mb": peak_vram,
        }
        history.append(step_record)

        print(
            f"Update {update_idx:02d}/{total_updates:02d} | "
            f"P Loss: {update_p_loss:+.4f} | "
            f"V Loss: {update_v_loss:.4f} | "
            f"Clip%: {update_clip_frac * 100:.1f}% | "
            f"Rew: {raw_r_mean:+.3f} | "
            f"Len: {resp_len:.1f} | "
            f"VRAM: {peak_vram:.0f}MB | "
            f"Time: {elapsed:.2f}s"
        )

    # Save policy checkpoint
    print(f"\nSaving final policy checkpoint to: {out}")
    policy.save_pretrained(out)

    # Save training trajectory log
    log_file = results_dir / f"{run_name}_train_log.json"
    with log_file.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved update metrics to: {log_file}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    args = ap.parse_args()
    run_ppo(
        config_path=args.config,
        output=args.output,
        updates=args.updates,
        clip_epsilon=args.clip_epsilon,
        kl_beta=args.kl_beta,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()