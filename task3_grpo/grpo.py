from __future__ import annotations

import numpy as np
import torch

from common.metrics import masked_mean, sampled_kl, sample_entropy


def group_relative_advantages(rewards: torch.Tensor, group_ids: torch.Tensor, eps: float = 1e-6):
    """Return one scalar advantage per sampled completion normalized within each prompt group."""
    advantages = torch.zeros_like(rewards, dtype=torch.float32)
    rewards_f = rewards.float()

    unique_groups = torch.unique(group_ids)
    for gid in unique_groups:
        mask = group_ids == gid
        g_rewards = rewards_f[mask]
        mean = g_rewards.mean()
        std = g_rewards.std(unbiased=False)
        std_clamped = std.clamp_min(eps)
        advantages[mask] = (g_rewards - mean) / std_clamped

    return advantages


def compute_group_stats(rewards: torch.Tensor, group_ids: torch.Tensor, tol: float = 1e-5):
    """Compute within-group reward standard deviation and uninformative group fraction."""
    unique_groups = torch.unique(group_ids)
    stds = []
    uninformative = []
    rewards_f = rewards.float()
    for gid in unique_groups:
        mask = group_ids == gid
        g_std = rewards_f[mask].std(unbiased=False).item()
        stds.append(g_std)
        uninformative.append(float(g_std <= tol))
    return {
        "mean_within_group_std": float(np.mean(stds)) if stds else 0.0,
        "uninformative_fraction": float(np.mean(uninformative)) if uninformative else 0.0,
    }


def grpo_policy_loss(
    new_logp,
    old_logp,
    seq_adv,
    token_mask,
    ref_logp,
    eps,
    beta,
    loss_type="grpo",
    max_completion_length: int | None = None,
):
    """PPO-style clipped GRPO loss for already-sampled completions with numerical guards."""
    log_ratio = (new_logp.float() - old_logp.float()).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)
    adv = seq_adv[:, None].float()
    s1 = ratio * adv
    s2 = ratio.clamp(1.0 - eps, 1.0 + eps) * adv
    objective = torch.minimum(s1, s2)

    token_sum = (objective * token_mask.float()).sum(-1)
    if loss_type == "grpo":
        denom = token_mask.float().sum(-1).clamp_min(1.0)
        per_sequence = token_sum / denom
        policy_term = -per_sequence.mean()
    elif loss_type == "dr_grpo":
        if max_completion_length is None:
            raise ValueError("dr_grpo requires max_completion_length")
        # Constant normalization rather than dividing by each response's realized length
        per_sequence = token_sum / float(max_completion_length)
        policy_term = -per_sequence.mean()
    else:
        raise ValueError(f"Unknown loss_type={loss_type!r}")

    log_ratio_ref_over_policy = (ref_logp.float() - new_logp.float()).clamp(-10.0, 10.0)
    per_token_kl = torch.exp(log_ratio_ref_over_policy) - log_ratio_ref_over_policy - 1.0
    kl = masked_mean(per_token_kl, token_mask)
    loss = policy_term + float(beta) * kl
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    return loss, {
        "policy_term": policy_term.detach(),
        "sampled_kl": kl.detach(),
        "clip_fraction": masked_mean(affected, token_mask).detach(),
        "ratio_mean": masked_mean(ratio.detach(), token_mask),
        "sample_entropy": sample_entropy(new_logp.detach(), token_mask),
    }


def mask_truncated_sequences(token_mask: torch.Tensor, truncated: list[bool] | torch.Tensor):
    """Zero-out loss mask for sequences that hit max completion length without EOS."""
    truncated = torch.as_tensor(truncated, device=token_mask.device, dtype=torch.bool)
    keep = (~truncated).to(token_mask.dtype)[:, None]
    return token_mask * keep