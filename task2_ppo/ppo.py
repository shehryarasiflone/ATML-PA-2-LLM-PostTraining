from __future__ import annotations

import torch

from common.metrics import masked_mean


def compute_gae(rewards, values, mask, gamma=1.0, lam=0.95):
    """Token-level GAE over response positions strictly in float32."""
    batch, steps = rewards.shape
    advantages = torch.zeros((batch, steps), device=rewards.device, dtype=torch.float32)
    last_adv = torch.zeros(batch, device=rewards.device, dtype=torch.float32)

    r_f = torch.nan_to_num(rewards.float(), nan=0.0)
    v_f = torch.nan_to_num(values.float(), nan=0.0)
    m_f = mask.float()

    for t in reversed(range(steps)):
        current_valid = m_f[:, t]
        if t + 1 < steps:
            next_valid = m_f[:, t + 1]
            next_value = v_f[:, t + 1] * next_valid
        else:
            next_valid = torch.zeros_like(current_valid)
            next_value = torch.zeros_like(last_adv)

        delta = r_f[:, t] + gamma * next_value - v_f[:, t]
        last_adv = delta + gamma * lam * next_valid * last_adv
        last_adv = last_adv * current_valid
        advantages[:, t] = last_adv

    returns = advantages + v_f
    return advantages, returns


def shaped_rewards(task_reward, policy_logp, ref_logp, response_mask, beta_kl):
    """Sampled-action KL shaping plus terminal learned reward."""
    kl_diff = (policy_logp.float() - ref_logp.float()).clamp(-10.0, 10.0)
    rewards = -float(beta_kl) * kl_diff * response_mask.float()
    for b in range(rewards.shape[0]):
        valid = int(response_mask[b].sum().item())
        if valid > 0:
            rewards[b, valid - 1] += task_reward[b].float()
    return rewards


def ppo_policy_loss(new_logp, old_logp, advantage, mask, eps=0.2):
    """Numerically guarded PPO clipped policy loss."""
    log_ratio = (new_logp.float() - old_logp.float()).clamp(-10.0, 10.0)
    ratio = torch.exp(log_ratio)

    surr1 = ratio * advantage.float()
    surr2 = ratio.clamp(1.0 - eps, 1.0 + eps) * advantage.float()

    # Pessimistic lower bound
    objective = torch.minimum(surr1, surr2)

    loss = -masked_mean(objective, mask)
    affected = ((ratio < (1.0 - eps)) | (ratio > (1.0 + eps))).float()
    clip_fraction = masked_mean(affected, mask)
    return loss, ratio.detach(), clip_fraction.detach()


def value_mse_loss(predicted_values, returns, mask, clip_diff=10.0):
    """Compute value MSE with bounded residuals strictly in float32."""
    p_val = torch.nan_to_num(predicted_values.float(), nan=0.0)
    ret = torch.nan_to_num(returns.float(), nan=0.0)
    diff = (p_val - ret).clamp(-clip_diff, clip_diff)
    return masked_mean(diff ** 2, mask)


def normalize_advantages(advantages, mask, eps=1e-6):
    valid = advantages[mask.bool()]
    if valid.numel() <= 1:
        return advantages
    mean = valid.mean()
    std = valid.std(unbiased=False).clamp_min(eps)
    normed = ((advantages - mean) / std) * mask
    return normed.clamp(-5.0, 5.0) * mask