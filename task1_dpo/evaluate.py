from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common.data import (
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import (
    batch_generate,
    response_sequence_logprobs,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import sampled_kl
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)
from task1_dpo.dpo import dpo_loss
from task1_dpo.train import make_collate


def evaluate_preference_loss(
    policy,
    rows: list[dict],
    tokenizer,
    cfg: dict,
    beta: float,
    batch_size: int = 4,
) -> dict:
    """Evaluate DPO loss and preference accuracy across held-out pairs."""
    max_len = int(cfg["max_sequence_length"])
    collate = make_collate(tokenizer, max_len)
    loader = DataLoader(rows, batch_size=batch_size, shuffle=False, collate_fn=collate)

    total_loss = 0.0
    total_acc = 0.0
    policy_margins = []
    ref_margins = []
    num_batches = 0

    policy.eval()
    with torch.no_grad():
        for chosen_batch, rejected_batch in loader:
            if chosen_batch is None:
                continue

            chosen_batch = {k: v.to(policy.device) for k, v in chosen_batch.items()}
            rejected_batch = {k: v.to(policy.device) for k, v in rejected_batch.items()}

            # 1. Trainable Policy forward passes
            pol_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
            pol_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            # 2. Reference Policy forward passes (disabling adapter)
            with reference_mode(policy):
                ref_chosen_logp, _, _ = response_sequence_logprobs(policy, chosen_batch)
                ref_rejected_logp, _, _ = response_sequence_logprobs(policy, rejected_batch)

            # 3. DPO Loss & Margin
            loss, metrics = dpo_loss(
                policy_chosen_logp=pol_chosen_logp,
                policy_rejected_logp=pol_rejected_logp,
                ref_chosen_logp=ref_chosen_logp,
                ref_rejected_logp=ref_rejected_logp,
                beta=beta,
            )

            total_loss += loss.item()
            total_acc += metrics["preference_accuracy"].item()
            policy_margins.extend((pol_chosen_logp - pol_rejected_logp).cpu().tolist())
            ref_margins.extend((ref_chosen_logp - ref_rejected_logp).cpu().tolist())
            num_batches += 1

    return {
        "dpo_loss": total_loss / max(1, num_batches),
        "preference_accuracy": total_acc / max(1, num_batches),
        "mean_policy_margin": float(np.mean(policy_margins)) if policy_margins else 0.0,
        "mean_ref_margin": float(np.mean(ref_margins)) if ref_margins else 0.0,
    }


def evaluate_generations_and_rewards(
    policy,
    reward_model,
    reward_tok,
    rows: list[dict],
    tokenizer,
    cfg: dict,
    batch_size: int = 8,
) -> dict:
    """Generate responses on held-out prompts, compute reward score, KL, and length stats."""
    prompts = [prompt_messages_from_preference(r) for r in rows]
    max_prompt_len = int(cfg["max_sequence_length"])
    max_new_tokens = int(cfg.get("max_generation_tokens", 256))
    gen_cfg = cfg.get("generation", {})
    temperature = float(gen_cfg.get("temperature", 0.7))
    top_p = float(gen_cfg.get("top_p", 0.9))
    do_sample = bool(gen_cfg.get("do_sample", True))

    all_rewards = []
    all_lengths = []
    all_kls = []
    sample_outputs = []

    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i : i + batch_size]

        # 1. Generate completions under evaluation policy
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

        # 2. Token-level log-probs under Policy
        pol_tok_logp, _ = response_token_logprobs(
            model=policy,
            sequences=gen["sequences"],
            attention_mask=gen["attention_mask"],
            prompt_width=gen["prompt_width"],
            response_ids=gen["response_ids"],
        )

        # 3. Token-level log-probs under Reference
        with reference_mode(policy):
            ref_tok_logp, _ = response_token_logprobs(
                model=policy,
                sequences=gen["sequences"],
                attention_mask=gen["attention_mask"],
                prompt_width=gen["prompt_width"],
                response_ids=gen["response_ids"],
            )

        # 4. Token-level Sampled KL
        kl = sampled_kl(pol_tok_logp, ref_tok_logp, gen["response_mask"])
        all_kls.append(kl.item())

        # 5. Reward Model Scoring
        rewards = score_reward_pairs(
            rm_model=reward_model,
            rm_tokenizer=reward_tok,
            prompts=b_prompts,
            responses=gen["responses"],
        )
        all_rewards.extend(rewards.cpu().tolist())
        all_lengths.extend(gen["response_lengths"])

        # Cache initial qualitative samples for report analysis
        if len(sample_outputs) < 10:
            for prompt_i, resp_i, r_i, len_i in zip(
                b_prompts, gen["responses"], rewards.cpu().tolist(), gen["response_lengths"]
            ):
                if len(sample_outputs) < 10:
                    sample_outputs.append({
                        "prompt": prompt_i,
                        "response": resp_i,
                        "reward_score": round(float(r_i), 4),
                        "response_length": int(len_i),
                    })

    lengths_np = np.array(all_lengths)
    q75, q25 = np.percentile(lengths_np, [75, 25])
    iqr = float(q75 - q25)

    return {
        "sampled_kl": float(np.mean(all_kls)),
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "response_length_mean": float(np.mean(lengths_np)),
        "response_length_std": float(np.std(lengths_np)),
        "response_length_median": float(np.median(lengths_np)),
        "response_length_iqr": iqr,
        "sample_generations": sample_outputs,
    }


def evaluate_dpo(
    config_path: str,
    adapter_path: str,
    eval_dataset: str | None = None,
    run_name: str = "standard",
    output_dir: str | None = None,
) -> dict:
    cfg = load_yaml(config_path)
    eval_rows = read_jsonl(eval_dataset or cfg["paths"]["dpo_standard_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])
    def get_token_count(msgs):
        out = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        if isinstance(out, dict) or hasattr(out, "input_ids"):
            return len(out["input_ids"])
        return len(out)

    max_len = int(cfg["max_sequence_length"])
    eval_rows = [
        r for r in eval_rows
        if get_token_count(prompt_messages_from_preference(r)) < max_len
    ]
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    reward_model, reward_tok = load_reward_model(cfg)

    beta = float(cfg.get("beta", 0.10))

    print(f"Evaluating: {run_name} ({adapter_path})")
    print(f"Loaded {len(eval_rows)} evaluation examples.")

    pref_metrics = evaluate_preference_loss(policy, eval_rows, tokenizer, cfg, beta)
    gen_metrics = evaluate_generations_and_rewards(policy, reward_model, reward_tok, eval_rows, tokenizer, cfg)

    results = {
        "run_name": run_name,
        "adapter_path": adapter_path,
        "beta": beta,
        "eval_dataset": eval_dataset or cfg["paths"]["dpo_standard_eval"],
        **pref_metrics,
        **gen_metrics,
    }

    out_dir = repo_path(output_dir or cfg.get("results_dir", "results/task1_dpo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_name}_eval.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- Evaluation Summary ---")
    print(f"Held-out DPO Loss:        {results['dpo_loss']:.4f}")
    print(f"Held-out Preference Acc: {results['preference_accuracy'] * 100:.2f}%")
    print(f"Sampled KL from Ref:     {results['sampled_kl']:.4f}")
    print(f"Reward Model Score:      {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Response Length:         {results['response_length_mean']:.2f} ± {results['response_length_std']:.2f} (IQR: {results['response_length_iqr']:.2f})")
    print(f"Saved results to: {out_file}\n")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--dataset")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--output-dir")
    args = ap.parse_args()

    evaluate_dpo(
        config_path=args.config,
        adapter_path=args.adapter,
        eval_dataset=args.dataset,
        run_name=args.name,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()