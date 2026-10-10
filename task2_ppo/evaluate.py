from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import (
    batch_generate,
    response_token_logprobs,
    score_reward_pairs,
)
from common.metrics import sampled_kl
from common.models import (
    clear_gpu,
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
)


@torch.no_grad()
def evaluate_ppo(
    config_path: str,
    adapter_path: str,
    run_name: str = "standard",
    output_dir: str | None = None,
    batch_size: int = 8,  # Increased from 2 to 8 for ~4x faster evaluation
) -> dict:
    cfg = load_yaml(config_path)
    rows = read_jsonl(cfg["paths"]["rl_prompt_eval"])
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=adapter_path, trainable=False)
    reward_model, reward_tok = load_reward_model(cfg)

    prompts = [prompt_messages(r) for r in rows]
    max_prompt_len = int(cfg.get("max_prompt_length", 256))
    max_new_tokens = int(cfg.get("eval_max_response_length", 768))
    temperature = float(cfg.get("generation", {}).get("temperature", 0.7))
    top_p = float(cfg.get("generation", {}).get("top_p", 0.9))

    print(f"\nEvaluating PPO Checkpoint: {run_name} ({adapter_path})")
    print(f"Loaded {len(prompts)} held-out RL evaluation prompts (Batch size: {batch_size}).")

    all_rewards = []
    all_lengths = []
    all_kls = []
    all_terminated = []
    sample_outputs = []

    total_prompts = len(prompts)
    for i in range(0, total_prompts, batch_size):
        b_prompts = prompts[i : i + batch_size]

        gen = batch_generate(
            model=policy,
            tokenizer=tokenizer,
            prompts=b_prompts,
            max_prompt_length=max_prompt_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        sequences = gen["sequences"].clone()
        attention_mask = gen["attention_mask"].clone()
        response_ids = gen["response_ids"].clone()
        response_mask = gen["response_mask"].clone()
        prompt_width = gen["prompt_width"]

        # Policy Log-probs
        pol_tok_logp, _ = response_token_logprobs(
            model=policy,
            sequences=sequences,
            attention_mask=attention_mask,
            prompt_width=prompt_width,
            response_ids=response_ids,
        )

        # Reference Log-probs
        with reference_mode(policy):
            ref_tok_logp, _ = response_token_logprobs(
                model=policy,
                sequences=sequences,
                attention_mask=attention_mask,
                prompt_width=prompt_width,
                response_ids=response_ids,
            )

        # Sampled KL
        kl = sampled_kl(pol_tok_logp, ref_tok_logp, response_mask)
        all_kls.append(kl.item())

        # Reward Model Score
        rewards = score_reward_pairs(
            rm_model=reward_model,
            rm_tokenizer=reward_tok,
            prompts=b_prompts,
            responses=gen["responses"],
            max_length=int(cfg.get("reward_max_length", 1280)),
        )
        all_rewards.extend(rewards.cpu().tolist())
        all_lengths.extend(gen["response_lengths"])
        all_terminated.extend(gen["terminated_with_eos"])

        if len(sample_outputs) < 8:
            for p_i, r_i, s_i, l_i in zip(
                b_prompts, gen["responses"], rewards.cpu().tolist(), gen["response_lengths"]
            ):
                if len(sample_outputs) < 8:
                    sample_outputs.append({
                        "prompt": p_i,
                        "response": r_i,
                        "reward": round(float(s_i), 4),
                        "length": int(l_i),
                    })

        # Regular visible progress updates every ~24-32 prompts
        processed = min(i + len(b_prompts), total_prompts)
        if processed % 24 == 0 or processed == total_prompts:
            print(f"Evaluated {processed:03d} / {total_prompts} prompts...")

        del gen, sequences, attention_mask, response_ids, response_mask, pol_tok_logp, ref_tok_logp, rewards
        torch.cuda.empty_cache()

    lengths_np = np.array(all_lengths)
    q75, q25 = np.percentile(lengths_np, [75, 25])
    iqr = float(q75 - q25)

    results = {
        "run_name": run_name,
        "adapter_path": adapter_path,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "sampled_kl": float(np.mean(all_kls)),
        "response_length_mean": float(np.mean(lengths_np)),
        "response_length_std": float(np.std(lengths_np)),
        "response_length_median": float(np.median(lengths_np)),
        "response_length_iqr": iqr,
        "eos_termination_rate": float(np.mean(all_terminated)),
        "sample_outputs": sample_outputs,
    }

    out_dir = repo_path(output_dir or cfg.get("results_dir", "results/task2_ppo"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{run_name}_eval.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n--- PPO Evaluation Summary ---")
    print(f"Reward Model Score:      {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    print(f"Sampled KL from Ref:     {results['sampled_kl']:.4f}")
    print(f"Response Length:         {results['response_length_mean']:.2f} ± {results['response_length_std']:.2f} (IQR: {results['response_length_iqr']:.2f})")
    print(f"EOS Termination Rate:    {results['eos_termination_rate'] * 100:.2f}%")
    print(f"Saved results to: {out_file}\n")

    clear_gpu(policy, reward_model)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--output-dir")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()
    evaluate_ppo(
        config_path=args.config,
        adapter_path=args.adapter,
        run_name=args.name,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()