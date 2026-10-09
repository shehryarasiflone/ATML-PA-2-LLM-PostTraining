from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer, trainable_parameters
from task1_dpo.dpo import dpo_loss


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg,
        "rows": rows,
        "tokenizer": tokenizer,
        "model": model,
        "loader": loader,
        "optimizer": optimizer,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def get_batch_logps(
    logits: torch.Tensor, 
    input_ids: torch.Tensor, 
    response_mask: torch.Tensor
) -> torch.Tensor:
    """
    Computes the sum of response-token log-probabilities under teacher forcing.
    logits: [batch_size, seq_len, vocab_size]
    input_ids: [batch_size, seq_len]
    response_mask: [batch_size, seq_len] (0 for pad/prompt, 1 for response)
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    shift_mask = response_mask[:, 1:].contiguous()

    log_probs = F.log_softmax(shift_logits, dim=-1)

    per_token_logps = torch.gather(
        log_probs, 
        dim=-1, 
        index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    sequence_logps = (per_token_logps * shift_mask).sum(dim=-1)
    return sequence_logps

def run_training(config_path: str, run_name: str, dataset_path: str | None = None, output_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg = bundle["cfg"]
    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    optimizer = bundle["optimizer"]
    loader = bundle["loader"]
    beta_val = bundle["beta"]
    
    ref_model = load_policy(cfg, trainable=False)
    
    model.train()
    ref_model.eval()
    
    print(f"Starting DPO Training: {run_name} (Beta: {beta_val})")
    
    for step, (chosen_batch, rejected_batch) in enumerate(loader):
        chosen_batch = {k: v.to(model.device) for k, v in chosen_batch.items()}
        rejected_batch = {k: v.to(model.device) for k, v in rejected_batch.items()}
        
        with torch.no_grad():
            ref_chosen_logits = ref_model(
                input_ids=chosen_batch["input_ids"],
                attention_mask=chosen_batch["attention_mask"]
            ).logits
            ref_rejected_logits = ref_model(
                input_ids=rejected_batch["input_ids"],
                attention_mask=rejected_batch["attention_mask"]
            ).logits
            
            ref_chosen_logps = get_batch_logps(
                ref_chosen_logits, chosen_batch["input_ids"], chosen_batch["response_mask"]
            )
            ref_rejected_logps = get_batch_logps(
                ref_rejected_logits, rejected_batch["input_ids"], rejected_batch["response_mask"]
            )
            
        policy_chosen_logits = model(
            input_ids=chosen_batch["input_ids"],
            attention_mask=chosen_batch["attention_mask"]
        ).logits
        policy_rejected_logits = model(
            input_ids=rejected_batch["input_ids"],
            attention_mask=rejected_batch["attention_mask"]
        ).logits
        
        policy_chosen_logps = get_batch_logps(
            policy_chosen_logits, chosen_batch["input_ids"], chosen_batch["response_mask"]
        )
        policy_rejected_logps = get_batch_logps(
            policy_rejected_logits, rejected_batch["input_ids"], rejected_batch["response_mask"]
        )
        
        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logps,
            policy_rejected_logp=policy_rejected_logps,
            ref_chosen_logp=ref_chosen_logps,
            ref_rejected_logp=ref_rejected_logps,
            beta=beta_val
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(
                f"Step {step:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Pref Acc: {metrics['preference_accuracy'].item():.2f} | "
                f"Policy Margin: {metrics['policy_margin_mean'].item():.3f}"
            )

    print(f"Saving checkpoint to {output}")
    model.save_pretrained(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()
