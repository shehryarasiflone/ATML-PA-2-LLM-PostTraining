from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

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


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the sum of response-token log-probabilities.
    Masks out prompt tokens (label == -100).
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    token_losses = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)), 
        shift_labels.view(-1)
    )
    token_losses = token_losses.view(shift_labels.size())
    
    mask = (shift_labels != -100)
    token_logps = -token_losses * mask
    return token_logps.sum(dim=-1)

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
            ref_chosen_logits = ref_model(**chosen_batch).logits
            ref_rejected_logits = ref_model(**rejected_batch).logits
            
            ref_chosen_logps = get_batch_logps(ref_chosen_logits, chosen_batch["labels"])
            ref_rejected_logps = get_batch_logps(ref_rejected_logits, rejected_batch["labels"])
            
        policy_chosen_logits = model(**chosen_batch).logits
        policy_rejected_logits = model(**rejected_batch).logits
        
        policy_chosen_logps = get_batch_logps(policy_chosen_logits, chosen_batch["labels"])
        policy_rejected_logps = get_batch_logps(policy_rejected_logits, rejected_batch["labels"])
        
        loss, metrics = dpo_loss(
            policy_chosen_logp=policy_chosen_logps,
            policy_rejected_logp=policy_rejected_logps,
            ref_chosen_logp=ref_chosen_logps,
            ref_rejected_logp=ref_rejected_logps,
            beta=beta_val
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 10 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Pref Acc: {metrics['preference_accuracy']:.2f}")

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
