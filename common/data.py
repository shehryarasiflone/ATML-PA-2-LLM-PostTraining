from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    path = repo_path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    if cfg.get("base_config"):
        base_path = repo_path(cfg["base_config"])
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with repo_path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(repo_path(path))


def last_assistant_text(messages) -> str:
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for m in reversed(messages):
            if isinstance(m, dict) and m.get("role") == "assistant":
                return str(m.get("content", ""))
    return str(messages)


def prompt_messages_from_preference(row: dict) -> list[dict]:
    chosen = row.get("chosen")
    if isinstance(chosen, list) and chosen:
        out = list(chosen)
        if isinstance(out[-1], dict) and out[-1].get("role") == "assistant":
            out = out[:-1]
        return out
    prompt = str(row.get("prompt", ""))
    return [{"role": "user", "content": prompt}]


def preference_responses(row: dict) -> tuple[str, str]:
    return last_assistant_text(row["chosen"]), last_assistant_text(row["rejected"])


def prompt_messages(row: dict) -> list[dict]:
    if isinstance(row.get("messages"), list):
        return row["messages"]
    if isinstance(row.get("prompt"), list):
        return row["prompt"]
    return [{"role": "user", "content": str(row.get("prompt", row.get("question", "")))}]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def encode_prompt_response(
    tokenizer,
    messages: list[dict],
    response: str,
    max_length: int,
):
    """
    Encode prompt + response for DPO.

    The prompt is kept intact for both chosen and rejected responses.
    If the combined sequence exceeds max_length, truncate the RESPONSE
    from the right rather than removing prompt/question tokens.
    """

    prompt_res = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )
    if isinstance(prompt_res, dict) or hasattr(prompt_res, "input_ids"):
        prompt_ids = list(prompt_res["input_ids"])
    elif hasattr(prompt_res, "tolist"):
        prompt_ids = prompt_res.tolist()
    else:
        prompt_ids = list(prompt_res)

    # A DPO example is meaningful only if we can condition on the prompt.
    if len(prompt_ids) >= max_length:
        raise ValueError(
            f"Prompt alone has {len(prompt_ids)} tokens, which does not fit "
            f"inside max_length={max_length}. "
            "Increase max_length or filter this example."
        )

    response_ids = tokenizer(
        response,
        add_special_tokens=False,
    )["input_ids"]

    response_budget = max_length - len(prompt_ids)

    # Keep an EOS token when the tokenizer defines one.
    eos_id = tokenizer.eos_token_id

    if eos_id is not None:
        # Reserve one token for EOS.
        content_budget = max(0, response_budget - 1)
        response_ids = response_ids[:content_budget] + [eos_id]
    else:
        response_ids = response_ids[:response_budget]

    ids = prompt_ids + response_ids

    response_mask = (
        [0] * len(prompt_ids)
        + [1] * len(response_ids)
    )

    assert len(ids) <= max_length
    assert len(ids) == len(response_mask)

    return ids, response_mask


def pad_batch(tokenizer, examples: list[tuple[list[int], list[int]]]):
    import torch

    max_len = max(len(ids) for ids, _ in examples)
    pad_id = tokenizer.pad_token_id
    input_ids, attention_mask, response_mask = [], [], []

    for ids, rmask in examples:
        pad = max_len - len(ids)
        input_ids.append([pad_id] * pad + ids)
        attention_mask.append([0] * pad + [1] * len(ids))
        response_mask.append([0] * pad + rmask)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32),
    }
