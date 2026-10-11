from __future__ import annotations

import hashlib
import itertools
import json
import logging
import re
import warnings
from pathlib import Path

# Silence 8-bit quantized matrix multiplication warnings
warnings.filterwarnings("ignore", message=".*MatMul8bitLt.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bitsandbytes")
logging.getLogger("bitsandbytes").setLevel(logging.ERROR)

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from common.data import repo_path
from common.models import clear_gpu, resolve_dtype

PAIRWISE_RUBRIC = """You are comparing two candidate solutions to the same math problem.

Evaluate mathematical correctness, internal consistency, relevance, and whether the stated final answer follows from the reasoning.
Do not prefer a response merely because it is longer, more confident, or more polished.
If neither response is meaningfully better, return TIE.

Return exactly one token: A, B, or TIE.

Problem:
{problem}

Candidate A:
{a}

Candidate B:
{b}

Preference:"""


class PairwiseAIJudge:
    """Fixed course judge with deterministic A/B orientation balancing and persistent caching."""

    def __init__(self, cfg: dict, cache_path: str | Path):
        self.cfg = cfg
        self.cache_path = repo_path(cache_path)
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["ai_judge_model"], padding_side="left", use_fast=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs = {"low_cpu_mem_usage": True}
        if torch.cuda.is_available() and bool(cfg.get("quantize_frozen_models", True)):
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=resolve_dtype(cfg.get("dtype", "float16")),
            )
            kwargs["device_map"] = "auto"
        else:
            kwargs["dtype"] = resolve_dtype(cfg.get("dtype", "float16"))

        self.model = AutoModelForCausalLM.from_pretrained(cfg["ai_judge_model"], **kwargs)
        self.model.eval()

    def _key(self, problem, a, b):
        payload = json.dumps(
            {"model": self.cfg["ai_judge_model"], "problem": problem, "a": a, "b": b},
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    @torch.no_grad()
    def compare(self, problem: str, a: str, b: str):
        key = self._key(problem, a, b)
        if key in self.cache:
            return self.cache[key]

        swap = int(key[:8], 16) % 2 == 1
        aa, bb = (b, a) if swap else (a, b)
        text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)

        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            return_tensors="pt",
            add_generation_prompt=True,
        )

        device = next(self.model.parameters()).device
        if isinstance(enc, torch.Tensor):
            input_ids = enc.to(device)
            attention_mask = None
        else:
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=4,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        prompt_len = input_ids.shape[1]
        decoded = self.tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True).strip().upper()
        m = re.search(r"\b(A|B|TIE)\b", decoded)
        result = m.group(1) if m else "TIE"
        if swap:
            result = {"A": "B", "B": "A", "TIE": "TIE"}[result]

        self.cache[key] = result
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        return result

    def group_rewards(self, problem: str, responses: list[str]):
        k = len(responses)
        wins = np.zeros(k, dtype=float)
        ties = np.zeros(k, dtype=float)
        for i, j in itertools.combinations(range(k), 2):
            pref = self.compare(problem, responses[i], responses[j])
            if pref == "A":
                wins[i] += 1
            elif pref == "B":
                wins[j] += 1
            else:
                ties[i] += 1
                ties[j] += 1
        return ((wins + 0.5 * ties) / max(k - 1, 1)).tolist()

    def close(self):
        clear_gpu(self.model)