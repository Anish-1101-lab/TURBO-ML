"""
Domain prompt loaders for Phase 2 data generation. Each loader returns a
plain list[str] of user-turn prompt strings; the caller wraps these in the
target's chat template uniformly across domains (see scripts/02_generate_data.py).

Domain -> dataset choices (see PROGRESS.md for why):
  code:      openai/openai_humaneval, split=test  (164 examples)
  reasoning: openai/gsm8k "main", split=test        (1319 examples)
  chat:      HuggingFaceH4/mt_bench_prompts, split=train (80 examples,
             first turn only -- see PROGRESS.md flag on multi-turn simplification)
"""
import ast

from datasets import load_dataset


def load_code_prompts() -> list:
    ds = load_dataset("openai/openai_humaneval", split="test")
    return [
        f"Complete the following function:\n\n```python\n{ex['prompt']}\n```"
        for ex in ds
    ]


def load_reasoning_prompts() -> list:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [ex["question"] for ex in ds]


def load_chat_prompts() -> list:
    ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train")
    prompts = []
    for ex in ds:
        raw = ex["prompt"]
        turns = ast.literal_eval(raw) if isinstance(raw, str) else raw
        prompts.append(turns[0])
    return prompts


DOMAIN_LOADERS = {
    "code": load_code_prompts,
    "reasoning": load_reasoning_prompts,
    "chat": load_chat_prompts,
}


# --- Tier 0 additions (additive only -- DOMAIN_LOADERS/load_code_prompts above
# are untouched so existing Phase 2+ scripts keep working unmodified). Tier 0's
# code domain is HumanEval + MBPP combined, per the spec. ---

def load_mbpp_prompts() -> list:
    ds = load_dataset("google-research-datasets/mbpp", "full", split="test")
    prompts = []
    for ex in ds:
        tests = "\n".join(ex["test_list"])
        prompts.append(
            f"Write a Python function to solve the following problem:\n\n{ex['text']}\n\n"
            f"Your function should satisfy these tests:\n```python\n{tests}\n```"
        )
    return prompts


def load_code_prompts_tier0() -> list:
    return load_code_prompts() + load_mbpp_prompts()


TIER0_DOMAIN_LOADERS = {
    "code": load_code_prompts_tier0,
    "reasoning": load_reasoning_prompts,
    "chat": load_chat_prompts,
}
