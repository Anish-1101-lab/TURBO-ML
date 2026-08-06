"""
Phase 1 validation: run the instrumented SD loop on a handful of short
prompts, then independently recompute p(x), q(x) for every drafted token
via a *separate*, deliberately unoptimized code path (fresh forward passes,
raw tensor indexing, no dataclasses / no reuse of speculative.sd_loop
internals) and diff against what the loop logged.

This is a correctness check, not a benchmark: it exists to catch bugs like
off-by-one position indexing, wrong vocab-dimension slicing, or a missed
temperature scaling before we trust the logged labels enough to scale up.

Usage: python scripts/01_validate_sd.py --models configs/models.yaml --sd configs/sd_config.yaml
"""
import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative.sd_loop import (draft_tokens, layer_indices_from_stride,
                                  run_speculative_decoding, verify_and_step)

VALIDATION_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
    "Q: What is 12 + 7?\nA:",
]

TOL_REL = 1e-3


def independent_prob(model, ids: torch.Tensor, token_id: int, vocab_size: int, temperature: float) -> float:
    """Fresh forward pass, no cache/state sharing with the SD loop, softmax
    over the real vocab only, probability of one specific token at the last position."""
    with torch.no_grad():
        out = model(input_ids=ids)
    logits = out.logits[0, -1, :vocab_size].float()
    probs = torch.softmax(logits / temperature, dim=-1)
    return probs[token_id].item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    scfg = yaml.safe_load(open(args.sd))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mcfg["device"]["cuda_visible_devices"])
    device = "cuda:0"

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    tok = AutoTokenizer.from_pretrained(mcfg["target"]["name"])
    target = AutoModelForCausalLM.from_pretrained(
        mcfg["target"]["name"], torch_dtype=dtype_map[mcfg["target"]["dtype"]]).to(device).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        mcfg["drafter"]["name"], torch_dtype=dtype_map[mcfg["drafter"]["dtype"]]).to(device).eval()

    vocab_size = tok.vocab_size
    layer_indices = layer_indices_from_stride(target.config.num_hidden_layers, scfg["layer_stride"])
    k = scfg["k"]
    temperature = scfg["temperature"]

    print(f"layer_indices = {layer_indices}\n")

    n_checked = 0
    n_mismatch = 0

    for prompt in VALIDATION_PROMPTS:
        print(f"=== prompt: {prompt!r} ===")
        context_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        generator = torch.Generator(device=device).manual_seed(scfg["seed"])
        draft_ids, q_scalars, q_full_dists, entropies = draft_tokens(
            drafter, context_ids, k, vocab_size, temperature, generator)
        result = verify_and_step(
            target, context_ids, draft_ids, q_scalars, q_full_dists, entropies,
            vocab_size, temperature, layer_indices, generator)

        full_ids = torch.cat([context_ids, draft_ids], dim=1)

        print(f"{'tok':>12} {'q(logged)':>10} {'q(indep)':>10} {'p(logged)':>10} {'p(indep)':>10} {'label':>8} {'accept':>7}")
        for i, rec in enumerate(result.records):
            tok_str = tok.decode([rec.token_id])

            # independent q: fresh drafter forward over context + first i drafted tokens
            prefix = torch.cat([context_ids, draft_ids[:, :i]], dim=1)
            q_indep = independent_prob(drafter, prefix, rec.token_id, vocab_size, temperature)

            # independent p: fresh target forward over context + all k drafted tokens,
            # reading out the logit at the same conditioning position as the loop used
            with torch.no_grad():
                out = target(input_ids=full_ids)
            p_logits = out.logits[0, rec.position, :vocab_size].float()
            p_probs = torch.softmax(p_logits / temperature, dim=-1)
            p_indep = p_probs[rec.token_id].item()

            q_match = abs(q_indep - rec.q) <= TOL_REL * max(abs(rec.q), 1e-8)
            p_match = abs(p_indep - rec.p) <= TOL_REL * max(abs(rec.p), 1e-8)
            label_check = min(1.0, rec.p / rec.q)
            label_match = abs(label_check - rec.label) <= 1e-9

            n_checked += 1
            if not (q_match and p_match and label_match):
                n_mismatch += 1

            flag = "" if (q_match and p_match and label_match) else "  <-- MISMATCH"
            print(f"{tok_str!r:>12} {rec.q:>10.4f} {q_indep:>10.4f} {rec.p:>10.4f} {p_indep:>10.4f} "
                  f"{rec.label:>8.4f} {str(rec.accepted):>7}{flag}")
        print()

    print(f"Checked {n_checked} drafted-token records, {n_mismatch} mismatches.")
    assert n_mismatch == 0, "Validation failed -- see MISMATCH rows above"
    print("Phase 1 validation PASSED.\n")

    # Also run a full short generation end-to-end as a smoke test.
    prompt = VALIDATION_PROMPTS[0]
    context_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    final_ids, records = run_speculative_decoding(
        target, drafter, context_ids, vocab_size, k=k,
        max_new_tokens=scfg["max_new_tokens"], temperature=temperature,
        layer_indices=layer_indices, seed=scfg["seed"])
    text = tok.decode(final_ids[0], skip_special_tokens=True)
    accept_rate = sum(r.accepted for r in records) / len(records)
    print(f"Smoke-test generation for {prompt!r}:")
    print(f"  output: {text!r}")
    print(f"  drafted-token records: {len(records)}, acceptance rate: {accept_rate:.2%}")


if __name__ == "__main__":
    main()
