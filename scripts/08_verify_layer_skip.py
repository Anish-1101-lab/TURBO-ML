"""
Phase 5 pre-check: does speculative/layer_skip.py's forward_prefix +
forward_suffix composition reproduce a REAL full-depth forward pass exactly?

Three things get checked, on a handful of real prompts:
  1. forward_prefix(target, ids, 24) matches output_hidden_states=True's
     hidden_states[24] exactly (max abs diff).
  2. forward_suffix(target, hs24, 28, 24) on the FULL sequence matches the
     real model's logits exactly (max abs diff + argmax agreement),
     confirming the prefix/suffix split reconstructs a normal forward pass.
  3. forward_suffix on a SHORTER length-restricted slice of hs24 (only up
     through some earlier position) still reproduces the real logits AT
     THAT position -- this is the causal-correctness check for the
     "restrict suffix length to just what's needed" trick used in
     gated_verify_and_step_fast.

This does NOT touch the SD loop, the gate, or generation. Pure mechanism
correctness, analogous to Phase 4's no-op-hook check before trusting the
ablation math.

Usage: python scripts/08_verify_layer_skip.py
"""
import argparse

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative.layer_skip import forward_prefix, forward_suffix

PROBE_LAYER = 24
N_TOTAL_LAYERS = 28

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Natasha has 3 times as many marbles as Tom. If Tom has 7 marbles, how many does Natasha have?",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mcfg["device"]["cuda_visible_devices"])
    device = "cuda:0"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tok = AutoTokenizer.from_pretrained(mcfg["target"]["name"])
    target = AutoModelForCausalLM.from_pretrained(
        mcfg["target"]["name"], dtype=dtype_map[mcfg["target"]["dtype"]]).to(device).eval()
    vocab_size = tok.vocab_size

    all_pass = True

    for prompt in PROMPTS:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        seq_len = ids.shape[1]

        with torch.no_grad():
            real_out = target(input_ids=ids, output_hidden_states=True)
        real_hs24 = real_out.hidden_states[PROBE_LAYER]
        real_logits = real_out.logits

        # Check 1: prefix matches real hidden_states[24]
        hs24 = forward_prefix(target, ids, PROBE_LAYER)
        prefix_diff = (hs24 - real_hs24).abs().max().item()

        # Check 2: prefix + full-length suffix matches real logits
        recon_logits = forward_suffix(target, hs24, N_TOTAL_LAYERS, PROBE_LAYER)
        logits_diff = (recon_logits[..., :vocab_size] - real_logits[..., :vocab_size]).abs().max().item()
        argmax_match = torch.equal(
            recon_logits[0, -1, :vocab_size].argmax(), real_logits[0, -1, :vocab_size].argmax())

        # Check 3: length-restricted suffix reproduces the real logit at an earlier position
        mid_pos = max(0, seq_len - 2)  # second-to-last position, arbitrary "earlier" point
        restricted_logits = forward_suffix(target, hs24[:, :mid_pos + 1, :], N_TOTAL_LAYERS, PROBE_LAYER)
        restricted_diff = (restricted_logits[0, -1, :vocab_size] - real_logits[0, mid_pos, :vocab_size]).abs().max().item()

        ok = prefix_diff < 1e-2 and logits_diff < 1e-1 and argmax_match and restricted_diff < 1e-1
        all_pass = all_pass and ok
        print(f"prompt={prompt[:40]!r:42} prefix_diff={prefix_diff:.5f} "
              f"logits_diff={logits_diff:.5f} argmax_match={argmax_match} "
              f"restricted_diff={restricted_diff:.5f} -> {'PASS' if ok else 'FAIL'}")

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")


if __name__ == "__main__":
    main()
