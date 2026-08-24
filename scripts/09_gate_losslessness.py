"""
Phase 5 step 2: losslessness validation. Attempts to DISPROVE losslessness,
not confirm it -- reports any measurable divergence honestly.

For each of a held-out sample of prompts (same style/scale as Phase 4/4b: 20
per domain x {code, reasoning, chat} = 60 prompts, fresh sample, capped at
128 new tokens), runs TWO independent generations with the SAME seed:

  1. Ungated baseline: speculative.sd_loop.run_speculative_decoding (the
     exact accept/reject rule, unchanged since Phase 1/2).
  2. Gated shadow: speculative.gated_sd_loop.run_gated_speculative_decoding_shadow
     (full compute paid throughout -- no speed benefit here, this is a
     correctness check, not a timing run -- but the ACTUAL generated
     trajectory is driven by the gate's decisions, exactly as
     gated_verify_and_step_fast would produce).

Two things get measured and reported, not just one:
  - GLOBAL: do the two runs' final generated token sequences match
    exactly? If not, at what position do they first diverge?
  - LOCAL: among positions where the gate actually fired, what fraction
    would the real exact accept/reject rule have rejected (mismatch=True
    in the shadow records)? This is the direct, live-generation analogue
    of the calibration-bin extrapolation the threshold was picked from in
    step 1 -- it either confirms or contradicts that extrapolation.

Usage: python scripts/09_gate_losslessness.py
"""
import argparse
import json
import os
import random

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.gate import load_mlp_gate_probe
from speculative.datasets import DOMAIN_LOADERS
from speculative.gated_sd_loop import run_gated_speculative_decoding_shadow
from speculative.sd_loop import run_speculative_decoding


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    parser.add_argument("--gate", default="configs/gate_config.yaml")
    parser.add_argument("--out", default="analysis/phase5")
    parser.add_argument("--out_name", default="losslessness.json",
                         help="output filename within --out, so repeat runs at a different "
                              "threshold don't overwrite prior results")
    parser.add_argument("--threshold", type=float, default=None,
                         help="override configs/gate_config.yaml's threshold for this run only")
    parser.add_argument("--n_prompts_per_domain", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    scfg = yaml.safe_load(open(args.sd))
    gcfg = yaml.safe_load(open(args.gate))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mcfg["device"]["cuda_visible_devices"])
    device = "cuda:0"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    tok = AutoTokenizer.from_pretrained(mcfg["target"]["name"])
    target = AutoModelForCausalLM.from_pretrained(
        mcfg["target"]["name"], dtype=dtype_map[mcfg["target"]["dtype"]]).to(device).eval()
    drafter = AutoModelForCausalLM.from_pretrained(
        mcfg["drafter"]["name"], dtype=dtype_map[mcfg["drafter"]["dtype"]]).to(device).eval()
    vocab_size = tok.vocab_size
    eos_id = tok.eos_token_id

    gate_model, gate_mu, gate_sigma = load_mlp_gate_probe(
        gcfg["probe_checkpoint"], hidden_dim=target.config.hidden_size,
        mlp_hidden_dim=gcfg["mlp"]["hidden_dim"], dropout=gcfg["mlp"]["dropout"], device=device)
    threshold = args.threshold if args.threshold is not None else gcfg["threshold"]
    probe_layer = gcfg["probe_layer"]
    print(f"gate: layer={probe_layer} threshold={threshold}")

    per_prompt = []
    total_gated = 0
    total_gate_mismatches = 0
    total_positions = 0
    exact_matches = 0
    n_prompts_done = 0

    for domain in ["code", "reasoning", "chat"]:
        prompts = DOMAIN_LOADERS[domain]()
        rng = random.Random(scfg["seed"] + 1)  # same sampling convention as scripts/05, 07
        order = list(range(len(prompts)))
        rng.shuffle(order)
        chosen = order[:args.n_prompts_per_domain]

        for pi, idx in enumerate(chosen):
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)
            seed = scfg["seed"] * 7 + pi  # same formula as scripts/05, 07 -- not required to match
                                            # those runs, just needs to be identical between the two
                                            # calls below for a given prompt

            ungated_ids, _ = run_speculative_decoding(
                target, drafter, context_ids, vocab_size, scfg["k"], args.max_new_tokens,
                scfg["temperature"], layer_indices=[], seed=seed, eos_token_id=eos_id)

            gated_ids, gated_records = run_gated_speculative_decoding_shadow(
                target, drafter, context_ids, vocab_size, scfg["k"], args.max_new_tokens,
                scfg["temperature"], seed, gate_model, gate_mu, gate_sigma, threshold,
                probe_layer, eos_token_id=eos_id)

            ungated_new = ungated_ids[0, context_ids.shape[1]:].tolist()
            gated_new = gated_ids[0, context_ids.shape[1]:].tolist()

            min_len = min(len(ungated_new), len(gated_new))
            first_divergence = None
            for j in range(min_len):
                if ungated_new[j] != gated_new[j]:
                    first_divergence = j
                    break
            exact_match = (first_divergence is None) and (len(ungated_new) == len(gated_new))
            exact_matches += int(exact_match)

            n_gated = sum(1 for r in gated_records if r["gated"])
            n_mismatch = sum(1 for r in gated_records if r["gated"] and r["mismatch"])
            total_gated += n_gated
            total_gate_mismatches += n_mismatch
            total_positions += len(gated_records)

            per_prompt.append(dict(
                domain=domain, idx=idx, seed=seed,
                exact_match=exact_match, first_divergence=first_divergence,
                ungated_len=len(ungated_new), gated_len=len(gated_new),
                n_positions=len(gated_records), n_gated=n_gated, n_gate_mismatch=n_mismatch,
            ))
            n_prompts_done += 1
            print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: exact_match={exact_match} "
                  f"first_divergence={first_divergence} n_gated={n_gated}/{len(gated_records)} "
                  f"mismatches={n_mismatch}")

    summary = dict(
        n_prompts=n_prompts_done,
        exact_match_rate=exact_matches / n_prompts_done,
        total_positions=total_positions,
        total_gated=total_gated,
        gate_fire_rate=total_gated / total_positions if total_positions else 0.0,
        total_gate_mismatches=total_gate_mismatches,
        local_mismatch_rate_among_gated=(total_gate_mismatches / total_gated) if total_gated else 0.0,
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, args.out_name), "w") as f:
        json.dump(dict(summary=summary, per_prompt=per_prompt), f, indent=2)
    print(f"\nSaved to {args.out}/{args.out_name}")


if __name__ == "__main__":
    main()
