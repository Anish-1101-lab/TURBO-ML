"""
Phase 4 follow-up: same causal check as scripts/05_causal_check.py, but the
ablation hook is restricted to only the specific verification position for
each drafted token, instead of ablating every position in the sequence.

Identical to scripts/05_causal_check.py in every other respect -- same
layer, same probe checkpoint, same random-direction control, same prompt
sampling (same seeds), same generator-consumption order in the verify step
-- so the clean-pass generation trajectory (and therefore the resulting
example set: same prompts, same drafted tokens, same p_clean, same
probe_score) is bit-for-bit identical to the original run. Only how the
two ablated conditions are computed differs. This makes the two results
directly comparable, isolating the effect of the position-restriction
change itself.

Usage: python scripts/07_causal_check_positional.py
"""
import argparse
import json
import os
import random

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.ablation import AblationHook, probe_direction_from_checkpoint, random_direction
from probes.causal_verify import causal_verify_and_step_positional
from speculative.datasets import DOMAIN_LOADERS
from speculative.sd_loop import draft_tokens

TARGET_LAYER = 24  # Phase 3 AUROC peak; hidden_states[24] = model.model.layers[23] output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    parser.add_argument("--probe_ckpt", default="analysis/phase3/probe_linear_layer24.pt")
    parser.add_argument("--out", default="analysis/phase4_positional")
    parser.add_argument("--n_prompts_per_domain", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    scfg = yaml.safe_load(open(args.sd))
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

    ckpt = torch.load(args.probe_ckpt, weights_only=False)
    probe_dir = probe_direction_from_checkpoint(ckpt)
    mu = ckpt["mu"].squeeze(0)
    probe_mean_proj = (mu * probe_dir).sum().item()

    rand_dir = random_direction(mu.shape[0], seed=scfg["seed"])
    rand_mean_proj = (mu * rand_dir).sum().item()

    layer_module = target.model.layers[TARGET_LAYER - 1]
    probe_hook = AblationHook(layer_module, probe_dir, probe_mean_proj, device)
    random_hook = AblationHook(layer_module, rand_dir, rand_mean_proj, device)
    print(f"probe direction norm check: {probe_dir.norm().item():.4f}, "
          f"random direction norm check: {rand_dir.norm().item():.4f}")
    print(f"cosine(probe_dir, rand_dir) = {(probe_dir * rand_dir).sum().item():.4f} (should be near 0)")

    all_records = []
    for domain in ["code", "reasoning", "chat"]:
        prompts = DOMAIN_LOADERS[domain]()
        rng = random.Random(scfg["seed"] + 1)  # SAME seed offset as scripts/05 -- reproduces the identical prompt sample
        order = list(range(len(prompts)))
        rng.shuffle(order)
        chosen = order[:args.n_prompts_per_domain]

        for pi, idx in enumerate(chosen):
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)

            generator = torch.Generator(device=device).manual_seed(scfg["seed"] * 7 + pi)  # SAME formula as scripts/05
            n_generated = 0
            while n_generated < args.max_new_tokens:
                draft_ids, q_scalars, q_full_dists, _ = draft_tokens(
                    drafter, context_ids, scfg["k"], vocab_size, scfg["temperature"], generator)
                records, new_context_ids = causal_verify_and_step_positional(
                    target, context_ids, draft_ids, q_scalars, q_full_dists,
                    vocab_size, scfg["temperature"], generator, probe_hook, random_hook,
                    ckpt, TARGET_LAYER)
                for r in records:
                    r["domain"] = domain
                all_records.extend(records)
                new_tail = new_context_ids[0, context_ids.shape[1]:]
                n_generated += new_tail.numel()
                context_ids = new_context_ids
                if (new_tail == eos_id).any():
                    break
            print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: {len(all_records)} total records so far")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "causal_records.json"), "w") as f:
        json.dump(all_records, f, indent=2)
    print(f"Saved {len(all_records)} records to {args.out}/causal_records.json")


if __name__ == "__main__":
    main()
