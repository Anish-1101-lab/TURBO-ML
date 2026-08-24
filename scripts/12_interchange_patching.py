"""
Phase 4c: interchange patching -- a stronger complement to Phase 4/4b's
mean-ablation causal check (does NOT replace it; Phase 4/4b's code, data,
and PROGRESS.md sections are untouched).

Phase 4/4b tested causality by ablating the probe's layer-24 direction
toward its Phase-3 training-population MEAN -- a blunt intervention that
destroys information toward an average rather than substituting a
specific alternative. Interchange patching is a targeted version of the
same mechanic (see probes/interchange.py): splice a REAL donor example's
own projection along a direction into a recipient's forward pass at the
recipient's own verification position, and see whether p(x) shifts toward
the donor's outcome.

Design:
  1. Fresh generation pass, 60 prompts (20/domain, same sampling convention
     as Phase 4/4b), driven by the REAL exact accept/reject rule
     (probes/interchange.py's clean_verify_and_collect) -- reused, not
     replayed, from Phase 4/4b's saved data, because reproducing this
     experiment requires each candidate's full token context, which
     Phase 4/4b's saved JSON never persisted (only scalar metrics).
  2. Per prompt, after its generation completes: pick the single
     highest-label record (label >= 0.9, "near-certain accept") and the
     single lowest-label record (label <= 0.3, "near-certain reject or
     borderline") from THAT prompt's own trajectory as a matched HIGH/LOW
     pair -- same domain and same underlying prompt by construction, so
     the pair differs mainly in the local context/candidate token near
     each position, not in unrelated context. Prompts where no record
     clears one or both thresholds contribute no pair; the achieved n is
     reported plainly, not padded.
  3. Per pair, per direction (probe direction from probe_linear_layer24.pt,
     and the SAME fixed random direction as Phase 4/4b, seed=sd_config's
     seed) -- position-restricted patching only (Phase 4b's stronger
     design): patch HIGH's donor activation into LOW's own forward pass at
     LOW's position, and LOW's into HIGH's, each measuring the resulting
     shift in p(x) for that recipient's own candidate token.

Usage: python scripts/12_interchange_patching.py
"""
import argparse
import json
import os
import random

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.ablation import AblationHook, probe_direction_from_checkpoint, random_direction
from probes.interchange import clean_verify_and_collect, patched_p_at_position
from speculative.datasets import DOMAIN_LOADERS
from speculative.sd_loop import draft_tokens

TARGET_LAYER = 24
HIGH_THRESHOLD = 0.9
LOW_THRESHOLD = 0.3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    parser.add_argument("--probe_ckpt", default="analysis/phase3/probe_linear_layer24.pt")
    parser.add_argument("--out", default="analysis/phase4c")
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
    probe_dir = probe_direction_from_checkpoint(ckpt).to(device)
    rand_dir = random_direction(ckpt["mu"].shape[1], seed=scfg["seed"]).to(device)  # SAME seed as Phase 4/4b
    print(f"probe direction norm check: {probe_dir.norm().item():.4f}, "
          f"random direction norm check: {rand_dir.norm().item():.4f}")
    print(f"cosine(probe_dir, rand_dir) = {(probe_dir * rand_dir).sum().item():.4f} (should be near 0)")

    layer_module = target.model.layers[TARGET_LAYER - 1]
    probe_hook = AblationHook(layer_module, probe_dir, mean_proj=0.0, device=device)
    random_hook = AblationHook(layer_module, rand_dir, mean_proj=0.0, device=device)

    pairs = []
    n_prompts_seen = 0
    n_prompts_qualified = 0

    for domain in ["code", "reasoning", "chat"]:
        prompts = DOMAIN_LOADERS[domain]()
        rng = random.Random(scfg["seed"] + 1)  # same sampling convention as Phase 4/4b
        order = list(range(len(prompts)))
        rng.shuffle(order)
        chosen = order[:args.n_prompts_per_domain]

        for pi, idx in enumerate(chosen):
            n_prompts_seen += 1
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)

            generator = torch.Generator(device=device).manual_seed(scfg["seed"] * 7 + pi)
            n_generated = 0
            prompt_records = []
            while n_generated < args.max_new_tokens:
                draft_ids, q_scalars, q_full_dists, _ = draft_tokens(
                    drafter, context_ids, scfg["k"], vocab_size, scfg["temperature"], generator)
                records, new_context_ids = clean_verify_and_collect(
                    target, context_ids, draft_ids, q_scalars, q_full_dists,
                    vocab_size, scfg["temperature"], generator, TARGET_LAYER)
                prompt_records.extend(records)
                new_tail = new_context_ids[0, context_ids.shape[1]:]
                n_generated += new_tail.numel()
                context_ids = new_context_ids
                if (new_tail == eos_id).any():
                    break

            high = max(prompt_records, key=lambda r: r["label"])
            low = min(prompt_records, key=lambda r: r["label"])

            if high["label"] < HIGH_THRESHOLD or low["label"] > LOW_THRESHOLD:
                print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: NO QUALIFYING PAIR "
                      f"(max label={high['label']:.3f}, min label={low['label']:.3f})")
                continue

            pair_result = dict(domain=domain, idx=idx,
                                label_high=high["label"], label_low=low["label"],
                                p_clean_high=high["p_clean"], p_clean_low=low["p_clean"])

            for dir_name, direction, hook in [("probe", probe_dir, probe_hook), ("random", rand_dir, random_hook)]:
                p_h2l = patched_p_at_position(
                    target, hook, direction, high["h_clean"], low["full_ids"], low["position"],
                    low["token_id"], vocab_size, scfg["temperature"])
                p_l2h = patched_p_at_position(
                    target, hook, direction, low["h_clean"], high["full_ids"], high["position"],
                    high["token_id"], vocab_size, scfg["temperature"])
                pair_result[f"p_patched_high_into_low_{dir_name}"] = p_h2l
                pair_result[f"p_patched_low_into_high_{dir_name}"] = p_l2h
                # positive = shift in the causally-predicted direction, for BOTH swap directions,
                # via the sign flip on low-into-high (predicted to shift p DOWN, not up)
                pair_result[f"causal_shift_high_into_low_{dir_name}"] = p_h2l - low["p_clean"]
                pair_result[f"causal_shift_low_into_high_{dir_name}"] = high["p_clean"] - p_l2h

            pairs.append(pair_result)
            n_prompts_qualified += 1
            print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: pair OK "
                  f"(label_high={high['label']:.3f} label_low={low['label']:.3f}) "
                  f"n_pairs_so_far={len(pairs)}")

    print(f"\n{n_prompts_qualified}/{n_prompts_seen} prompts yielded a qualifying pair "
          f"(HIGH label >= {HIGH_THRESHOLD}, LOW label <= {LOW_THRESHOLD})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "interchange_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=2)
    print(f"Saved {len(pairs)} pairs to {args.out}/interchange_pairs.json")


if __name__ == "__main__":
    main()
