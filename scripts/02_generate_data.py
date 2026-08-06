"""
Phase 2: generate the small-scale (~100k drafted-token) probe dataset across
three domains, using the validated instrumented SD loop from Phase 1.

For each domain: shuffle its prompt list, wrap each prompt in the target's
chat template, run the SD loop capped at max_new_tokens_per_prompt, log every
drafted-token record to a ShardWriter. Cycles back through the prompt list
(reshuffled, new seeds) if a domain's budget isn't met in one pass -- this
matters most for chat (only 80 unique MT-Bench prompts).

Usage: python scripts/02_generate_data.py --out data/phase2_small
"""
import argparse
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative.datasets import DOMAIN_LOADERS
from speculative.sd_loop import layer_indices_from_stride, run_speculative_decoding
from speculative.storage import ShardWriter, write_manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    parser.add_argument("--datagen", default="configs/datagen_config.yaml")
    parser.add_argument("--out", default="data/phase2_small")
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    scfg = yaml.safe_load(open(args.sd))
    dcfg = yaml.safe_load(open(args.datagen))

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
    eos_id = tok.eos_token_id
    print(f"layer_indices={layer_indices}, eos_token_id={eos_id}")

    stats = {}
    t_start = time.time()

    for domain, budget in dcfg["domains"].items():
        print(f"\n=== domain: {domain}, budget: {budget} ===")
        prompts = DOMAIN_LOADERS[domain]()
        import random
        rng = random.Random(scfg["seed"])
        order = list(range(len(prompts)))
        rng.shuffle(order)

        writer = ShardWriter(os.path.join(args.out, domain), domain, layer_indices,
                              shard_size=dcfg["shard_size"])
        n_records = 0
        n_accept = 0
        cursor = 0
        pass_idx = 0
        prompt_counter = 0
        t_domain = time.time()

        while n_records < budget and pass_idx < dcfg["max_passes"]:
            if cursor >= len(order):
                cursor = 0
                pass_idx += 1
                rng.shuffle(order)

            idx = order[cursor]
            cursor += 1
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            input_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)

            seed = scfg["seed"] * 1_000_003 + prompt_counter
            _, records = run_speculative_decoding(
                target, drafter, input_ids, vocab_size, k=scfg["k"],
                max_new_tokens=dcfg["max_new_tokens_per_prompt"], temperature=scfg["temperature"],
                layer_indices=layer_indices, seed=seed, eos_token_id=eos_id)

            prompt_id = f"{domain}_{idx}_pass{pass_idx}"
            for r in records:
                writer.add(dict(
                    hidden_states=r.hidden_states, label=r.label, p=r.p, q=r.q,
                    token_id=r.token_id, accepted=r.accepted,
                    drafter_entropy=r.drafter_entropy, position=r.position,
                    prompt_id=prompt_id,
                ))
                n_records += 1
                n_accept += int(r.accepted)

            prompt_counter += 1
            if prompt_counter % 10 == 0:
                elapsed = time.time() - t_domain
                print(f"  [{domain}] prompts={prompt_counter} pass={pass_idx} "
                      f"records={n_records}/{budget} accept_rate={n_accept/max(n_records,1):.3f} "
                      f"elapsed={elapsed:.0f}s")

        writer.close()
        stats[domain] = dict(
            n_records=n_records, accept_rate=n_accept / max(n_records, 1),
            n_prompts_used=prompt_counter, passes_over_dataset=pass_idx + 1,
            n_shards=writer.shard_idx, wall_clock_sec=time.time() - t_domain,
        )
        print(f"=== domain {domain} done: {stats[domain]} ===")

    manifest = dict(
        layer_indices=layer_indices, k=scfg["k"], temperature=scfg["temperature"],
        seed=scfg["seed"], eos_token_id=eos_id,
        target=mcfg["target"]["name"], drafter=mcfg["drafter"]["name"],
        stats=stats, total_wall_clock_sec=time.time() - t_start,
    )
    write_manifest(args.out, manifest)
    print(f"\nManifest written to {os.path.join(args.out, 'manifest.json')}")
    print(f"Total wall clock: {manifest['total_wall_clock_sec']:.0f}s")


if __name__ == "__main__":
    main()
