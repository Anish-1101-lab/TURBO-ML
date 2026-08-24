"""
Tier 0 Pass 1 + Pass 2 data generation. Config-driven (configs/tier0_config.yaml)
-- no hardcoded paths or model names here.

Smoke-test usage (small scale, sanity-check the pipeline before a full run):
    python scripts/15_tier0_generate.py --config configs/tier0_config.yaml \
        --domains code reasoning --temperatures 0.0 0.7 --drafter A \
        --n-prompts 4 --max-new-tokens 48 --out data/tier0_smoke

Full-scale usage (no --n-prompts cap, all domains/temperatures/drafters) is
a separate, much longer-running invocation -- not run by this script's
default args, left for a follow-up decision after the smoke test.
"""
import argparse
import os
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative.datasets import TIER0_DOMAIN_LOADERS
from speculative.tier0_features import build_unigram_log_freq
from speculative.tier0_loop import run_tier0_sd
from speculative.tier0_storage import records_to_pass1_rows, save_pass1, save_pass2_per_layer

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier0_config.yaml")
    ap.add_argument("--domains", nargs="+", default=None, help="default: all domains in config")
    ap.add_argument("--temperatures", nargs="+", type=float, default=None, help="default: all temps in config")
    ap.add_argument("--drafter", default="A", choices=["A", "B", "C"])
    ap.add_argument("--n-prompts", type=int, default=None, help="cap prompts/domain (smoke test); default: unbounded (uses target_volume_per_domain)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override config's max_new_tokens_per_prompt")
    ap.add_argument("--out", default=None, help="override; default splits into <cfg pass1_dir>/<cfg pass2_dir>")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    domains = args.domains or list(cfg["domains"].keys())
    temperatures = args.temperatures if args.temperatures is not None else cfg["temperatures"]
    max_new_tokens = args.max_new_tokens or cfg["max_new_tokens_per_prompt"]
    probed_layers = cfg["probed_layers"]
    gamma = cfg["gamma"]
    seed = cfg["seed"]

    if args.out:
        pass1_dir = os.path.join(args.out, "pass1")
        pass2_dir = os.path.join(args.out, "pass2")
    else:
        pass1_dir = cfg["paths"]["pass1_dir"]
        pass2_dir = cfg["paths"]["pass2_dir"]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["cuda_visible_devices"])
    device = "cuda:0"

    tok = AutoTokenizer.from_pretrained(cfg["target"]["name"])
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"]["name"], torch_dtype=DTYPE_MAP[cfg["target"]["dtype"]]).to(device).eval()

    dspec = cfg["drafters"][args.drafter]
    drafter = AutoModelForCausalLM.from_pretrained(
        dspec["name"], torch_dtype=DTYPE_MAP[dspec["dtype"]]).to(device).eval()

    vocab_size = tok.vocab_size
    eos_id = tok.eos_token_id
    print(f"target={cfg['target']['name']} drafter={args.drafter}={dspec['name']}")
    print(f"probed_layers={probed_layers} gamma={gamma} vocab_size={vocab_size} eos_token_id={eos_id}")

    freq_cache = cfg["unigram_freq"]["cache_path"]
    if os.path.exists(freq_cache):
        unigram_log_freq = torch.load(freq_cache)
        print(f"loaded cached unigram_log_freq from {freq_cache}")
    else:
        print("building unigram_log_freq table from WikiText-103 sample (one-time)...")
        unigram_log_freq = build_unigram_log_freq(
            tok, vocab_size, cfg["unigram_freq"]["n_samples"], seed)
        os.makedirs(os.path.dirname(freq_cache), exist_ok=True)
        torch.save(unigram_log_freq, freq_cache)
        print(f"cached to {freq_cache}")

    manifest = dict(config=args.config, domains=domains, temperatures=temperatures,
                     drafter=args.drafter, n_prompts_cap=args.n_prompts, stats={})
    t_start = time.time()

    # Budget-based generation with pass-cycling (mirrors scripts/02_generate_data.py's
    # Phase 2 pattern), EXCEPT: at T=0 (greedy) we do NOT cycle passes. Greedy decoding
    # is a deterministic function of (prompt, seed-independent argmax choices) -- a
    # second pass over the same prompt produces byte-identical events, so cycling would
    # just write duplicate rows, not new information. If a domain doesn't have enough
    # unique prompts to reach target_volume_per_domain at T=0 in one pass (chat's 80
    # MT-Bench prompts, in particular), we stop after one pass and report the actual
    # achieved volume honestly rather than pad it with duplicates. At T>0 (stochastic),
    # cycling with a fresh seed per pass gives genuinely different samples, so cycling
    # is legitimate there and used to reach target_volume_per_domain.
    target_volume = cfg["target_volume_per_domain"]
    max_passes = cfg["max_passes"]

    for domain in domains:
        prompts = TIER0_DOMAIN_LOADERS[domain]()
        import random

        for temperature in temperatures:
            rng = random.Random(seed)
            order = list(range(len(prompts)))
            rng.shuffle(order)

            n_records = 0
            n_accept = 0
            cursor = 0
            pass_idx = 0
            prompt_counter = 0
            t_dt = time.time()

            while True:
                if args.n_prompts is not None and prompt_counter >= args.n_prompts:
                    break
                if args.n_prompts is None and n_records >= target_volume:
                    break
                if cursor >= len(order):
                    if temperature == 0.0 or args.n_prompts is not None:
                        break  # no cycling at greedy decoding, or smoke-test prompt cap already exhausted
                    pass_idx += 1
                    if pass_idx >= max_passes:
                        break
                    cursor = 0
                    rng.shuffle(order)

                idx = order[cursor]
                cursor += 1
                messages = [{"role": "user", "content": prompts[idx]}]
                chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                input_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)

                run_seed = seed * 1_000_003 + hash((domain, temperature, idx, pass_idx)) % 1_000_000
                _, records = run_tier0_sd(
                    target, drafter, input_ids, vocab_size, k=gamma,
                    max_new_tokens=max_new_tokens, decode_temperature=temperature,
                    probed_layers=probed_layers, tokenizer=tok, unigram_log_freq=unigram_log_freq,
                    seed=run_seed, eos_token_id=eos_id)

                prompt_id = f"{domain}_{idx}_pass{pass_idx}"
                rows = records_to_pass1_rows(records, prompt_id, domain, args.drafter, temperature)
                save_pass1(rows, pass1_dir, domain, temperature, args.drafter, shard_idx=prompt_counter)
                save_pass2_per_layer(records, probed_layers, pass2_dir, domain, temperature,
                                      args.drafter, shard_idx=prompt_counter)

                n_records += len(records)
                n_accept += sum(r["accepted"] for r in records)
                prompt_counter += 1
                if prompt_counter % 20 == 0:
                    print(f"  [{domain} T={temperature}] prompt {prompt_counter} pass={pass_idx} "
                          f"records={len(records)} cum_records={n_records}/{target_volume if args.n_prompts is None else '-'} "
                          f"accept_rate={n_accept/max(n_records,1):.3f} elapsed={time.time()-t_dt:.0f}s")

            hit_budget = args.n_prompts is None and n_records >= target_volume
            manifest["stats"][f"{domain}_T{temperature}"] = dict(
                n_records=n_records, accept_rate=n_accept / max(n_records, 1),
                n_prompts_used=prompt_counter, passes_used=pass_idx + 1,
                hit_target_volume=hit_budget, wall_clock_sec=time.time() - t_dt,
            )
            print(f"=== {domain} T={temperature} done: records={n_records} "
                  f"hit_target={hit_budget} passes={pass_idx+1} ===")

    manifest["total_wall_clock_sec"] = time.time() - t_start
    os.makedirs(pass1_dir, exist_ok=True)
    with open(os.path.join(pass1_dir, "manifest.json"), "w") as f:
        import json
        json.dump(manifest, f, indent=2)
    print(f"\nDone. Manifest -> {os.path.join(pass1_dir, 'manifest.json')}")
    print(f"Total wall clock: {manifest['total_wall_clock_sec']:.0f}s")


if __name__ == "__main__":
    main()
