"""
Phase 5 step 3: wall-clock benchmarking, threshold=0.995 ONLY. Threshold=0.98
is not benchmarked -- its 15% full-completion divergence rate (see
PROGRESS.md step 2) already rules it out as a serious candidate.

IMPORTANT: this gate is confirmed NOT lossless (5% overall full-completion
divergence at threshold=0.995, 10% for reasoning specifically, n=20 -- see
PROGRESS.md step 2). Any speedup number produced by this script must always
be reported paired with that divergence rate. A speedup number reported on
its own is a misleading statement of this result and must not appear that
way anywhere -- not in PROGRESS.md, not in plots, not in code comments.

Uses speculative/gated_sd_loop.py's REAL layer-skipping path
(gated_verify_and_step_fast via run_gated_speculative_decoding_fast) -- NOT
step 2's shadow/validation path, which deliberately paid full compute to
enable comparison. This script actually skips layers 25-28 for gated
positions.

Timing hygiene (read before trusting any number below):
  - One short throwaway generation on both models before the timed loop,
    excluded from all reported numbers (CUDA kernel warmup / caching).
  - ONE timed run per prompt per condition (ungated, gated-fast) -- no
    repeated trials of the same prompt. Aggregate statistics (mean/median)
    come from variation across 60 distinct prompts, not repeats. This is a
    real limitation: no per-prompt noise estimate, just cross-prompt spread.
  - Fixed order per prompt: ungated timed first, then gated. NOT
    alternated/randomized across prompts, so any systematic drift over the
    course of the run (thermal, clock boost, other tenants' load spiking on
    this shared GPU) would bias for or against one condition consistently
    rather than averaging out. Disclosed, not corrected for.
  - torch.cuda.synchronize() brackets each condition's ENTIRE timed
    generation (not per round, not per position) -- avoids injecting sync
    overhead into the number being measured.
  - The probe's own forward-pass cost is measured in a SEPARATE
    microbenchmark after the main loop (not interleaved with generation
    timing, which would require per-call synchronization that contaminates
    the very number being measured), using real hidden-state vectors
    collected during the gated runs, then extrapolated by the actual number
    of probe calls made -- reported separately, never folded into the
    headline gated-vs-ungated number.
  - GPU: read from nvidia-smi at run start and logged, since latency numbers
    are meaningless without knowing what hardware and utilization state
    produced them.

Usage: python scripts/10_gate_benchmark.py
"""
import argparse
import json
import os
import random
import subprocess
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from probes.gate import gate_probability, load_mlp_gate_probe
from speculative.datasets import DOMAIN_LOADERS
from speculative.gated_sd_loop import run_gated_speculative_decoding_fast
from speculative.sd_loop import run_speculative_decoding

THRESHOLD = 0.995  # the only threshold benchmarked -- see module docstring


def timed_run(fn, *fn_args, **fn_kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = fn(*fn_args, **fn_kwargs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--sd", default="configs/sd_config.yaml")
    parser.add_argument("--gate", default="configs/gate_config.yaml")
    parser.add_argument("--out", default="analysis/phase5")
    parser.add_argument("--n_prompts_per_domain", type=int, default=20)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    mcfg = yaml.safe_load(open(args.models))
    scfg = yaml.safe_load(open(args.sd))
    gcfg = yaml.safe_load(open(args.gate))
    # THRESHOLD (0.995) is authoritative for this script regardless of what
    # configs/gate_config.yaml currently has on disk -- see module docstring.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(mcfg["device"]["cuda_visible_devices"])
    device = "cuda:0"
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    gpu_info = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
         "--format=csv"], capture_output=True, text=True).stdout
    print("=== GPU state at run start ===")
    print(gpu_info)

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
    probe_layer = gcfg["probe_layer"]
    n_total_layers = gcfg["n_total_layers"]
    print(f"gate: layer={probe_layer} threshold={THRESHOLD} (0.98 NOT benchmarked -- see module docstring)")

    # --- warmup (excluded from all reported numbers) ---
    warmup_ids = tok("Hello, how are you?", return_tensors="pt").input_ids.to(device)
    run_speculative_decoding(target, drafter, warmup_ids, vocab_size, scfg["k"], 16,
                              scfg["temperature"], layer_indices=[], seed=0, eos_token_id=eos_id)
    run_gated_speculative_decoding_fast(target, drafter, warmup_ids, vocab_size, scfg["k"], 16,
                                         scfg["temperature"], 0, gate_model, gate_mu, gate_sigma,
                                         THRESHOLD, probe_layer, n_total_layers, eos_token_id=eos_id)
    print("warmup done\n")

    per_prompt = []
    collected_hidden_states = []  # for the separate probe-overhead microbenchmark
    total_gated_positions = 0
    total_positions = 0

    for domain in ["code", "reasoning", "chat"]:
        prompts = DOMAIN_LOADERS[domain]()
        rng = random.Random(scfg["seed"] + 1)  # same sampling convention as scripts/05, 07, 09
        order = list(range(len(prompts)))
        rng.shuffle(order)
        chosen = order[:args.n_prompts_per_domain]

        for pi, idx in enumerate(chosen):
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)
            seed = scfg["seed"] * 7 + pi  # same formula as scripts/05, 07, 09

            (ungated_ids, _), ungated_time = timed_run(
                run_speculative_decoding, target, drafter, context_ids, vocab_size, scfg["k"],
                args.max_new_tokens, scfg["temperature"], layer_indices=[], seed=seed, eos_token_id=eos_id)

            (gated_ids, gated_records), gated_time = timed_run(
                run_gated_speculative_decoding_fast, target, drafter, context_ids, vocab_size, scfg["k"],
                args.max_new_tokens, scfg["temperature"], seed, gate_model, gate_mu, gate_sigma,
                THRESHOLD, probe_layer, n_total_layers, eos_token_id=eos_id)

            n_gated = sum(1 for r in gated_records if r["gated"])
            total_gated_positions += n_gated
            total_positions += len(gated_records)

            ungated_new_len = ungated_ids.shape[1] - context_ids.shape[1]
            gated_new_len = gated_ids.shape[1] - context_ids.shape[1]

            per_prompt.append(dict(
                domain=domain, idx=idx, seed=seed,
                ungated_time_s=ungated_time, gated_time_s=gated_time,
                ungated_new_tokens=ungated_new_len, gated_new_tokens=gated_new_len,
                n_positions=len(gated_records), n_gated=n_gated,
            ))
            print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: ungated={ungated_time:.3f}s "
                  f"gated={gated_time:.3f}s speedup={ungated_time / gated_time:.3f}x "
                  f"n_gated={n_gated}/{len(gated_records)}")

    # --- separate probe-overhead microbenchmark ---
    # Collect real hidden states from one more short gated generation (not timed above),
    # purely to get realistic activation vectors for the isolated probe timing.
    probe_bench_ids = tok("Explain how a hash map works.", return_tensors="pt").input_ids.to(device)
    from speculative.layer_skip import forward_prefix
    with torch.no_grad():
        hs = forward_prefix(target, probe_bench_ids, probe_layer)
    sample_vecs = [hs[0, p, :].clone() for p in range(hs.shape[1])]
    n_warmup_probe_calls = 20
    n_timed_probe_calls = 500
    for i in range(n_warmup_probe_calls):
        gate_probability(gate_model, gate_mu, gate_sigma, sample_vecs[i % len(sample_vecs)])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(n_timed_probe_calls):
        gate_probability(gate_model, gate_mu, gate_sigma, sample_vecs[i % len(sample_vecs)])
    torch.cuda.synchronize()
    probe_time_per_call = (time.perf_counter() - t0) / n_timed_probe_calls

    ungated_times = [p["ungated_time_s"] for p in per_prompt]
    gated_times = [p["gated_time_s"] for p in per_prompt]

    def mean(xs):
        return sum(xs) / len(xs)

    def median(xs):
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    def domain_stats(domain):
        dp = [p for p in per_prompt if p["domain"] == domain]
        u = [p["ungated_time_s"] for p in dp]
        g = [p["gated_time_s"] for p in dp]
        gated_pos = sum(p["n_gated"] for p in dp)
        pos = sum(p["n_positions"] for p in dp)
        return dict(
            n=len(dp), mean_ungated_s=mean(u), median_ungated_s=median(u),
            mean_gated_s=mean(g), median_gated_s=median(g),
            speedup_mean=mean(u) / mean(g), speedup_median=median(u) / median(g),
            gate_fire_rate=gated_pos / pos if pos else 0.0,
        )

    summary = dict(
        threshold=THRESHOLD,
        note="PAIR EVERY SPEEDUP NUMBER HERE WITH PROGRESS.md's step 2 divergence rate "
             "(5% overall, 10% reasoning) -- this file alone does not establish losslessness.",
        n_prompts=len(per_prompt),
        overall=dict(
            mean_ungated_s=mean(ungated_times), median_ungated_s=median(ungated_times),
            mean_gated_s=mean(gated_times), median_gated_s=median(gated_times),
            speedup_mean=mean(ungated_times) / mean(gated_times),
            speedup_median=median(ungated_times) / median(gated_times),
            gate_fire_rate=total_gated_positions / total_positions if total_positions else 0.0,
        ),
        by_domain={d: domain_stats(d) for d in ["code", "reasoning", "chat"]},
        probe_overhead=dict(
            per_call_seconds=probe_time_per_call,
            n_timed_calls=n_timed_probe_calls,
            n_warmup_calls=n_warmup_probe_calls,
            total_probe_calls_in_main_run=total_positions,
            estimated_total_probe_overhead_across_all_gated_runs_seconds=(
                probe_time_per_call * total_positions),
            estimated_total_gated_wall_clock_seconds=sum(gated_times),
        ),
        timing_hygiene=dict(
            repeats_per_prompt=1, warmup_excluded=True,
            order="ungated timed first, then gated, per prompt (not alternated/randomized)",
            gpu_info_at_start=gpu_info.strip(),
        ),
    )

    print("\n=== SUMMARY (threshold=0.995; PAIR WITH step 2's 5% overall / 10% reasoning divergence rate) ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "benchmark_t0995.json"), "w") as f:
        json.dump(dict(summary=summary, per_prompt=per_prompt), f, indent=2)
    print(f"\nSaved to {args.out}/benchmark_t0995.json")


if __name__ == "__main__":
    main()
