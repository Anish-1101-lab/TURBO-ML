"""
Post-hoc diagnostic for the Phase 5 report's discussion section: WHY was
reasoning slower under gating while code was faster (PROGRESS.md Phase 5
step 3)? Not answered by anything already logged -- step 3's benchmark
script only saved aggregate n_gated/n_positions per prompt, not per-round
or per-position detail. This is the minimal rerun needed to get that
detail; it adds NO new mechanism code (reuses
run_gated_speculative_decoding_shadow exactly as step 2 did) and does NOT
re-measure timing -- gate decisions are a deterministic function of hidden
state + probe + threshold, identical between the shadow and fast paths (see
scripts/08_verify_layer_skip.py), so this reproduces step 3's gating
pattern exactly without needing the timed/fast path.

Same 60 held-out prompts, same seeds, same threshold=0.995 as step 3.

Reconstructs round boundaries from the flat per-position record list alone
(no round index was ever logged): within a round, drafted positions are
consecutive integers (pos, pos+1, ...); a new round always starts at a
position that breaks that run, since a round either stops early on
rejection (next round starts further ahead) or runs all k positions plus a
bonus token (next round starts k+1 ahead). A maximal run of consecutive
positions IS a round.

Reports, per domain:
  - mean/median gated vs. really-verified ("real") positions per round
  - mean number of real-verification events per round (i.e. how many
    separate no-cache suffix recomputations a round triggers -- candidate
    mechanism: reasoning's higher confidence variance could mean more
    scattered real-verification calls per round, not just more of them)
  - WHERE in the completion gated positions fire: bucketed into
    early/mid/late thirds of that prompt's generated length

Usage: python scripts/11_gate_round_analysis.py
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
from speculative.gated_sd_loop import gated_verify_and_step_shadow
from speculative.sd_loop import draft_tokens

THRESHOLD = 0.995

# NOTE: an earlier version of this script inferred round boundaries
# post-hoc from gaps in the position sequence (a round that ends via
# rejection has NO position gap before the next round -- only a round
# that fully completes with a bonus token does, since the bonus position
# is never logged as a record). That silently merged consecutive rounds
# together whenever a rejection occurred, which is common (rejection is
# the normal, frequent SD outcome, not rare). Fixed by tracking round
# boundaries directly during generation instead of reconstructing them
# after the fact -- see the manual per-round loop in main() below, which
# mirrors run_gated_speculative_decoding_shadow's loop but keeps each
# round's records as a separate list rather than flattening them.


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
    probe_layer = gcfg["probe_layer"]

    per_prompt_rounds = []

    for domain in ["code", "reasoning", "chat"]:
        prompts = DOMAIN_LOADERS[domain]()
        rng = random.Random(scfg["seed"] + 1)
        order = list(range(len(prompts)))
        rng.shuffle(order)
        chosen = order[:args.n_prompts_per_domain]

        for pi, idx in enumerate(chosen):
            messages = [{"role": "user", "content": prompts[idx]}]
            chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            context_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)
            prompt_len = context_ids.shape[1]
            seed = scfg["seed"] * 7 + pi

            # Manual generation loop (mirrors run_gated_speculative_decoding_shadow)
            # so each round's records are kept separate -- exact round boundaries,
            # no post-hoc inference. See NOTE above.
            generator = torch.Generator(device=device).manual_seed(seed)
            cur_context = context_ids
            n_generated = 0
            rounds = []
            while n_generated < args.max_new_tokens:
                draft_ids, q_scalars, q_full_dists, entropies = draft_tokens(
                    drafter, cur_context, scfg["k"], vocab_size, scfg["temperature"], generator)
                records, new_context_ids, used_bonus = gated_verify_and_step_shadow(
                    target, cur_context, draft_ids, q_scalars, q_full_dists,
                    vocab_size, scfg["temperature"], generator,
                    gate_model, gate_mu, gate_sigma, THRESHOLD, probe_layer)
                rounds.append(records)
                new_tokens_this_round = new_context_ids[0, cur_context.shape[1]:]
                n_generated += new_tokens_this_round.numel()
                cur_context = new_context_ids
                if eos_id is not None and (new_tokens_this_round == eos_id).any():
                    break

            gated_ids = cur_context
            gated_records = [r for rd in rounds for r in rd]
            gen_len = gated_ids.shape[1] - prompt_len

            round_stats = []
            for rd in rounds:
                n_gated_in_round = sum(1 for r in rd if r["gated"])
                n_real_in_round = sum(1 for r in rd if not r["gated"])
                round_stats.append(dict(size=len(rd), n_gated=n_gated_in_round, n_real=n_real_in_round))

            fire_positions = []
            for r in gated_records:
                if r["gated"]:
                    frac = (r["position"] - prompt_len) / max(1, gen_len)
                    fire_positions.append(frac)

            per_prompt_rounds.append(dict(
                domain=domain, idx=idx, prompt_len=prompt_len, gen_len=gen_len,
                n_rounds=len(rounds), round_stats=round_stats, fire_position_fracs=fire_positions,
            ))
            print(f"[{domain}] prompt {pi + 1}/{len(chosen)}: n_rounds={len(rounds)} "
                  f"gen_len={gen_len} n_gated={sum(r['n_gated'] for r in round_stats)}")

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    def median(xs):
        s = sorted(xs)
        n = len(s)
        if n == 0:
            return 0.0
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    summary = {}
    for domain in ["code", "reasoning", "chat"]:
        dp = [p for p in per_prompt_rounds if p["domain"] == domain]
        all_rounds = [rd for p in dp for rd in p["round_stats"]]
        gated_per_round = [rd["n_gated"] for rd in all_rounds]
        real_per_round = [rd["n_real"] for rd in all_rounds]
        real_events_per_round_nonzero = [rd["n_real"] for rd in all_rounds if rd["n_real"] > 0]
        rounds_with_any_real = sum(1 for rd in all_rounds if rd["n_real"] > 0)
        all_fire_fracs = [f for p in dp for f in p["fire_position_fracs"]]
        early = sum(1 for f in all_fire_fracs if f < 1 / 3)
        mid = sum(1 for f in all_fire_fracs if 1 / 3 <= f < 2 / 3)
        late = sum(1 for f in all_fire_fracs if f >= 2 / 3)
        n_fire = len(all_fire_fracs)

        summary[domain] = dict(
            n_prompts=len(dp),
            mean_rounds_per_prompt=mean([p["n_rounds"] for p in dp]),
            mean_gen_len=mean([p["gen_len"] for p in dp]),
            mean_gated_per_round=mean(gated_per_round), median_gated_per_round=median(gated_per_round),
            mean_real_per_round=mean(real_per_round), median_real_per_round=median(real_per_round),
            frac_rounds_with_any_real_verification=rounds_with_any_real / len(all_rounds) if all_rounds else 0.0,
            mean_real_events_per_round_when_nonzero=mean(real_events_per_round_nonzero),
            n_gate_fire_events=n_fire,
            fire_position_early_frac=early / n_fire if n_fire else 0.0,
            fire_position_mid_frac=mid / n_fire if n_fire else 0.0,
            fire_position_late_frac=late / n_fire if n_fire else 0.0,
        )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "round_composition.json"), "w") as f:
        json.dump(dict(summary=summary, per_prompt=per_prompt_rounds), f, indent=2)
    print(f"\nSaved to {args.out}/round_composition.json")


if __name__ == "__main__":
    main()
