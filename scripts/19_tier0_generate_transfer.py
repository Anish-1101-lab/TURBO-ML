"""
E3 drafter-swap generation: a smaller MATCHED subsample for drafters B and C
(not the full ~150k/domain volume drafter A gets) -- ~40k events/domain,
split train/val/test using the SAME deterministic (domain, idx)-hash split
as drafter A (probes/tier0_splits.assign_split), so a prompt assigned "test"
is guaranteed never in ANY drafter's train set, without needing B/C to touch
the identical observed prompt subset A happened to walk through. T=0.7 only
(the deployed/stochastic regime) -- judgment call, see NOTES.md: keeps E3's
added generation cost to ~1-1.5h/drafter instead of ~8h, since the primary
E3 question (does a frozen-A probe transfer) doesn't need E1/E2's full
cheap-feature-decomposition volume.

Usage: python scripts/19_tier0_generate_transfer.py --config configs/tier0_config.yaml \
    --drafter B --out data/tier0_transfer
"""
import argparse
import json
import os
import random
import time

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from speculative.datasets import TIER0_DOMAIN_LOADERS
from speculative.tier0_features import build_unigram_log_freq
from speculative.tier0_loop import run_tier0_sd
from speculative.tier0_storage import records_to_pass1_rows, save_pass1, save_pass2_per_layer
from probes.tier0_splits import assign_split

DTYPE_MAP = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier0_config.yaml")
    ap.add_argument("--drafter", required=True, choices=["B", "C"])
    ap.add_argument("--domains", nargs="+", default=None)
    ap.add_argument("--temperatures", nargs="+", type=float, default=[0.7])
    ap.add_argument("--total-events-per-domain", type=int, default=40000)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    domains = args.domains or list(cfg["domains"].keys())
    probed_layers = cfg["probed_layers"]
    gamma = cfg["gamma"]
    seed = cfg["seed"]
    split_frac = cfg["split"]

    pass1_dir = os.path.join(args.out or "data/tier0_transfer", "pass1")
    pass2_dir = os.path.join(args.out or "data/tier0_transfer", "pass2")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["cuda_visible_devices"])
    device = "cuda:0"

    tok = AutoTokenizer.from_pretrained(cfg["target"]["name"])
    target = AutoModelForCausalLM.from_pretrained(
        cfg["target"]["name"], torch_dtype=DTYPE_MAP[cfg["target"]["dtype"]]).to(device).eval()

    dspec = cfg["drafters"][args.drafter]
    cross_tok = dspec.get("cross_tokenizer", False)
    drafter_tok = AutoTokenizer.from_pretrained(dspec["name"]) if cross_tok else tok
    drafter = AutoModelForCausalLM.from_pretrained(
        dspec["name"], torch_dtype=DTYPE_MAP[dspec["dtype"]]).to(device).eval()

    vocab_size = tok.vocab_size
    eos_id = tok.eos_token_id
    print(f"target={cfg['target']['name']} drafter={args.drafter}={dspec['name']} cross_tokenizer={cross_tok}")

    freq_cache = cfg["unigram_freq"]["cache_path"]
    unigram_log_freq = torch.load(freq_cache)

    manifest = dict(drafter=args.drafter, domains=domains, temperatures=args.temperatures,
                     total_events_per_domain=args.total_events_per_domain, stats={},
                     cross_tokenizer_drop_stats={})
    t_start = time.time()
    role_budget = dict(
        train=int(round(args.total_events_per_domain * split_frac["train_frac"])),
        val=int(round(args.total_events_per_domain * split_frac["val_frac"])),
        test=int(round(args.total_events_per_domain * split_frac["test_frac"])),
    )

    for domain in domains:
        prompts = TIER0_DOMAIN_LOADERS[domain]()
        role_idxs = {"train": [], "val": [], "test": []}
        for idx in range(len(prompts)):
            key = f"{domain}_{idx}"
            role = assign_split(key, split_frac["train_frac"], split_frac["val_frac"], split_frac["test_frac"], seed)
            role_idxs[role].append(idx)

        rng = random.Random(seed + hash(domain) % 10_000)
        for role in role_idxs:
            rng.shuffle(role_idxs[role])

        for temperature in args.temperatures:
            n_records = 0
            n_accept = 0
            n_cross_tok_dropped = 0
            n_cross_tok_total = 0
            prompt_counter = 0
            t_dt = time.time()

            for role, budget in role_budget.items():
                role_records = 0
                for idx in role_idxs[role]:
                    if role_records >= budget:
                        break
                    messages = [{"role": "user", "content": prompts[idx]}]
                    chat_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    input_ids = tok(chat_text, return_tensors="pt").input_ids.to(device)

                    if cross_tok:
                        _, records, drop_stats = run_tier0_sd_cross_tokenizer(
                            target, drafter, tok, drafter_tok, input_ids, vocab_size, gamma,
                            args.max_new_tokens, temperature, probed_layers, unigram_log_freq,
                            seed * 1_000_003 + hash((domain, temperature, idx)) % 1_000_000, eos_id)
                        n_cross_tok_dropped += drop_stats["dropped"]
                        n_cross_tok_total += drop_stats["total"]
                    else:
                        run_seed = seed * 1_000_003 + hash((domain, temperature, idx)) % 1_000_000
                        _, records = run_tier0_sd(
                            target, drafter, input_ids, vocab_size, k=gamma,
                            max_new_tokens=args.max_new_tokens, decode_temperature=temperature,
                            probed_layers=probed_layers, tokenizer=tok, unigram_log_freq=unigram_log_freq,
                            seed=run_seed, eos_token_id=eos_id)

                    prompt_id = f"{domain}_{idx}_pass0_role{role}"
                    rows = records_to_pass1_rows(records, prompt_id, domain, args.drafter, temperature)
                    for r in rows:
                        r["split_role_forced"] = role
                    save_pass1(rows, pass1_dir, domain, temperature, args.drafter, shard_idx=prompt_counter)
                    save_pass2_per_layer(records, probed_layers, pass2_dir, domain, temperature,
                                          args.drafter, shard_idx=prompt_counter)

                    n_records += len(records)
                    role_records += len(records)
                    n_accept += sum(r["accepted"] for r in records)
                    prompt_counter += 1

                print(f"  [{domain} T={temperature} drafter={args.drafter}] role={role} "
                      f"records={role_records}/{budget}")

            manifest["stats"][f"{domain}_T{temperature}"] = dict(
                n_records=n_records, accept_rate=n_accept / max(n_records, 1),
                n_prompts_used=prompt_counter, wall_clock_sec=time.time() - t_dt,
            )
            if cross_tok:
                drop_rate = n_cross_tok_dropped / max(n_cross_tok_total, 1)
                manifest["cross_tokenizer_drop_stats"][f"{domain}_T{temperature}"] = dict(
                    dropped=n_cross_tok_dropped, total=n_cross_tok_total, drop_rate=drop_rate)
                print(f"  [{domain} T={temperature}] cross-tokenizer drop rate: {drop_rate:.3f}")

    manifest["total_wall_clock_sec"] = time.time() - t_start
    os.makedirs(pass1_dir, exist_ok=True)
    with open(os.path.join(pass1_dir, f"manifest_{args.drafter}.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nDone. -> {os.path.join(pass1_dir, f'manifest_{args.drafter}.json')}")
    print(f"Total wall clock: {manifest['total_wall_clock_sec']:.0f}s")


def run_tier0_sd_cross_tokenizer(target, drafter, target_tok, drafter_tok, input_ids, vocab_size,
                                  k, max_new_tokens, decode_temperature, probed_layers, unigram_log_freq,
                                  seed, eos_token_id):
    """Cross-tokenizer drafting (Llama-3.2 drafter, Qwen2.5 target): the
    drafter proposes tokens in ITS OWN vocab, gets detokenized to text, then
    re-tokenized under the TARGET's vocab. An event is dropped whenever the
    re-tokenized text doesn't collapse back to a single target token (i.e.
    the drafter's token boundary doesn't line up with any target token) --
    those proposals can't be scored against the target's per-token
    accept/reject rule at all. Drop rate is tracked and returned; per the
    spec, if it exceeds ~15% overall the comparison is confounded and should
    be dropped, not forced.
    """
    import torch as _torch

    generator = _torch.Generator(device=input_ids.device).manual_seed(seed)
    # Re-encode the same prompt text under the drafter's own tokenizer so its
    # drafting forward passes use ITS vocab/ids throughout.
    prompt_text = target_tok.decode(input_ids[0], skip_special_tokens=False)
    drafter_ids = drafter_tok(prompt_text, return_tensors="pt").input_ids.to(input_ids.device)

    context_ids = input_ids       # grows in TARGET vocab space
    d_context_ids = drafter_ids   # grows in DRAFTER vocab space
    all_records = []
    n_generated = 0
    n_dropped = 0
    n_total = 0

    while n_generated < max_new_tokens:
        # Draft k tokens in the drafter's own vocab.
        cur = d_context_ids
        draft_pieces = []
        for _ in range(k):
            out = drafter(input_ids=cur)
            logits = out.logits[0, -1, :].float()
            if decode_temperature == 0:
                tok_id = logits.argmax().item()
            else:
                probs = _torch.softmax(logits / decode_temperature, dim=-1)
                tok_id = _torch.multinomial(probs, 1, generator=generator).item()
            draft_pieces.append(tok_id)
            cur = _torch.cat([cur, _torch.tensor([[tok_id]], device=cur.device)], dim=1)

        # Detokenize the drafted PIECE (not the whole running context, to keep
        # boundary effects local) and re-tokenize under the target vocab.
        piece_text = drafter_tok.decode(draft_pieces, skip_special_tokens=True)
        n_total += 1
        retok = target_tok(piece_text, add_special_tokens=False).input_ids
        if len(retok) != 1:
            # Drafter's piece doesn't collapse to exactly one target token --
            # can't score it under the per-token accept/reject rule. Drop this
            # round: fall back to a plain greedy/sampled target continuation
            # (bonus-token-only round) so generation can still proceed, but
            # log no probe records for it.
            n_dropped += 1
            bonus_logits_full = target(input_ids=context_ids, output_hidden_states=False).logits[0, -1, :vocab_size].float()
            if decode_temperature == 0:
                new_token = bonus_logits_full.argmax().item()
            else:
                new_token = _torch.multinomial(_torch.softmax(bonus_logits_full / decode_temperature, dim=-1), 1, generator=generator).item()
            context_ids = _torch.cat([context_ids, _torch.tensor([[new_token]], device=context_ids.device)], dim=1)
            new_text = target_tok.decode([new_token], skip_special_tokens=True)
            d_context_ids = drafter_tok(target_tok.decode(context_ids[0], skip_special_tokens=False),
                                         return_tensors="pt").input_ids.to(context_ids.device)
            n_generated += 1
            if eos_token_id is not None and new_token == eos_token_id:
                break
            continue

        x_i = retok[0]
        n = context_ids.shape[1]
        full_ids = _torch.cat([context_ids, _torch.tensor([[x_i]], device=context_ids.device)], dim=1)
        out = target(input_ids=full_ids, output_hidden_states=True)
        pos_dec, pos_tok = n - 1, n
        raw_logits_dec = out.logits[0, pos_dec, :vocab_size].float()
        feature_temp = decode_temperature if decode_temperature > 0 else 1.0
        p_dist_feat = _torch.softmax(raw_logits_dec / feature_temp, dim=-1)
        p_i = p_dist_feat[x_i].item()

        if decode_temperature == 0:
            accepted = (x_i == raw_logits_dec.argmax().item())
            label = float(accepted)
        else:
            q_i = p_dist_feat[x_i].item()  # cross-tokenizer: no native drafter q for this target token; use target's own as a neutral fallback (documented limitation)
            label = min(1.0, p_i / max(q_i, 1e-12))
            accepted = (_torch.rand(1, generator=generator, device=generator.device).item() < label)

        from speculative.tier0_features import lens_features, logit_lens_batch, surface_features
        import math
        h_dec = out.hidden_states[:]  # noqa
        feat = dict(position_in_draft_window=1, prefix_length=n, log_prefix_length=math.log(n + 1),
                    q_draft=p_i, q_entropy=0.0, q_margin=0.0,
                    log_unigram_freq=unigram_log_freq[x_i].item(), **surface_features(target_tok, x_i))
        lens = {}
        h_dec_batch = _torch.stack([out.hidden_states[L][0, pos_dec, :] for L in probed_layers])
        h_tok_batch = _torch.stack([out.hidden_states[L][0, min(pos_tok, out.hidden_states[L].shape[1]-1), :] for L in probed_layers])
        probs_dec = logit_lens_batch(target, h_dec_batch, vocab_size)
        probs_tok = logit_lens_batch(target, h_tok_batch, vocab_size)
        for li, L in enumerate(probed_layers):
            lens[f"pdec_L{L}"] = lens_features(probs_dec[li], x_i)
            lens[f"ptok_L{L}"] = lens_features(probs_tok[li], x_i)
        hs_dec = {L: out.hidden_states[L][0, pos_dec, :].to(_torch.float16).cpu() for L in probed_layers}
        hs_tok = {L: out.hidden_states[L][0, min(pos_tok, out.hidden_states[L].shape[1]-1), :].to(_torch.float16).cpu() for L in probed_layers}
        all_records.append(dict(position_dec=pos_dec, position_tok=pos_tok, token_id=x_i, p=p_i, q=p_i,
                                 label=label, accepted=accepted, features=feat, lens=lens, hs_dec=hs_dec, hs_tok=hs_tok))

        context_ids = full_ids if accepted else _torch.cat(
            [context_ids, target(input_ids=context_ids).logits[0, -1, :vocab_size].float().argmax().view(1, 1)], dim=1)
        d_context_ids = drafter_tok(target_tok.decode(context_ids[0], skip_special_tokens=False),
                                     return_tensors="pt").input_ids.to(context_ids.device)
        n_generated += 1
        if eos_token_id is not None and context_ids[0, -1].item() == eos_token_id:
            break

    return context_ids, all_records, dict(dropped=n_dropped, total=n_total)


if __name__ == "__main__":
    main()
