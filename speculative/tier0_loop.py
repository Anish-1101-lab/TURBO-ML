"""
Tier 0 instrumented SD loop. Extends speculative/sd_loop.py's algorithm
(same Leviathan & Chen accept/reject rule, same no-KV-cache-across-rounds
simplification) to:

  (a) support T=0 greedy decoding as a first-class mode -- deterministic
      argmax token selection and accept iff draft token == target argmax,
      rather than a degenerate p/q division;
  (b) extract hidden states + Group A logit-lens features at BOTH P_dec
      (input position holding x_{i-1} -- what speculative/sd_loop.py reads,
      causally blind to x_i) and P_tok (position x_i itself occupies,
      which CAN see x_i), for every probed layer -- needed for E2;
  (c) record Group A/B/C cheap features alongside the raw hidden states,
      needed for E1's cheap-feature decomposition.

Deliberate judgment call, flagged here rather than buried: every
probability-based feature (Group A logit-lens distributions, Group B
drafter q/entropy/margin) is computed at
    feature_temp = decode_temperature if decode_temperature > 0 else 1.0.
A T=0 softmax is degenerate (one-hot), so at T=0 there is no real
distribution to read "how confident was the model" from at the sampling
temperature itself -- using the natural T=1 distribution for that purpose,
while using pure argmax for the actual token selection and accept/reject
decision, keeps "which token gets picked" and "how confident was the
underlying distribution" as separate concerns. This also makes Group
A/B features directly comparable across the T=0 and T=0.7 conditions
(same softmax temperature for the descriptive statistics either way).
"""
import math

import torch

from speculative.tier0_features import lens_features, logit_lens_batch, surface_features


@torch.no_grad()
def tier0_draft_tokens(drafter, context_ids, k, vocab_size, decode_temperature, generator):
    device = context_ids.device
    feature_temp = decode_temperature if decode_temperature > 0 else 1.0
    cur_ids = context_ids
    q_scalars, q_full_dists, entropies, margins, tokens = [], [], [], [], []

    for _ in range(k):
        out = drafter(input_ids=cur_ids)
        logits = out.logits[0, -1, :vocab_size].float()
        probs = torch.softmax(logits / feature_temp, dim=-1)

        x = logits.argmax().item() if decode_temperature == 0 else \
            torch.multinomial(probs, 1, generator=generator).item()

        top2 = torch.topk(probs, 2).values
        tokens.append(x)
        q_scalars.append(probs[x].item())
        q_full_dists.append(probs)
        clamped = probs.clamp_min(1e-12)
        entropies.append(-(clamped * clamped.log()).sum().item())
        margins.append((top2[0] - top2[1]).item())
        cur_ids = torch.cat([cur_ids, torch.tensor([[x]], device=device)], dim=1)

    draft_ids = torch.tensor([tokens], device=device, dtype=context_ids.dtype)
    return draft_ids, q_scalars, q_full_dists, entropies, margins


@torch.no_grad()
def tier0_verify_and_step(target, context_ids, draft_ids, q_scalars, q_full_dists,
                           entropies, margins, vocab_size, decode_temperature,
                           probed_layers, tokenizer, unigram_log_freq, generator):
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)
    feature_temp = decode_temperature if decode_temperature > 0 else 1.0

    out = target(input_ids=full_ids, output_hidden_states=True)

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos_dec = n - 1 + i   # causally blind to x_i -- the position speculative/sd_loop.py reads
        pos_tok = n + i        # x_i's own position -- can see x_i
        x_i = draft_ids[0, i].item()

        raw_logits_dec = out.logits[0, pos_dec, :vocab_size].float()
        p_dist_feat = torch.softmax(raw_logits_dec / feature_temp, dim=-1)
        p_i = p_dist_feat[x_i].item()

        if decode_temperature == 0:
            target_argmax = raw_logits_dec.argmax().item()
            accepted = (x_i == target_argmax)
            label = float(accepted)
        else:
            q_i = q_scalars[i]
            label = min(1.0, p_i / q_i)
            r = torch.rand(1, generator=generator, device=generator.device).item()
            accepted = r < label

        feat = dict(
            position_in_draft_window=i + 1,
            prefix_length=n + i,
            log_prefix_length=math.log(n + i + 1),
            q_draft=q_scalars[i], q_entropy=entropies[i], q_margin=margins[i],
            log_unigram_freq=unigram_log_freq[x_i].item(),
            **surface_features(tokenizer, x_i),
        )

        h_dec_batch = torch.stack([out.hidden_states[L][0, pos_dec, :] for L in probed_layers])
        h_tok_batch = torch.stack([out.hidden_states[L][0, pos_tok, :] for L in probed_layers])
        probs_dec = logit_lens_batch(target, h_dec_batch, vocab_size)
        probs_tok = logit_lens_batch(target, h_tok_batch, vocab_size)

        lens = {}
        for li, L in enumerate(probed_layers):
            lens[f"pdec_L{L}"] = lens_features(probs_dec[li], x_i)
            lens[f"ptok_L{L}"] = lens_features(probs_tok[li], x_i)

        hs_dec = {L: out.hidden_states[L][0, pos_dec, :].to(torch.float16).cpu() for L in probed_layers}
        hs_tok = {L: out.hidden_states[L][0, pos_tok, :].to(torch.float16).cpu() for L in probed_layers}

        records.append(dict(
            position_dec=pos_dec, position_tok=pos_tok, token_id=x_i,
            p=p_i, q=q_scalars[i], label=label, accepted=accepted,
            features=feat, lens=lens, hs_dec=hs_dec, hs_tok=hs_tok,
        ))

        if accepted:
            accepted_count += 1
        else:
            if decode_temperature == 0:
                resample_token = target_argmax
            else:
                residual = torch.clamp(p_dist_feat - q_full_dists[i], min=0.0)
                residual = residual / residual.sum()
                resample_token = torch.multinomial(residual, 1, generator=generator).item()
            break

    used_bonus = False
    if resample_token is not None:
        new_token = resample_token
    else:
        bonus_pos = n - 1 + k
        bonus_logits = out.logits[0, bonus_pos, :vocab_size].float()
        if decode_temperature == 0:
            new_token = bonus_logits.argmax().item()
        else:
            bonus_probs = torch.softmax(bonus_logits / decode_temperature, dim=-1)
            new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()
        used_bonus = True

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids, accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return records, new_context_ids, accepted_count, used_bonus


def run_tier0_sd(target, drafter, input_ids, vocab_size, k, max_new_tokens,
                  decode_temperature, probed_layers, tokenizer, unigram_log_freq,
                  seed, eos_token_id=None):
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    context_ids = input_ids
    all_records = []
    n_generated = 0

    while n_generated < max_new_tokens:
        draft_ids, q_scalars, q_full_dists, entropies, margins = tier0_draft_tokens(
            drafter, context_ids, k, vocab_size, decode_temperature, generator)
        records, new_context_ids, accepted_count, used_bonus = tier0_verify_and_step(
            target, context_ids, draft_ids, q_scalars, q_full_dists, entropies, margins,
            vocab_size, decode_temperature, probed_layers, tokenizer, unigram_log_freq, generator)
        all_records.extend(records)
        new_tokens_this_round = new_context_ids[0, context_ids.shape[1]:]
        n_generated += new_tokens_this_round.numel()
        context_ids = new_context_ids

        if eos_token_id is not None and (new_tokens_this_round == eos_token_id).any():
            break

    return context_ids, all_records
