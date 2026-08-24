"""
Phase 5: gated verification. Two variants of the verify step, sharing the
same gate DECISION logic (probes/gate.py) but differing in whether they
actually skip compute:

- `gated_verify_and_step_shadow`: ALWAYS runs the full standard forward pass
  (every layer, same cost as the ungated baseline -- no speed benefit).
  Used only to validate losslessness (Phase 5 step 2), where we need to know
  the REAL accept/reject outcome at every position in order to check whether
  the gate's decision agrees with it. The gate's decision (not the real
  draw) determines what actually gets committed to the generated
  trajectory -- matching exactly what the fast path below would do -- while
  the real outcome is logged alongside for direct comparison.

- `gated_verify_and_step_fast`: uses speculative/layer_skip.py to actually
  skip layers 25-28 for gated positions. Used only for wall-clock
  benchmarking (Phase 5 step 3), once step 2 has passed. Does NOT know the
  real accept/reject outcome for gated positions -- that's the entire point
  of skipping the computation that would produce it.

Both variants preserve the SAME per-position torch.rand() call, in the SAME
order, regardless of whether the gate fires -- mirroring the rigor already
established in Phase 4b's causal_verify_and_step_positional (see its
docstring). This keeps RNG state aligned with the ungated baseline
(speculative/sd_loop.py's verify_and_step) as long as possible, so that any
divergence measured in step 2 is attributable only to actual gate/exact-rule
disagreements, not to incidental drift in the random stream.

The gate NEVER decides a rejection. A rejection can only be produced by the
real accept/reject rule, which by construction only runs when the gate does
not fire. See probes/gate.py's should_gate docstring.
"""
import torch

from probes.gate import gate_probability, should_gate
from speculative.layer_skip import forward_prefix, forward_suffix
from speculative.sd_loop import draft_tokens


@torch.no_grad()
def gated_verify_and_step_shadow(target, context_ids: torch.Tensor, draft_ids: torch.Tensor,
                                  q_scalars: list, q_full_dists: list,
                                  vocab_size: int, temperature: float, generator: torch.Generator,
                                  gate_model, gate_mu, gate_sigma, gate_threshold: float,
                                  probe_layer: int):
    """Validation-mode gated verify step. Full compute, every position --
    see module docstring. Returns (records, new_context_ids, used_bonus)."""
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)

    out = target(input_ids=full_ids, output_hidden_states=True)

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos = n - 1 + i
        target_logits = out.logits[0, pos, :vocab_size].float()
        p_dist = torch.softmax(target_logits / temperature, dim=-1)
        x_i = draft_ids[0, i].item()
        q_i = q_scalars[i]
        p_i = p_dist[x_i].item()
        label = min(1.0, p_i / q_i)

        h_pos = out.hidden_states[probe_layer][0, pos, :]
        gate_prob = gate_probability(gate_model, gate_mu, gate_sigma, h_pos)
        gated = should_gate(gate_prob, gate_threshold)

        r = torch.rand(1, generator=generator, device=generator.device).item()
        real_accept = r < label
        commit_accept = True if gated else real_accept
        mismatch = gated and not real_accept  # gate said accept, exact rule would have rejected

        records.append(dict(
            position=pos, token_id=x_i, q=q_i, p=p_i, label=label,
            gate_prob=gate_prob, gated=gated, real_accept=real_accept,
            commit_accept=commit_accept, mismatch=mismatch,
        ))

        if commit_accept:
            accepted_count += 1
        else:
            residual = torch.clamp(p_dist - q_full_dists[i], min=0.0)
            residual = residual / residual.sum()
            resample_token = torch.multinomial(residual, 1, generator=generator).item()
            break

    used_bonus = False
    if resample_token is not None:
        new_token = resample_token
    else:
        bonus_pos = n - 1 + k
        bonus_logits = out.logits[0, bonus_pos, :vocab_size].float()
        bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
        new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()
        used_bonus = True

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids, accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return records, new_context_ids, used_bonus


def run_gated_speculative_decoding_shadow(target, drafter, input_ids: torch.Tensor, vocab_size: int,
                                           k: int, max_new_tokens: int, temperature: float, seed: int,
                                           gate_model, gate_mu, gate_sigma, gate_threshold: float,
                                           probe_layer: int, eos_token_id: int = None):
    """Drives a full generation using gated_verify_and_step_shadow -- i.e.
    the GATE's decisions (not the real accept/reject draw) determine what
    actually gets committed, exactly mirroring what gated_verify_and_step_fast
    would produce, while still paying full compute so every position's real
    outcome is known for comparison. See speculative/sd_loop.py's
    run_speculative_decoding, which this mirrors structurally so that, given
    the same seed and prompt, the only source of trajectory divergence
    between the two is actual gate/exact-rule disagreement."""
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    context_ids = input_ids
    all_records = []
    n_generated = 0

    while n_generated < max_new_tokens:
        draft_ids, q_scalars, q_full_dists, entropies = draft_tokens(
            drafter, context_ids, k, vocab_size, temperature, generator)
        records, new_context_ids, used_bonus = gated_verify_and_step_shadow(
            target, context_ids, draft_ids, q_scalars, q_full_dists,
            vocab_size, temperature, generator,
            gate_model, gate_mu, gate_sigma, gate_threshold, probe_layer)
        all_records.extend(records)
        new_tokens_this_round = new_context_ids[0, context_ids.shape[1]:]
        n_generated += new_tokens_this_round.numel()
        context_ids = new_context_ids

        if eos_token_id is not None and (new_tokens_this_round == eos_token_id).any():
            break

    return context_ids, all_records


@torch.no_grad()
def gated_verify_and_step_fast(target, context_ids: torch.Tensor, draft_ids: torch.Tensor,
                                q_scalars: list, q_full_dists: list,
                                vocab_size: int, temperature: float, generator: torch.Generator,
                                gate_model, gate_mu, gate_sigma, gate_threshold: float,
                                n_prefix_layers: int, n_total_layers: int):
    """Speed-mode gated verify step. Real layer-skip via
    speculative/layer_skip.py: the 24-layer prefix is always paid (shared
    across the whole round, needed for the gate regardless of its
    decisions). The 4-layer suffix is only ever computed for positions the
    gate does NOT clear, and -- when it IS computed -- restricted to the
    shortest prefix length that covers the position currently being
    resolved, not the whole round. See speculative/layer_skip.py's
    module docstring for why this does not guarantee the full 4/28 ceiling
    in aggregate (a round that reaches the bonus-token step, or where
    several positions in a row need real verification, still pays close to
    full cost) -- that's exactly what step 3 measures rather than assumes.

    Returns (records, new_context_ids, used_bonus). Records for gated
    positions do NOT contain a real p/label -- that's the point."""
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)

    hs_prefix = forward_prefix(target, full_ids, n_prefix_layers)  # [1, n+k, hidden_dim], always paid

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos = n - 1 + i
        h_pos = hs_prefix[0, pos, :]
        gate_prob = gate_probability(gate_model, gate_mu, gate_sigma, h_pos)
        gated = should_gate(gate_prob, gate_threshold)

        r = torch.rand(1, generator=generator, device=generator.device).item()  # alignment, see module docstring

        if gated:
            accepted_count += 1
            records.append(dict(position=pos, token_id=draft_ids[0, i].item(),
                                 gated=True, gate_prob=gate_prob))
            continue

        suffix_len = pos + 1  # only as much of the sequence as this position needs
        logits = forward_suffix(target, hs_prefix[:, :suffix_len, :], n_total_layers, n_prefix_layers)
        p_dist = torch.softmax(logits[0, -1, :vocab_size].float() / temperature, dim=-1)
        x_i = draft_ids[0, i].item()
        q_i = q_scalars[i]
        p_i = p_dist[x_i].item()
        label = min(1.0, p_i / q_i)
        accept = r < label

        records.append(dict(position=pos, token_id=x_i, gated=False, gate_prob=gate_prob,
                             p=p_i, label=label, accepted=accept))

        if accept:
            accepted_count += 1
        else:
            residual = torch.clamp(p_dist - q_full_dists[i], min=0.0)
            residual = residual / residual.sum()
            resample_token = torch.multinomial(residual, 1, generator=generator).item()
            break

    used_bonus = False
    if resample_token is not None:
        new_token = resample_token
    else:
        bonus_pos = n - 1 + k
        suffix_len = bonus_pos + 1  # = n + k, the full round length -- no savings on this call
        logits = forward_suffix(target, hs_prefix[:, :suffix_len, :], n_total_layers, n_prefix_layers)
        bonus_probs = torch.softmax(logits[0, -1, :vocab_size].float() / temperature, dim=-1)
        new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()
        used_bonus = True

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids, accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return records, new_context_ids, used_bonus


def run_gated_speculative_decoding_fast(target, drafter, input_ids: torch.Tensor, vocab_size: int,
                                         k: int, max_new_tokens: int, temperature: float, seed: int,
                                         gate_model, gate_mu, gate_sigma, gate_threshold: float,
                                         n_prefix_layers: int, n_total_layers: int,
                                         eos_token_id: int = None):
    """Drives a full generation using gated_verify_and_step_fast -- the real
    layer-skipping path, for wall-clock benchmarking (Phase 5 step 3).
    Mirrors run_gated_speculative_decoding_shadow's structure exactly, swapping
    in the fast verify step; see that function's docstring for why matching
    structure matters for comparability with the ungated baseline."""
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    context_ids = input_ids
    all_records = []
    n_generated = 0

    while n_generated < max_new_tokens:
        draft_ids, q_scalars, q_full_dists, entropies = draft_tokens(
            drafter, context_ids, k, vocab_size, temperature, generator)
        records, new_context_ids, used_bonus = gated_verify_and_step_fast(
            target, context_ids, draft_ids, q_scalars, q_full_dists,
            vocab_size, temperature, generator,
            gate_model, gate_mu, gate_sigma, gate_threshold,
            n_prefix_layers, n_total_layers)
        all_records.extend(records)
        new_tokens_this_round = new_context_ids[0, context_ids.shape[1]:]
        n_generated += new_tokens_this_round.numel()
        context_ids = new_context_ids

        if eos_token_id is not None and (new_tokens_this_round == eos_token_id).any():
            break

    return context_ids, all_records
