"""
Instrumented speculative decoding, implemented from scratch (no
`model.generate(assistant_model=...)` shortcuts) so every step is
inspectable and every logged value is traceable to a specific tensor slice.

Algorithm: exact Leviathan & Chen (2023) accept-reject rule.
  - Drafter proposes k tokens autoregressively, sampling x_i ~ q_i.
  - Target does ONE forward pass over (context + k drafted tokens), giving
    target distributions p_1..p_k at the same conditioning points the
    drafter used, plus one extra p_{k+1} for a free "bonus" token.
  - Walk i=1..k: accept x_i with probability min(1, p_i(x_i)/q_i(x_i)).
    On first rejection, resample from residual r = normalize(max(0, p-q))
    and stop. If all k accepted, sample a bonus token from p_{k+1}.

Deliberate simplification (documented in PROGRESS.md): no incremental KV
cache across rounds. Every drafting step and every verification step
recomputes the forward pass over the full sequence so far. This avoids
cache-truncation/rollback bugs that are hard to hand-verify, at the cost of
throughput. Fine at the ~100k-token small-scale stage; revisit if Phase 2
turns out too slow.

The probe *label* logged per drafted token is the deterministic quantity
min(1, p(x)/q(x)) -- NOT the stochastic accept/reject outcome -- per the
project's label-noise requirement. It is computed for every one of the k
drafted tokens in a round regardless of where the sequential walk actually
stops, since it only depends on p and q, both of which are already fully
computed by the single parallel target forward pass.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class DraftRecord:
    position: int          # 0-indexed position in the target's input sequence
    token_id: int
    q: float                # drafter probability of this token
    p: float                # target probability of this token
    label: float             # min(1, p/q) -- the probe regression/ranking target
    drafter_entropy: float   # entropy of the full drafter distribution at this step (difficulty proxy)
    hidden_states: dict      # {layer_index: 1D tensor [hidden_dim], fp16, on CPU}
    accepted: bool            # actual stochastic outcome (for bookkeeping / sanity checks only)


@dataclass
class RoundResult:
    records: list
    new_context_ids: torch.Tensor
    num_accepted: int
    used_bonus_token: bool


def layer_indices_from_stride(num_layers: int, stride: int) -> list:
    return list(range(stride, num_layers + 1, stride))


def _entropy(probs: torch.Tensor) -> float:
    p = probs.clamp_min(1e-12)
    return (-(p * p.log()).sum()).item()


@torch.no_grad()
def draft_tokens(drafter, context_ids: torch.Tensor, k: int, vocab_size: int,
                  temperature: float, generator: torch.Generator):
    """Autoregressively sample k tokens from the drafter, full recompute each step.

    Returns: (draft_ids [1, k] long tensor, q_scalars [k] list, q_full_dists [k] list of
    [vocab_size] tensors, entropies [k] list)
    """
    device = context_ids.device
    cur_ids = context_ids
    q_scalars, q_full_dists, entropies, tokens = [], [], [], []

    for _ in range(k):
        out = drafter(input_ids=cur_ids)
        logits = out.logits[0, -1, :vocab_size].float()
        probs = torch.softmax(logits / temperature, dim=-1)
        x = torch.multinomial(probs, 1, generator=generator).item()

        tokens.append(x)
        q_scalars.append(probs[x].item())
        q_full_dists.append(probs)
        entropies.append(_entropy(probs))
        cur_ids = torch.cat([cur_ids, torch.tensor([[x]], device=device)], dim=1)

    draft_ids = torch.tensor([tokens], device=device, dtype=context_ids.dtype)
    return draft_ids, q_scalars, q_full_dists, entropies


@torch.no_grad()
def verify_and_step(target, context_ids: torch.Tensor, draft_ids: torch.Tensor,
                     q_scalars: list, q_full_dists: list, entropies: list,
                     vocab_size: int, temperature: float, layer_indices: list,
                     generator: torch.Generator) -> RoundResult:
    """One target forward pass over (context + drafted tokens); walk the
    accept/reject rule; return logged records for all k drafted tokens plus
    the updated context."""
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)

    out = target(input_ids=full_ids, output_hidden_states=True)

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos = n - 1 + i  # input position whose output logit predicts drafted token i
        target_logits = out.logits[0, pos, :vocab_size].float()
        p_dist = torch.softmax(target_logits / temperature, dim=-1)

        x_i = draft_ids[0, i].item()
        q_i = q_scalars[i]
        p_i = p_dist[x_i].item()
        label = min(1.0, p_i / q_i)

        hs = {L: out.hidden_states[L][0, pos, :].to(torch.float16).cpu() for L in layer_indices}

        r = torch.rand(1, generator=generator, device=generator.device).item()
        accept = r < label

        records.append(DraftRecord(
            position=pos, token_id=x_i, q=q_i, p=p_i, label=label,
            drafter_entropy=entropies[i], hidden_states=hs, accepted=accept,
        ))

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
        bonus_logits = out.logits[0, bonus_pos, :vocab_size].float()
        bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
        new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()
        used_bonus = True

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids,
        accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return RoundResult(records=records, new_context_ids=new_context_ids,
                        num_accepted=accepted_count, used_bonus_token=used_bonus)


def run_speculative_decoding(target, drafter, input_ids: torch.Tensor, vocab_size: int,
                              k: int, max_new_tokens: int, temperature: float,
                              layer_indices: list, seed: int):
    """Drive rounds of draft+verify until max_new_tokens have been generated
    (may overshoot slightly since a round produces 1..k+1 tokens at once)."""
    generator = torch.Generator(device=input_ids.device).manual_seed(seed)
    context_ids = input_ids
    all_records = []
    n_generated = 0

    while n_generated < max_new_tokens:
        draft_ids, q_scalars, q_full_dists, entropies = draft_tokens(
            drafter, context_ids, k, vocab_size, temperature, generator)
        result = verify_and_step(
            target, context_ids, draft_ids, q_scalars, q_full_dists, entropies,
            vocab_size, temperature, layer_indices, generator)
        all_records.extend(result.records)
        n_new = result.new_context_ids.shape[1] - context_ids.shape[1]
        context_ids = result.new_context_ids
        n_generated += n_new

    return context_ids, all_records
