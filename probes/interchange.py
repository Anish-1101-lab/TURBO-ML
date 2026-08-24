"""
Phase 4c: interchange patching. Does NOT modify probes/ablation.py or
probes/causal_verify.py -- reuses probes/ablation.py's AblationHook
mechanics completely unchanged.

Key observation this phase relies on: interchange-patching a single
direction is mechanically IDENTICAL to Phase 4/4b's mean-ablation of that
direction -- both replace the component of a hidden state along a unit
direction u with some target scalar value, leaving every orthogonal
component untouched:
    h_patched = h + (target_proj - (h . u)) * u
Phase 4/4b always set target_proj to the Phase 3 training-population mean
(mu . u) -- a blunt intervention toward an average. Phase 4c sets it
instead to a REAL donor example's own projection along u, taken from an
actual high-confidence or low-confidence generation -- a stronger,
targeted substitution, addressing the "mean-ablation destroys information
toward an average rather than substituting a specific alternative" caveat
already on record from Phase 4. AblationHook's `mean_proj` is a plain
mutable attribute, so no changes to that class are needed -- this file
just sets it per-patch instead of once at construction.

Position-restricted (not diffuse whole-sequence) patching is used
throughout, matching Phase 4b -- the stronger of Phase 4/4b's two designs,
already shown to avoid the cross-position-dilution confound.
"""
import torch


def clean_verify_and_collect(target, context_ids, draft_ids, q_scalars, q_full_dists,
                              vocab_size, temperature, generator, probe_layer_index):
    """Like speculative.sd_loop.verify_and_step (same exact accept/reject
    rule, same generation behavior, no patching applied here) but
    additionally returns, per drafted-token record, the exact `full_ids`
    tensor (context+draft) that produced it and its raw hidden state at
    `probe_layer_index` -- both needed later to re-run a patched forward
    pass at that exact position for a pair this record gets selected into.
    Generation itself is driven only by the real, unmodified rule."""
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)

    with torch.no_grad():
        out = target(input_ids=full_ids, output_hidden_states=True)

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos = n - 1 + i
        logits = out.logits[0, pos, :vocab_size].float()
        p_dist = torch.softmax(logits / temperature, dim=-1)
        x_i = draft_ids[0, i].item()
        q_i = q_scalars[i]
        p_i = p_dist[x_i].item()
        label = min(1.0, p_i / q_i)
        h = out.hidden_states[probe_layer_index][0, pos, :].detach().clone()

        records.append(dict(
            position=pos, token_id=x_i, p_clean=p_i, label=label,
            full_ids=full_ids.detach().clone(), h_clean=h,
        ))

        r = torch.rand(1, generator=generator, device=generator.device).item()
        if r < label:
            accepted_count += 1
        else:
            residual = torch.clamp(p_dist - q_full_dists[i], min=0.0)
            residual = residual / residual.sum()
            resample_token = torch.multinomial(residual, 1, generator=generator).item()
            break

    if resample_token is not None:
        new_token = resample_token
    else:
        bonus_logits = out.logits[0, n - 1 + k, :vocab_size].float()
        bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
        new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids, accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return records, new_context_ids


def patched_p_at_position(target, hook, direction, donor_h, full_ids, pos, x_id,
                           vocab_size, temperature):
    """Splice donor_h's projection along `direction` into the recipient's
    forward pass at `pos` only (hook.target_positions restricts to that one
    position), leaving the rest of the recipient's own activations
    untouched. Returns the patched p(x_id) at that position."""
    donor_proj = (donor_h.to(direction.device).float() * direction).sum().item()
    hook.mean_proj = donor_proj
    hook.target_positions = [pos]
    hook.active = True
    with torch.no_grad():
        out = target(input_ids=full_ids)
    hook.active = False
    p = torch.softmax(out.logits[0, pos, :vocab_size].float() / temperature, dim=-1)[x_id].item()
    return p
