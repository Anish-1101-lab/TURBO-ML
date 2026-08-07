"""
Phase 4 variant of speculative.sd_loop.verify_and_step: runs the target
forward pass THREE times per round on the identical (context, draft_ids) --
clean, probe-direction-ablated, random-direction-ablated -- and records p(x)
under all three conditions plus the probe's own pre-ablation prediction.

The actual generation path (accept/reject walk, what gets appended to the
conversation) is driven ONLY by the clean pass, exactly like Phase 1/2 --
the two ablated passes are purely diagnostic measurements and never affect
what token the conversation actually continues with.
"""
import torch


def causal_verify_and_step(target, context_ids, draft_ids, q_scalars, q_full_dists,
                            vocab_size, temperature, generator, probe_hook, random_hook,
                            probe_ckpt, probe_layer_index):
    n = context_ids.shape[1]
    k = draft_ids.shape[1]
    full_ids = torch.cat([context_ids, draft_ids], dim=1)

    probe_hook.active = False
    random_hook.active = False
    with torch.no_grad():
        out_clean = target(input_ids=full_ids, output_hidden_states=True)

    probe_hook.active = True
    with torch.no_grad():
        out_probe_abl = target(input_ids=full_ids)
    probe_hook.active = False

    random_hook.active = True
    with torch.no_grad():
        out_rand_abl = target(input_ids=full_ids)
    random_hook.active = False

    w = probe_ckpt["state_dict"]["linear.weight"].squeeze(0)
    b = probe_ckpt["state_dict"]["linear.bias"].item()
    mu = probe_ckpt["mu"].squeeze(0)
    sigma = probe_ckpt["sigma"].squeeze(0)

    records = []
    accepted_count = 0
    resample_token = None

    for i in range(k):
        pos = n - 1 + i
        logits_clean = out_clean.logits[0, pos, :vocab_size].float()
        p_dist_clean = torch.softmax(logits_clean / temperature, dim=-1)
        x_i = draft_ids[0, i].item()
        q_i = q_scalars[i]
        p_clean = p_dist_clean[x_i].item()
        label_clean = min(1.0, p_clean / q_i)

        p_probe_abl = torch.softmax(out_probe_abl.logits[0, pos, :vocab_size].float() / temperature, dim=-1)[x_i].item()
        p_rand_abl = torch.softmax(out_rand_abl.logits[0, pos, :vocab_size].float() / temperature, dim=-1)[x_i].item()

        h_clean = out_clean.hidden_states[probe_layer_index][0, pos, :].float().cpu()
        probe_logit = (w * ((h_clean - mu) / sigma)).sum() + b
        probe_score = torch.sigmoid(probe_logit).item()

        records.append(dict(
            position=pos, token_id=x_i, q=q_i, p_clean=p_clean, label_clean=label_clean,
            probe_score=probe_score, p_probe_ablated=p_probe_abl, p_random_ablated=p_rand_abl,
        ))

        r = torch.rand(1, generator=generator, device=generator.device).item()
        if r < label_clean:
            accepted_count += 1
        else:
            residual = torch.clamp(p_dist_clean - q_full_dists[i], min=0.0)
            residual = residual / residual.sum()
            resample_token = torch.multinomial(residual, 1, generator=generator).item()
            break

    if resample_token is not None:
        new_token = resample_token
    else:
        bonus_logits = out_clean.logits[0, n - 1 + k, :vocab_size].float()
        bonus_probs = torch.softmax(bonus_logits / temperature, dim=-1)
        new_token = torch.multinomial(bonus_probs, 1, generator=generator).item()

    accepted_ids = draft_ids[0, :accepted_count]
    new_context_ids = torch.cat([
        context_ids, accepted_ids.view(1, -1),
        torch.tensor([[new_token]], device=context_ids.device, dtype=context_ids.dtype),
    ], dim=1)

    return records, new_context_ids
