"""
Phase 5: mechanics for actually running the target model through only a
PREFIX or SUFFIX of its decoder layers, so the gate realizes a real compute
reduction rather than a storage-only one.

This distinction matters because of a note already on record from Phase 1:
`output_hidden_states=True` computes every layer regardless of which ones
you look at afterward -- it is a storage/IO-only knob, not a compute-saving
one. To actually skip layers 25-28 for a gated position, the forward pass
itself has to be truncated.

Mechanism: temporarily swap `target.model.layers` (an `nn.ModuleList`) for a
shorter `nn.ModuleList` covering only the layers we want to run, call the
model's own real forward code on that truncated stack, then restore the
original list. This reuses the exact layer submodules and the library's own
forward logic (causal masking, rotary position embeddings, etc.) instead of
hand-reimplementing decoder internals, which would be fragile and
version-specific. Swapping one `nn.ModuleList` for another is a normal,
supported PyTorch operation (both are `nn.Module` instances); the risk this
approach does NOT have is the one a hand-rolled reimplementation would carry
(silently computing something subtly different from a real forward pass).

For the prefix pass, the model's own final RMSNorm is replaced with a no-op
for the duration of the call -- that norm is only supposed to apply once,
after the LAST decoder layer, right before the LM head. Applied after only
24 layers it would silently produce a wrongly-normalized vector, and
downstream code (the probe) expects the SAME raw, unnormalized
hidden_states[24] convention already used everywhere else in this repo
(Phase 3 training data, Phase 4/4b ablation hooks).

For the suffix pass, `inputs_embeds` is used to inject the prefix's output
hidden states directly as the input to the truncated (last-N-layers) stack,
bypassing the embedding lookup. `position_ids` must be passed explicitly as
the TRUE absolute sequence positions (not restarting at 0), since Qwen2's
rotary position embeddings depend on absolute position and layers 25-28
still need to attend correctly relative to the real sequence.

KNOWN LIMITATION (flagged, not fixed here): no KV-cache is used anywhere in
this project (established in Phase 1 for correctness-first simplicity), so
each suffix call recomputes attention from scratch over whatever length it's
given. If a round needs the suffix pass at more than one length (e.g. one
non-gated position followed by another), each call redoes the overlapping
prefix's attention work rather than incrementally extending a cache. This
is a real inefficiency, not a rounding error -- see PROGRESS.md Phase 5
"Gate design" for why this makes the ~14% ceiling a maximum, not a
guarantee, and why step 3 measures wall-clock rather than trusting the
layer-count arithmetic.

UNVALIDATED AS OF WRITING: `forward_prefix` + `forward_suffix` composed
together are claimed, not yet verified, to reproduce a real full-depth
forward pass's logits exactly. That numerical check (analogous to Phase 4's
"verified a no-op hook exactly reproduces unhooked logits before writing the
ablation math") is the first thing that should be run, before this is
trusted for anything else.
"""
import torch


def _swap_layers(target, layer_slice):
    original_layers = target.model.layers
    target.model.layers = torch.nn.ModuleList(list(original_layers[layer_slice]))
    return original_layers


@torch.no_grad()
def forward_prefix(target, input_ids: torch.Tensor, n_prefix_layers: int) -> torch.Tensor:
    """Embedding + the first `n_prefix_layers` decoder layers only. Returns
    the raw (unnormalized) residual-stream hidden state at that depth --
    i.e. exactly what output_hidden_states=True's hidden_states[n_prefix_layers]
    would give, WITHOUT computing any layer beyond it.
    Shape: [1, seq_len, hidden_dim]."""
    original_layers = _swap_layers(target, slice(0, n_prefix_layers))
    original_norm = target.model.norm
    target.model.norm = torch.nn.Identity()
    try:
        out = target.model(input_ids=input_ids, use_cache=False)
    finally:
        target.model.layers = original_layers
        target.model.norm = original_norm
    return out.last_hidden_state


@torch.no_grad()
def forward_suffix(target, hidden_state_prefix: torch.Tensor,
                    n_total_layers: int, n_prefix_layers: int) -> torch.Tensor:
    """Continue from a prefix hidden state (as produced by forward_prefix,
    optionally length-sliced by the caller to only the positions actually
    needed) through the remaining `n_total_layers - n_prefix_layers`
    decoder layers, the model's REAL final norm (untouched this time --
    this is the genuine end of the stack), and the LM head.

    `hidden_state_prefix` must cover positions [0, seq_len) of the SAME
    underlying sequence forward_prefix was run on -- position_ids are
    reconstructed as the literal range [0, seq_len), matching true absolute
    position for correct rotary embeddings.

    Returns raw logits over the full padded vocab (same convention as
    `model(...).logits` elsewhere in this repo) -- slice to
    `[..., :vocab_size]` at the call site before softmax."""
    seq_len = hidden_state_prefix.shape[1]
    device = hidden_state_prefix.device
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    original_layers = _swap_layers(target, slice(n_prefix_layers, n_total_layers))
    try:
        out = target.model(inputs_embeds=hidden_state_prefix, position_ids=position_ids, use_cache=False)
    finally:
        target.model.layers = original_layers

    return target.lm_head(out.last_hidden_state)
