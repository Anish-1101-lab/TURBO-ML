"""
Mean-ablation of a single direction in the residual stream at a target
decoder layer, via a forward hook -- used for Phase 4's causal check.

For a chosen unit direction u and hidden state h (raw, unstandardized):
    h_ablated = h + (mean_proj - (h . u)) * u
i.e. replace exactly the component of h along u with its typical
(training-population) value, leaving every orthogonal component untouched.
mean_proj = mu . u, where mu is the per-dimension training-set mean already
saved alongside the Phase 3 linear probe -- this works for ANY unit
direction (not just the probe's own), which is what lets the random-
direction control use the exact same ablation mechanics as the real one.
"""
import torch


def probe_direction_from_checkpoint(ckpt: dict) -> torch.Tensor:
    """Direction in RAW hidden-state space that the (standardized-input)
    linear probe reads its score from: score = w . ((h - mu) / sigma) + b,
    so d(score)/dh is proportional to w / sigma."""
    w = ckpt["state_dict"]["linear.weight"].squeeze(0)  # [hidden_dim]
    sigma = ckpt["sigma"].squeeze(0)  # [hidden_dim]
    direction = w / sigma
    return direction / direction.norm()


def random_direction(hidden_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(hidden_dim, generator=g)
    return v / v.norm()


class AblationHook:
    """Registers on model.model.layers[layer_idx_0based]; while `active` is
    True, ablates `direction` (unit vector, raw hidden-state space) from
    that layer's output. Toggle `active` instead of add/removing the hook
    repeatedly, so clean/ablated forward passes can reuse one hook object.

    `target_positions`: None (default) ablates every position in the
    sequence -- the original Phase 4 "diffuse" experiment. Set to a list
    of 0-indexed sequence positions to restrict the ablation to only those
    positions (all other positions pass through unmodified) -- the Phase 4
    position-restricted follow-up, which avoids ablating positions other
    than the one actually being measured."""

    def __init__(self, layer_module, direction: torch.Tensor, mean_proj: float, device):
        self.direction = direction.to(device)
        self.mean_proj = mean_proj
        self.active = False
        self.target_positions = None
        self.handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        if not self.active:
            return output
        h = output
        direction = self.direction.to(h.dtype)
        proj = (h * direction).sum(dim=-1, keepdim=True)  # [.., 1]
        ablated = h + (self.mean_proj - proj) * direction
        if self.target_positions is None:
            return ablated
        mask = torch.zeros(h.shape[1], dtype=torch.bool, device=h.device)
        mask[self.target_positions] = True
        mask = mask.view(1, -1, 1)
        return torch.where(mask, ablated, h)

    def remove(self):
        self.handle.remove()
