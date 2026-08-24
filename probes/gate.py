"""
Phase 5: verification-skipping gate.

FRAMING (see PROGRESS.md "Causal check -- final summary"): Phase 3 found the
layer-24 MLP probe is a real but modest predictor of accept/reject (AUROC
~0.85-0.86). Phase 4 + 4b found NO evidence that this probe's direction is
causally load-bearing for the target's own verification computation -- that
question came back negative under two ablation designs. This gate is used
anyway, on "predictive is enough for a systems shortcut" grounds only. It is
NOT justified by, and does not imply, any causal story about how the target
computes accept/reject internally. Nothing in this file should be read as
evidence about mechanism.

MLP probe specifically (not linear): Phase 3 found linear probes poorly
calibrated (ECE ~0.25-0.28) despite decent AUROC, while MLP probes are
well-calibrated (ECE ~0.01-0.03) at every layer. A gate that trusts a
probability threshold as "confidence" needs that probability to actually
mean what it says -- so MLP is the only defensible choice here.
"""
import torch

from probes.models import MLPProbe


def load_mlp_gate_probe(checkpoint_path: str, hidden_dim: int, mlp_hidden_dim: int,
                         dropout: float, device: str):
    ckpt = torch.load(checkpoint_path, weights_only=False)
    assert ckpt["probe_type"] == "mlp", (
        f"gate requires an MLP probe checkpoint (calibration reason above), got {ckpt['probe_type']}")
    model = MLPProbe(hidden_dim, mlp_hidden_dim, dropout)
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    mu = ckpt["mu"].to(device)       # [1, hidden_dim], Phase 3 train-split standardization stats
    sigma = ckpt["sigma"].to(device)  # [1, hidden_dim]
    return model, mu, sigma


@torch.no_grad()
def gate_probability(model, mu, sigma, hidden_state: torch.Tensor) -> float:
    """hidden_state: raw (unnormalized) layer-24 residual-stream vector,
    shape [hidden_dim], same device as mu/sigma. Standardizes with the
    frozen Phase 3 train-split mu/sigma (never refit here) before scoring,
    matching exactly how the probe was trained and evaluated."""
    x = ((hidden_state.float().unsqueeze(0) - mu) / sigma)
    logit = model(x)
    return torch.sigmoid(logit).item()


def should_gate(prob: float, threshold: float) -> bool:
    """The gate only ever pre-empts an ACCEPT. It is never consulted to
    decide a rejection -- a rejection can only come from the real
    accept/reject rule, which by construction is only computed when the
    gate does NOT fire. See speculative/gated_sd_loop.py."""
    return prob >= threshold
