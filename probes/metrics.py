"""
Evaluation metrics for probes.

Deliberate design choice: the probe is TRAINED on the deterministic
min(1,p/q) label (soft target, avoids coin-flip label noise -- see
PROGRESS.md Phase 1). But AUROC and calibration are inherently about how
well a predicted probability matches REALIZED binary outcomes -- there is
no other honest way to define them. So evaluation here uses the actual
stochastic `accepted` outcome as ground truth, which is exactly the
standard way any probabilistic forecast gets validated (e.g. a weather
model's rain probability is checked against whether it actually rained,
not against some other deterministic proxy). Training and evaluation
targets are intentionally different for this reason; this is not an
inconsistency.
"""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


@torch.no_grad()
def predict_proba(model, X: torch.Tensor, device: str, batch_size: int = 4096) -> np.ndarray:
    model.eval()
    probs = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        probs.append(torch.sigmoid(model(xb)).cpu())
    return torch.cat(probs).numpy()


def auroc(y_true_binary, y_prob) -> float:
    return float(roc_auc_score(y_true_binary, y_prob))


def expected_calibration_error(y_true_binary, y_prob, n_bins: int = 15):
    y_true_binary = np.asarray(y_true_binary, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    ece = 0.0
    bin_stats = []
    n = len(y_prob)
    for b in range(n_bins):
        mask = bin_ids == b
        count = int(mask.sum())
        if count == 0:
            bin_stats.append(dict(bin=b, lo=float(bin_edges[b]), hi=float(bin_edges[b + 1]), count=0, conf=None, acc=None))
            continue
        conf = float(y_prob[mask].mean())
        acc = float(y_true_binary[mask].mean())
        ece += (count / n) * abs(acc - conf)
        bin_stats.append(dict(bin=b, lo=float(bin_edges[b]), hi=float(bin_edges[b + 1]), count=count, conf=conf, acc=acc))
    return float(ece), bin_stats
