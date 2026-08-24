"""
E1 cheap-feature decomposition: M0/M_A/M_AB/M_ABC/M_h/M_full/M_offset per
layer, evaluated at P_dec (the position the existing Phase 3 probe reads --
see Step 0 finding), pooled across domains for training with per-domain
slicing for reporting (mirrors Phase 3's own pooled-training/per-domain-
reporting design).

Judgment call: M_A/M_AB/M_ABC (plain sklearn LogisticRegression) are trained
directly on the realized binary `accepted` outcome, not the soft min(1,p/q)
label -- sklearn's LogisticRegression doesn't take soft targets, and doing a
custom soft-label cross-entropy fit for these small cheap-feature models
would add real complexity for no comparability gain, since every model here
(cheap or not) is ultimately EVALUATED against the same realized-outcome
test AUROC regardless of what it was trained on. M_h/M_full/M_offset keep
the existing soft-label training convention (probes/train.py) for continuity
with Phase 3.
"""
import numpy as np
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score

from probes.metrics import expected_calibration_error, predict_proba
from probes.models import LinearProbe, MLPProbe
from probes.tier0_data import GROUP_B_COLS, GROUP_C_COLS, group_a_cols
from probes.tier0_offset import predict_offset_logits, train_offset_probe
from probes.train import train_probe

C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


def cheap_feature_matrix(df, layer: int, groups: str, position: str = "pdec"):
    """groups: subset of 'ABC', e.g. 'A', 'AB', 'ABC'. position: 'pdec' or 'ptok'
    -- selects which position's Group A logit-lens columns to use (Group B/C
    are position-independent, they're properties of the drafted token/context,
    not of which hidden state is being read)."""
    cols = []
    if "A" in groups:
        cols += group_a_cols(layer, position)
    if "B" in groups:
        cols += GROUP_B_COLS
    if "C" in groups:
        cols += GROUP_C_COLS
    return df[cols].to_numpy(dtype=np.float64), cols


def fit_cheap_logreg(X_train, y_train, X_val, y_val):
    best = None
    for C in C_GRID:
        clf = LogisticRegression(max_iter=3000, C=C)
        clf.fit(X_train, y_train)
        try:
            auroc = roc_auc_score(y_val, clf.decision_function(X_val))
        except ValueError:
            continue
        if best is None or auroc > best[0]:
            best = (auroc, clf, C)
    return best[1], best[2], best[0]


def auroc_auprc_ece(y_true, y_prob, n_bins=15):
    from sklearn.metrics import average_precision_score
    auroc = roc_auc_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_prob) if len(set(y_true)) > 1 else float("nan")
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
    return auroc, auprc, ece


def bernoulli_loglik(y_true, y_prob):
    p = np.clip(y_prob, 1e-12, 1 - 1e-12)
    return float(np.sum(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def bootstrap_prompt_indices(split_keys_test: np.ndarray, n_boot: int, seed: int):
    rng = np.random.RandomState(seed)
    unique_prompts = np.unique(split_keys_test)
    key_to_rows = {k: np.where(split_keys_test == k)[0] for k in unique_prompts}
    for b in range(n_boot):
        draw = rng.choice(unique_prompts, size=len(unique_prompts), replace=True)
        yield np.concatenate([key_to_rows[k] for k in draw])


def fit_probe_pair(X_train_h, y_train_soft, y_train_bin, X_val_h, y_val_bin, X_test_h,
                    hidden_dim, dropout, train_cfg, device, seed):
    """Returns dict {'linear': (model, test_logits_fn), 'mlp': (...)} -- fits both
    architectures on hs_dec (or hs_dec+cheap concat, caller decides X_*_h content)."""
    out = {}
    for arch_name, ctor in (("linear", lambda d: LinearProbe(d)), ("mlp", lambda d: MLPProbe(d, hidden_dim, dropout))):
        model = ctor(X_train_h.shape[1])
        model, val_auroc, epochs = train_probe(
            model, X_train_h, y_train_soft, X_val_h, y_val_bin,
            lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
            batch_size=train_cfg["batch_size"], max_epochs=train_cfg["max_epochs"],
            patience=train_cfg["patience"], device=device, seed=seed)
        test_prob = predict_proba(model, X_test_h, device)
        out[arch_name] = dict(model=model, val_auroc=val_auroc, epochs=epochs, test_prob=test_prob)
    return out


def fit_offset_probe_pair(X_train_h, y_train_soft, offset_train, X_val_h, y_val_bin, offset_val,
                           X_test_h, offset_test, hidden_dim, dropout, train_cfg, device, seed):
    out = {}
    for arch_name, ctor in (("linear", lambda d: LinearProbe(d)), ("mlp", lambda d: MLPProbe(d, hidden_dim, dropout))):
        model = ctor(X_train_h.shape[1])
        model, val_auroc, epochs = train_offset_probe(
            model, X_train_h, y_train_soft, offset_train, X_val_h, y_val_bin, offset_val,
            lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"],
            batch_size=train_cfg["batch_size"], max_epochs=train_cfg["max_epochs"],
            patience=train_cfg["patience"], device=device, seed=seed)
        test_logits = predict_offset_logits(model, X_test_h, offset_test, device)
        test_prob = torch.sigmoid(test_logits).numpy()
        out[arch_name] = dict(model=model, val_auroc=val_auroc, epochs=epochs, test_prob=test_prob,
                               test_logits=test_logits.numpy())
    return out


def reverse_decodability_r2(hs_train, hs_test, feat_train, feat_test, alpha=10.0):
    """R^2 of a Ridge regression from hidden state -> cheap feature (train-fit, test-eval)."""
    reg = Ridge(alpha=alpha)
    reg.fit(hs_train, feat_train)
    pred = reg.predict(hs_test)
    ss_res = np.sum((feat_test - pred) ** 2)
    ss_tot = np.sum((feat_test - feat_test.mean()) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
