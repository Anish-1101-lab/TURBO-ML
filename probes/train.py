"""
Train a probe with soft-label BCE loss against the deterministic min(1,p/q)
label, early-stopping on validation AUROC computed against the REALIZED
stochastic accept/reject outcome (see probes/metrics.py docstring for why
these are deliberately different targets for training vs. evaluation).
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train_probe(model, X_train, y_train_soft, X_val, y_val_binary,
                 lr: float, weight_decay: float, batch_size: int, max_epochs: int,
                 patience: int, device: str, seed: int):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_train.shape[0]
    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_train[idx].to(device)
            yb = y_train_soft[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device)).cpu()
        val_auroc = roc_auc_score(y_val_binary.numpy(), val_logits.numpy())

        if val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val_auroc, epoch + 1
