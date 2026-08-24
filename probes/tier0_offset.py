"""
Training loop for M_offset: identical to probes/train.py's train_probe,
except the model's raw output has a FIXED (non-trainable) per-example offset
added before the loss/AUROC -- the cheap model's (M_ABC) own logit. If the
residual stream adds nothing beyond the cheap features, the trainable part's
weights should shrink toward zero (via weight_decay) and AUROC should
converge to M_ABC's.
"""
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def train_offset_probe(model, X_train, y_train_soft, offset_train,
                        X_val, y_val_binary, offset_val,
                        lr, weight_decay, batch_size, max_epochs, patience, device, seed):
    torch.manual_seed(seed)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    n = X_train.shape[0]
    best_val_auroc = -1.0
    best_state = None
    epochs_no_improve = 0
    epoch = 0

    for epoch in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb, ob = X_train[idx].to(device), y_train_soft[idx].to(device), offset_train[idx].to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb) + ob, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_logits = (model(X_val.to(device)) + offset_val.to(device)).cpu()
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


@torch.no_grad()
def predict_offset_logits(model, X, offset, device: str, batch_size: int = 8192):
    model.eval()
    out = []
    for i in range(0, X.shape[0], batch_size):
        xb = X[i:i + batch_size].to(device)
        ob = offset[i:i + batch_size].to(device)
        out.append((model(xb) + ob).cpu())
    return torch.cat(out)
