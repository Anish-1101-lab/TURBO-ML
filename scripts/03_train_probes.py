"""
Phase 3: train a linear probe and a small MLP probe per logged layer,
predicting the deterministic min(1,p/q) label. One probe per (layer, probe
type), trained on data POOLED across all three domains -- this tests
whether the accept/reject signal is domain-general, and results are then
sliced by domain at evaluation time (see PROGRESS.md for why pooled vs.
per-domain probes was the chosen default).

Usage: python scripts/03_train_probes.py --config configs/probe_config.yaml
"""
import argparse
import json
import os
import time

import torch
import yaml

from probes.data import load_all_domains
from probes.metrics import auroc, expected_calibration_error, predict_proba
from probes.models import LinearProbe, MLPProbe
from probes.splits import make_split
from probes.train import train_probe

DOMAINS = ["code", "reasoning", "chat"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))

    device = "cuda:0" if (cfg.get("use_gpu", False) and torch.cuda.is_available()) else "cpu"
    print(f"device: {device}")

    print("Loading data...")
    t0 = time.time()
    ds = load_all_domains(cfg["data_dir"], domains=DOMAINS)
    layer_indices = ds["layer_indices"].tolist()
    print(f"Loaded {ds['hidden_states'].shape[0]} records, layers={layer_indices}, {time.time()-t0:.1f}s")

    split_assign = make_split(ds, cfg["split"]["train_frac"], cfg["split"]["val_frac"],
                               cfg["split"]["test_frac"], cfg["seed"])
    split_code = torch.tensor([{"train": 0, "val": 1, "test": 2}[s] for s in split_assign])
    train_mask, val_mask, test_mask = split_code == 0, split_code == 1, split_code == 2
    n_unique = len(set(ds["split_key"]))
    print(f"unique prompts: {n_unique}, record split sizes: "
          f"train={train_mask.sum().item()} val={val_mask.sum().item()} test={test_mask.sum().item()}")

    domain_arr = ds["domain"]
    label_soft = ds["label"]
    accepted_binary = ds["accepted"].float()

    os.makedirs(cfg["out_dir"], exist_ok=True)
    results = []
    reliability = {}

    for li, layer in enumerate(layer_indices):
        X = ds["hidden_states"][:, li, :].float()

        mu = X[train_mask].mean(dim=0, keepdim=True)
        sigma = X[train_mask].std(dim=0, keepdim=True).clamp_min(1e-6)
        Xn = (X - mu) / sigma

        X_train, y_train = Xn[train_mask], label_soft[train_mask]
        X_val, y_val_binary = Xn[val_mask], accepted_binary[val_mask]
        X_test = Xn[test_mask]
        y_test_binary = accepted_binary[test_mask].numpy()
        test_domains = [domain_arr[i] for i in range(len(domain_arr)) if test_mask[i]]

        for probe_type in ["linear", "mlp"]:
            t0 = time.time()
            if probe_type == "linear":
                model = LinearProbe(X.shape[1])
            else:
                model = MLPProbe(X.shape[1], cfg["mlp"]["hidden_dim"], cfg["mlp"]["dropout"])

            model, best_val_auroc, n_epochs = train_probe(
                model, X_train, y_train, X_val, y_val_binary,
                lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"],
                batch_size=cfg["train"]["batch_size"], max_epochs=cfg["train"]["max_epochs"],
                patience=cfg["train"]["patience"], device=device, seed=cfg["seed"])

            test_prob = predict_proba(model, X_test, device)
            overall_auroc = auroc(y_test_binary, test_prob)
            overall_ece, bin_stats = expected_calibration_error(y_test_binary, test_prob, cfg["calibration"]["n_bins"])
            reliability[f"{probe_type}_layer{layer}_overall"] = bin_stats

            row = dict(layer=layer, probe_type=probe_type, val_auroc=best_val_auroc,
                       test_auroc=overall_auroc, test_ece=overall_ece, domain="overall",
                       n_epochs=n_epochs, elapsed_sec=time.time() - t0)
            results.append(row)
            print(f"layer={layer} probe={probe_type} val_auroc={best_val_auroc:.4f} "
                  f"test_auroc={overall_auroc:.4f} test_ece={overall_ece:.4f} "
                  f"epochs={n_epochs} ({row['elapsed_sec']:.1f}s)")

            for dom in DOMAINS:
                dom_mask = torch.tensor([d == dom for d in test_domains])
                if dom_mask.sum() == 0:
                    continue
                dom_prob = test_prob[dom_mask.numpy()]
                dom_true = y_test_binary[dom_mask.numpy()]
                dom_auroc = auroc(dom_true, dom_prob)
                dom_ece, _ = expected_calibration_error(dom_true, dom_prob, cfg["calibration"]["n_bins"])
                results.append(dict(layer=layer, probe_type=probe_type, val_auroc=best_val_auroc,
                                     test_auroc=dom_auroc, test_ece=dom_ece, domain=dom,
                                     n_epochs=n_epochs, elapsed_sec=row["elapsed_sec"]))

            torch.save(dict(state_dict=model.state_dict(), mu=mu, sigma=sigma, layer=layer, probe_type=probe_type),
                       os.path.join(cfg["out_dir"], f"probe_{probe_type}_layer{layer}.pt"))

    with open(os.path.join(cfg["out_dir"], "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(cfg["out_dir"], "reliability.json"), "w") as f:
        json.dump(reliability, f, indent=2)
    print(f"\nSaved results.json and reliability.json to {cfg['out_dir']}")


if __name__ == "__main__":
    main()
