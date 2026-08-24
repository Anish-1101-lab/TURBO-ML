"""
E2 two-position decomposition: fit a probe (linear + MLP, pooled-domain
training, per-domain reporting -- same convention as E1) at BOTH P_dec and
P_tok, every probed layer, both temperatures, all domains. Reports the
AUROC(P_tok) - AUROC(P_dec) gap -- the isolated draft-agreement signal.

Usage: python scripts/18_tier0_e2.py --config configs/tier0_config.yaml \
    --out results/tier0_e2_raw.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import yaml

from probes.tier0_data import load_all_domains_pass1, load_hs_layer
from probes.tier0_e1 import auroc_auprc_ece, fit_probe_pair
from probes.tier0_splits import make_tier0_split


def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier0_config.yaml")
    ap.add_argument("--out", default="results/tier0_e2_raw.json")
    ap.add_argument("--pass1-dir", default=None)
    ap.add_argument("--pass2-dir", default=None)
    ap.add_argument("--drafter", default="A")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    pass1_dir = args.pass1_dir or cfg["paths"]["pass1_dir"]
    pass2_dir = args.pass2_dir or cfg["paths"]["pass2_dir"]
    domains = list(cfg["domains"].keys())
    temperatures = cfg["temperatures"]
    probed_layers = cfg["probed_layers"]
    seed = cfg["seed"]
    train_cfg = dict(lr=0.001, weight_decay=0.0001, batch_size=1024, max_epochs=100, patience=10)
    hidden_dim, dropout = 256, 0.1

    results = []
    t0 = time.time()

    for temperature in temperatures:
        print(f"\n########## E2 temperature={temperature} ##########")
        df = load_all_domains_pass1(pass1_dir, domains, temperature, args.drafter)
        domains_and_idxs = {d: set(df.loc[df.domain == d, "split_key"]) for d in domains}
        split_map = make_tier0_split(domains_and_idxs, cfg["split"]["train_frac"],
                                      cfg["split"]["val_frac"], cfg["split"]["test_frac"], seed)
        df["split"] = df["split_key"].map(split_map)

        tr_mask = (df["split"] == "train").to_numpy()
        va_mask = (df["split"] == "val").to_numpy()
        te_mask = (df["split"] == "test").to_numpy()
        y_bin = df["accepted"].to_numpy()
        y_soft = df["label"].to_numpy()
        domain_test = df.loc[te_mask, "domain"].to_numpy()

        for layer in probed_layers:
            t_layer = time.time()
            hs_dec_parts, hs_tok_parts = [], []
            for d in domains:
                hs = load_hs_layer(pass2_dir, d, temperature, args.drafter, layer)
                hs_dec_parts.append(hs["hs_dec"])
                hs_tok_parts.append(hs["hs_tok"])
            hs_dec, hs_tok = np.concatenate(hs_dec_parts), np.concatenate(hs_tok_parts)

            for pos_name, hs_full in (("P_dec", hs_dec), ("P_tok", hs_tok)):
                Xtr, Xva, Xte = to_tensor(hs_full[tr_mask]), to_tensor(hs_full[va_mask]), to_tensor(hs_full[te_mask])
                ysoft_t, yva_t, yte_bin = to_tensor(y_soft[tr_mask]), to_tensor(y_bin[va_mask]), y_bin[te_mask]
                out = fit_probe_pair(Xtr, ysoft_t, y_soft[tr_mask], Xva, yva_t, Xte,
                                      hidden_dim, dropout, train_cfg, args.device, seed)
                for arch in ("linear", "mlp"):
                    prob = out[arch]["test_prob"]
                    auroc, auprc, ece = auroc_auprc_ece(yte_bin, prob)
                    results.append(dict(experiment="E2", domain="pooled", layer=layer, position=pos_name,
                                         temperature=temperature, model=f"probe_{arch}", metric="auroc", value=auroc))
                    for dom in domains:
                        m = domain_test == dom
                        if m.sum() > 5 and len(set(yte_bin[m])) > 1:
                            a, _, _ = auroc_auprc_ece(yte_bin[m], prob[m])
                            results.append(dict(experiment="E2", domain=dom, layer=layer, position=pos_name,
                                                 temperature=temperature, model=f"probe_{arch}", metric="auroc", value=a))
            del hs_dec, hs_tok
            gc.collect()
            print(f"  layer {layer} done in {time.time()-t_layer:.0f}s")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)

    print(f"\nTotal E2 wall clock: {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
