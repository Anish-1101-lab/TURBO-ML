"""
E3 drafter-swap transfer. T=0.7 only (matches drafters B/C's generation --
see scripts/19_tier0_generate_transfer.py's judgment call). Per domain,
layer, position: fit a probe on drafter A's train split (frozen), evaluate
on A's own test (same-drafter baseline), on B/C's test (frozen transfer),
against an upper bound (retrained from scratch on B/C's own train) and a
lower bound (probe trained on A's train with shuffled labels).

Usage: python scripts/20_tier0_e3.py --config configs/tier0_config.yaml \
    --drafters B C --out results/tier0_e3_raw.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import yaml

from probes.tier0_data import load_pass1_df, load_hs_layer
from probes.tier0_e1 import fit_probe_pair, auroc_auprc_ece
from probes.tier0_splits import assign_split


def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def add_split(df, split_frac, seed):
    df["split"] = df["split_key"].apply(
        lambda k: assign_split(k, split_frac["train_frac"], split_frac["val_frac"], split_frac["test_frac"], seed))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier0_config.yaml")
    ap.add_argument("--drafters", nargs="+", default=["B", "C"])
    ap.add_argument("--transfer-dir", default="data/tier0_transfer")
    ap.add_argument("--out", default="results/tier0_e3_raw.json")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    domains = list(cfg["domains"].keys())
    probed_layers = cfg["probed_layers"]
    seed = cfg["seed"]
    split_frac = cfg["split"]
    train_cfg = dict(lr=0.001, weight_decay=0.0001, batch_size=1024, max_epochs=100, patience=10)
    hidden_dim, dropout = 256, 0.1

    a_pass1 = cfg["paths"]["pass1_dir"]
    a_pass2 = cfg["paths"]["pass2_dir"]
    t_pass1 = os.path.join(args.transfer_dir, "pass1")
    t_pass2 = os.path.join(args.transfer_dir, "pass2")

    drafters_available = []
    for d in args.drafters:
        if os.path.exists(os.path.join(t_pass1, f"manifest_{d}.json")):
            drafters_available.append(d)
        else:
            print(f"WARNING: no manifest for drafter {d} in {t_pass1} -- skipping (dropped or not yet generated)")

    results = []
    t0 = time.time()

    for domain in domains:
        df_a = load_pass1_df(a_pass1, domain, args.temperature, "A")
        df_a = add_split(df_a, split_frac, seed)
        yA_bin = df_a["accepted"].to_numpy()
        yA_soft = df_a["label"].to_numpy()
        trA, vaA, teA = [(df_a["split"] == s).to_numpy() for s in ("train", "val", "test")]
        base_seed = seed

        drafter_dfs = {}
        for d in drafters_available:
            ddf = load_pass1_df(t_pass1, domain, args.temperature, d)
            ddf = add_split(ddf, split_frac, seed)
            drafter_dfs[d] = ddf

        for layer in probed_layers:
            t_layer = time.time()
            for pos_name, key in (("P_dec", "hs_dec"), ("P_tok", "hs_tok")):
                hsA = load_hs_layer(a_pass2, domain, args.temperature, "A", layer)[key]
                XA_tr, XA_va, XA_te = to_tensor(hsA[trA]), to_tensor(hsA[vaA]), to_tensor(hsA[teA])
                yA_soft_t, yA_va_t = to_tensor(yA_soft[trA]), to_tensor(yA_bin[vaA])
                yA_te_bin = yA_bin[teA]

                # Frozen probe trained on A
                fitA = fit_probe_pair(XA_tr, yA_soft_t, yA_soft[trA], XA_va, yA_va_t, XA_te,
                                       hidden_dim, dropout, train_cfg, args.device, base_seed)

                # Shuffled-label lower bound (same A train data, permuted soft labels)
                rng = np.random.RandomState(base_seed)
                y_shuf = yA_soft[trA].copy()
                rng.shuffle(y_shuf)
                fitShuf = fit_probe_pair(XA_tr, to_tensor(y_shuf), y_shuf, XA_va, yA_va_t, XA_te,
                                          hidden_dim, dropout, train_cfg, args.device, base_seed)

                for arch in ("linear", "mlp"):
                    same_drafter_auroc, _, _ = auroc_auprc_ece(yA_te_bin, fitA[arch]["test_prob"])
                    lower_bound_auroc, _, _ = auroc_auprc_ece(yA_te_bin, fitShuf[arch]["test_prob"])
                    results.append(dict(experiment="E3", domain=domain, layer=layer, position=pos_name,
                                         temperature=args.temperature, model=f"A_to_A_baseline_{arch}",
                                         metric="auroc", value=same_drafter_auroc))
                    results.append(dict(experiment="E3", domain=domain, layer=layer, position=pos_name,
                                         temperature=args.temperature, model=f"shuffled_lower_bound_{arch}",
                                         metric="auroc", value=lower_bound_auroc))

                    for d in drafters_available:
                        ddf = drafter_dfs[d]
                        teD = (ddf["split"] == "test").to_numpy()
                        trD = (ddf["split"] == "train").to_numpy()
                        vaD = (ddf["split"] == "val").to_numpy()
                        if teD.sum() < 20 or trD.sum() < 50:
                            print(f"  skip {domain} L{layer} {pos_name} drafter={d}: too few rows "
                                  f"(train={trD.sum()}, test={teD.sum()})")
                            continue
                        hsD = load_hs_layer(t_pass2, domain, args.temperature, d, layer)[key]
                        yD_bin = ddf["accepted"].to_numpy()
                        yD_soft = ddf["label"].to_numpy()

                        # Frozen transfer: A's fitted model applied to D's test rows
                        from probes.metrics import predict_proba
                        XD_te = to_tensor(hsD[teD])
                        transfer_prob = predict_proba(fitA[arch]["model"], XD_te, args.device)
                        transfer_auroc, _, _ = auroc_auprc_ece(yD_bin[teD], transfer_prob)

                        # Upper bound: retrained from scratch on D's own train/val/test
                        XD_tr, XD_va = to_tensor(hsD[trD]), to_tensor(hsD[vaD])
                        yD_va_t = to_tensor(yD_bin[vaD])
                        fitD = fit_probe_pair(XD_tr, to_tensor(yD_soft[trD]), yD_soft[trD], XD_va, yD_va_t, XD_te,
                                               hidden_dim, dropout, train_cfg, args.device, base_seed)
                        retrained_auroc, _, _ = auroc_auprc_ece(yD_bin[teD], fitD[arch]["test_prob"])

                        denom = retrained_auroc - 0.5
                        transfer_ratio = (transfer_auroc - 0.5) / denom if abs(denom) > 1e-6 else float("nan")

                        results.append(dict(experiment="E3", domain=domain, layer=layer, position=pos_name,
                                             temperature=args.temperature, model=f"frozen_A_to_{d}_{arch}",
                                             metric="auroc", value=transfer_auroc))
                        results.append(dict(experiment="E3", domain=domain, layer=layer, position=pos_name,
                                             temperature=args.temperature, model=f"retrained_{d}_upper_bound_{arch}",
                                             metric="auroc", value=retrained_auroc))
                        results.append(dict(experiment="E3", domain=domain, layer=layer, position=pos_name,
                                             temperature=args.temperature, model=f"transfer_ratio_A_to_{d}_{arch}",
                                             metric="transfer_ratio", value=float(transfer_ratio)))
            gc.collect()
            print(f"  {domain} layer {layer} done in {time.time()-t_layer:.0f}s")
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)

    print(f"\nTotal E3 wall clock: {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
