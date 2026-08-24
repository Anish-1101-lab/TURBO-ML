"""
E1 cheap-feature decomposition. Config-driven. Pooled-domain training
(mirrors Phase 3), per-domain + pooled reporting, per temperature.
--position selects which Pass-2 hidden-state cache and which Group A
logit-lens columns to source from ("pdec", the default -- matches the
existing Phase 3 probe -- or "ptok", added for the P_tok follow-up: does
P_tok's extra signal over P_dec collapse into cheap features, specifically
q_draft, the same way P_dec's did into target/drafter entropy?). No new
GPU generation needed either way -- both positions' hidden states and
Group A features were already captured during the original Tier 0 Pass 2.

Usage: python scripts/17_tier0_e1.py --config configs/tier0_config.yaml \
    --position pdec --out results/tier0_e1_raw.json
"""
import argparse
import gc
import json
import os
import time

import numpy as np
import torch
import yaml

from probes.tier0_data import GROUP_B_COLS, GROUP_C_COLS, group_a_cols, load_all_domains_pass1, load_hs_layer
from probes.tier0_e1 import (auroc_auprc_ece, bernoulli_loglik, bootstrap_prompt_indices,
                              cheap_feature_matrix, fit_cheap_logreg, fit_offset_probe_pair,
                              fit_probe_pair, reverse_decodability_r2)
from probes.tier0_splits import make_tier0_split
from scipy import stats


def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/tier0_config.yaml")
    ap.add_argument("--out", default="results/tier0_e1_raw.json")
    ap.add_argument("--pass1-dir", default=None)
    ap.add_argument("--pass2-dir", default=None)
    ap.add_argument("--drafter", default="A")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--position", default="pdec", choices=["pdec", "ptok"])
    args = ap.parse_args()

    position = args.position
    hs_key = "hs_dec" if position == "pdec" else "hs_tok"
    position_out = "P_dec" if position == "pdec" else "P_tok"

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
        print(f"\n########## temperature={temperature} ##########")
        df = load_all_domains_pass1(pass1_dir, domains, temperature, args.drafter)
        print(f"loaded {len(df)} rows across domains={domains}")

        domains_and_idxs = {d: set(df.loc[df.domain == d, "split_key"]) for d in domains}
        split_map = make_tier0_split(domains_and_idxs, cfg["split"]["train_frac"],
                                      cfg["split"]["val_frac"], cfg["split"]["test_frac"], seed)
        df["split"] = df["split_key"].map(split_map)
        assert df["split"].isna().sum() == 0
        print(df.groupby(["domain", "split"]).size())

        tr_mask = (df["split"] == "train").to_numpy()
        va_mask = (df["split"] == "val").to_numpy()
        te_mask = (df["split"] == "test").to_numpy()
        y_bin = df["accepted"].to_numpy()
        y_soft = df["label"].to_numpy()
        split_keys_test = df.loc[te_mask, "split_key"].to_numpy()
        domain_test = df.loc[te_mask, "domain"].to_numpy()
        base_rate = float(y_bin[tr_mask].mean())

        for layer in probed_layers:
            t_layer = time.time()
            print(f"\n=== T={temperature} layer={layer} ===")
            hs_parts = []
            for d in domains:
                hs_parts.append(load_hs_layer(pass2_dir, d, temperature, args.drafter, layer)[hs_key])
            hs_dec = np.concatenate(hs_parts)  # variable name kept for diff-minimality; holds whichever position was requested
            assert hs_dec.shape[0] == len(df), (hs_dec.shape, len(df))

            Xh_tr, Xh_va, Xh_te = hs_dec[tr_mask], hs_dec[va_mask], hs_dec[te_mask]
            ytr_soft, yva_bin, yte_bin = y_soft[tr_mask], y_bin[va_mask], y_bin[te_mask]

            # ---- cheap models M_A, M_AB, M_ABC ----
            cheap_models = {}
            cheap_test_probs = {}
            for groups in ("A", "AB", "ABC"):
                Xc, cols = cheap_feature_matrix(df, layer, groups, position)
                Xc_tr, Xc_va, Xc_te = Xc[tr_mask], Xc[va_mask], Xc[te_mask]
                clf, C, val_auroc = fit_cheap_logreg(Xc_tr, y_bin[tr_mask], Xc_va, y_bin[va_mask])
                test_prob = clf.predict_proba(Xc_te)[:, 1]
                cheap_models[groups] = dict(clf=clf, C=C, val_auroc=val_auroc, cols=cols)
                cheap_test_probs[groups] = test_prob
                auroc, auprc, ece = auroc_auprc_ece(yte_bin, test_prob)
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model=f"M_{groups}", metric="auroc", value=auroc))
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model=f"M_{groups}", metric="auprc", value=auprc))
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model=f"M_{groups}", metric="ece", value=ece))
                for dom in domains:
                    m = domain_test == dom
                    if m.sum() > 5 and len(set(yte_bin[m])) > 1:
                        a, ap_, e = auroc_auprc_ece(yte_bin[m], test_prob[m])
                        results.append(dict(experiment="E1", domain=dom, layer=layer, position=position_out,
                                             temperature=temperature, model=f"M_{groups}", metric="auroc", value=a))

            # M0: base rate
            results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                 temperature=temperature, model="M0", metric="auroc", value=0.5))
            results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                 temperature=temperature, model="M0", metric="base_rate", value=base_rate))

            # ---- M_h: hidden state alone ----
            Xh_tr_t, Xh_va_t, Xh_te_t = to_tensor(Xh_tr), to_tensor(Xh_va), to_tensor(Xh_te)
            ysoft_t = to_tensor(ytr_soft)
            yva_t, yte_t = to_tensor(yva_bin), to_tensor(yte_bin)
            m_h = fit_probe_pair(Xh_tr_t, ysoft_t, ytr_soft, Xh_va_t, yva_t, Xh_te_t,
                                  hidden_dim, dropout, train_cfg, args.device, seed)

            # ---- M_full: hidden state concat ABC ----
            Xc_full, cols_full = cheap_feature_matrix(df, layer, "ABC", position)
            Xfull = np.concatenate([hs_dec, Xc_full], axis=1)
            Xf_tr_t, Xf_va_t, Xf_te_t = to_tensor(Xfull[tr_mask]), to_tensor(Xfull[va_mask]), to_tensor(Xfull[te_mask])
            m_full = fit_probe_pair(Xf_tr_t, ysoft_t, ytr_soft, Xf_va_t, yva_t, Xf_te_t,
                                     hidden_dim, dropout, train_cfg, args.device, seed)

            # ---- M_offset: h_l with M_ABC's logit as fixed offset ----
            abc_clf = cheap_models["ABC"]["clf"]
            Xc_abc, _ = cheap_feature_matrix(df, layer, "ABC", position)
            offset_all = abc_clf.decision_function(Xc_abc)
            off_tr, off_va, off_te = to_tensor(offset_all[tr_mask]), to_tensor(offset_all[va_mask]), to_tensor(offset_all[te_mask])
            m_offset = fit_offset_probe_pair(Xh_tr_t, ysoft_t, off_tr, Xh_va_t, yva_t, off_va,
                                              Xh_te_t, off_te, hidden_dim, dropout, train_cfg, args.device, seed)

            for arch in ("linear", "mlp"):
                for name, out in (("M_h", m_h), ("M_full", m_full), ("M_offset", m_offset)):
                    prob = out[arch]["test_prob"]
                    auroc, auprc, ece = auroc_auprc_ece(yte_bin, prob)
                    results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                         temperature=temperature, model=f"{name}_{arch}", metric="auroc", value=auroc))
                    results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                         temperature=temperature, model=f"{name}_{arch}", metric="auprc", value=auprc))
                    results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                         temperature=temperature, model=f"{name}_{arch}", metric="ece", value=ece))
                    for dom in domains:
                        m = domain_test == dom
                        if m.sum() > 5 and len(set(yte_bin[m])) > 1:
                            a, ap_, e = auroc_auprc_ece(yte_bin[m], prob[m])
                            results.append(dict(experiment="E1", domain=dom, layer=layer, position=position_out,
                                                 temperature=temperature, model=f"{name}_{arch}", metric="auroc", value=a))

            # ---- Delta AUROC (M_offset - M_ABC), bootstrap CI over prompts, per arch ----
            for arch in ("linear", "mlp"):
                offset_prob = m_offset[arch]["test_prob"]
                abc_prob = cheap_test_probs["ABC"]
                point_delta = (roc_auc_score_safe(yte_bin, offset_prob) - roc_auc_score_safe(yte_bin, abc_prob))
                boot_deltas = []
                for rows in bootstrap_prompt_indices(split_keys_test, args.n_boot, seed):
                    yb, ob, cb = yte_bin[rows], offset_prob[rows], abc_prob[rows]
                    if len(set(yb)) < 2:
                        continue
                    boot_deltas.append(roc_auc_score_safe(yb, ob) - roc_auc_score_safe(yb, cb))
                boot_deltas = np.array(boot_deltas)
                lo, hi = np.percentile(boot_deltas, [2.5, 97.5]) if len(boot_deltas) else (float("nan"), float("nan"))
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model=f"delta_offset_vs_ABC_{arch}",
                                     metric="delta_auroc", value=float(point_delta), ci_low=float(lo), ci_high=float(hi)))

                # LR test: M_offset (arch) vs M_ABC, df = # trainable params in offset model
                ll_full = bernoulli_loglik(yte_bin, offset_prob)
                ll_null = bernoulli_loglik(yte_bin, abc_prob)
                n_params = sum(p.numel() for p in m_offset[arch]["model"].parameters())
                lr_stat = 2 * (ll_full - ll_null)
                p_value = float(stats.chi2.sf(max(lr_stat, 0.0), df=n_params))
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model=f"LRtest_offset_vs_ABC_{arch}",
                                     metric="lr_stat", value=float(lr_stat), extra_df=n_params, extra_p=p_value))

            # ---- reverse decodability R^2: hs (this position) -> a few key cheap features ----
            for feat in ("logp_draft", "entropy", "q_draft", "q_entropy"):
                if feat in ("logp_draft", "entropy"):
                    col = f"{position}_L{layer}_{feat}"
                else:
                    col = feat
                ftr = df.loc[tr_mask, col].to_numpy(dtype=np.float64)
                fte = df.loc[te_mask, col].to_numpy(dtype=np.float64)
                r2 = reverse_decodability_r2(hs_dec[tr_mask], hs_dec[te_mask], ftr, fte)
                results.append(dict(experiment="E1", domain="pooled", layer=layer, position=position_out,
                                     temperature=temperature, model="reverse_decode", metric=f"r2_{feat}", value=float(r2)))

            del hs_dec, Xfull
            gc.collect()
            print(f"  layer {layer} done in {time.time()-t_layer:.0f}s")

            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)

    print(f"\nTotal E1 wall clock: {time.time()-t0:.0f}s, {len(results)} result rows -> {args.out}")


def roc_auc_score_safe(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")


if __name__ == "__main__":
    main()
