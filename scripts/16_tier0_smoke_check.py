"""
Tier 0 smoke-test sanity checks -- run after a small scripts/15_tier0_generate.py
invocation, BEFORE trusting the pipeline enough to scale to full volume.

Checks:
  1. Shape/dtype sanity on the Pass 1 rows just written.
  2. Spec's own required check: at T=0 (greedy), P_tok, near-final probed
     layer (27), an independent readout (logit lens through layer 27's
     residual at P_tok) should predict the TRUE accept/reject outcome
     (decided from the true final layer at P_dec) with AUROC approaching
     1.0. If it doesn't, something in the label/position pipeline is wrong
     -- per the spec, investigate before generating anything at scale.
  3. Preliminary (NOT a real E1 run -- tiny n, no train/test split) check
     of whether cheap Group A features at P_dec/layer 24 already separate
     accepted from rejected well, as an early signal for the cheap-feature
     hypothesis Step 0 raised.

Usage: python scripts/16_tier0_smoke_check.py --pass1-dir data/tier0_smoke/pass1
"""
import argparse
import glob
import os

import torch
from sklearn.metrics import roc_auc_score


def load_rows(pass1_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(pass1_dir, "*.pt"))):
        rows.extend(torch.load(f, weights_only=False))
    return rows


def auroc_univariate(rows, feature_key, label_key="accepted", higher_is_better=True):
    y = [int(r[label_key]) for r in rows]
    x = [r[feature_key] for r in rows]
    if len(set(y)) < 2:
        return None, len(rows)
    score = x if higher_is_better else [-v for v in x]
    return roc_auc_score(y, score), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass1-dir", default="data/tier0_smoke/pass1")
    args = ap.parse_args()

    rows = load_rows(args.pass1_dir)
    print(f"loaded {len(rows)} Pass-1 rows from {args.pass1_dir}")
    if not rows:
        print("NO ROWS FOUND -- generation did not produce data, stopping.")
        return

    r0 = rows[0]
    n_cols = len(r0)
    print(f"columns per row: {n_cols}")
    print(f"sample keys: {sorted(r0.keys())[:12]} ...")
    domains = sorted(set(r["domain"] for r in rows))
    temps = sorted(set(r["temperature"] for r in rows))
    print(f"domains present: {domains}, temperatures present: {temps}")

    for dom in domains:
        for T in temps:
            sub = [r for r in rows if r["domain"] == dom and r["temperature"] == T]
            if not sub:
                continue
            acc_rate = sum(r["accepted"] for r in sub) / len(sub)
            print(f"  [{dom} T={T}] n={len(sub)} accept_rate={acc_rate:.3f}")

    print("\n=== Check 2: P_tok/layer-27 lens vs true accept/reject (T=0 only) ===")
    t0_rows = [r for r in rows if r["temperature"] == 0.0]
    if t0_rows:
        auroc, n = auroc_univariate(t0_rows, "ptok_L27_logp_draft")
        print(f"AUROC(ptok_L27_logp_draft -> accepted), T=0, n={n}: "
              f"{auroc if auroc is not None else 'undefined (single class)'}")
        if auroc is not None and auroc < 0.9:
            print("*** WARNING: below the spec's ~1.0 expectation -- investigate "
                  "position indexing / label generation before scaling up. ***")
    else:
        print("no T=0 rows in this smoke run -- skipped.")

    print("\n=== Check 3 (preliminary, NOT full E1): P_dec/layer-24 cheap features vs accepted ===")
    for T in temps:
        sub = [r for r in rows if r["temperature"] == T]
        if len(sub) < 10:
            continue
        for feat, hib in [("pdec_L24_logp_draft", True), ("pdec_L24_entropy", False),
                           ("pdec_L24_p_top1", True), ("q_draft", True)]:
            auroc, n = auroc_univariate(sub, feat, higher_is_better=hib)
            print(f"  T={T} AUROC({feat} -> accepted), n={n}: "
                  f"{auroc if auroc is not None else 'undefined (single class)'}")


if __name__ == "__main__":
    main()
