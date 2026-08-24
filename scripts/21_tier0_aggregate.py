"""
Aggregate E1/E2/E3 raw JSON results into the tidy long-format deliverable
(results/tier0_results.parquet) and write results/tier0_verdict.md applying
the spec's exact E1 decision rule.

Usage: python scripts/21_tier0_aggregate.py --e1 results/tier0_e1_raw.json \
    --e2 results/tier0_e2_raw.json --e3 results/tier0_e3_raw.json \
    --out-parquet results/tier0_results.parquet --out-verdict results/tier0_verdict.md
"""
import argparse
import json
import os

import pandas as pd


def load_json(path):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping")
        return []
    return json.load(open(path))


def e1_verdict(df_e1, peak_layer_mlp, temperature):
    sub = df_e1[(df_e1.temperature == temperature) & (df_e1.domain == "pooled") &
                (df_e1.layer == peak_layer_mlp) & (df_e1.model == "delta_offset_vs_ABC_mlp") &
                (df_e1.metric == "delta_auroc")]
    if sub.empty:
        return "NO DATA", None, None, None
    row = sub.iloc[0]
    delta, lo, hi = row["value"], row.get("ci_low"), row.get("ci_high")
    if delta < 0.03 and (hi is not None and hi < 0.05):
        verdict = "NO_DISTINCT_FEATURE"
    elif delta > 0.08 and (lo is not None and lo > 0.03):
        verdict = "RESIDUAL_STRUCTURE"
    else:
        verdict = "INCONCLUSIVE"
    return verdict, delta, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e1", default="results/tier0_e1_raw.json")
    ap.add_argument("--e2", default="results/tier0_e2_raw.json")
    ap.add_argument("--e3", default="results/tier0_e3_raw.json")
    ap.add_argument("--out-parquet", default="results/tier0_results.parquet")
    ap.add_argument("--out-verdict", default="results/tier0_verdict.md")
    args = ap.parse_args()

    rows = load_json(args.e1) + load_json(args.e2) + load_json(args.e3)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_parquet), exist_ok=True)
    df.to_parquet(args.out_parquet, index=False)
    print(f"wrote {len(df)} tidy rows -> {args.out_parquet}")

    e1 = df[df.experiment == "E1"]
    e2 = df[df.experiment == "E2"]
    e3 = df[df.experiment == "E3"]

    lines = ["# Tier 0 verdict: does the acceptance probe read a distinct verification feature?\n"]

    # --- E1 ---
    lines.append("## E1 -- cheap-feature decomposition\n")
    for temperature in sorted(e1.temperature.dropna().unique()):
        mh = e1[(e1.temperature == temperature) & (e1.domain == "pooled") &
                (e1.model == "M_h_mlp") & (e1.metric == "auroc")]
        if mh.empty:
            continue
        peak_row = mh.loc[mh["value"].idxmax()]
        peak_layer = int(peak_row["layer"])
        peak_auroc = peak_row["value"]
        verdict, delta, lo, hi = e1_verdict(e1, peak_layer, temperature)
        m_abc = e1[(e1.temperature == temperature) & (e1.domain == "pooled") &
                    (e1.layer == peak_layer) & (e1.model == "M_ABC") & (e1.metric == "auroc")]
        abc_auroc = m_abc["value"].iloc[0] if not m_abc.empty else float("nan")
        lines.append(f"**T={temperature}**: peak M_h(MLP) AUROC = {peak_auroc:.4f} at layer {peak_layer}. "
                      f"M_ABC (cheap-only) AUROC at that layer = {abc_auroc:.4f}. "
                      f"ΔAUROC(M_offset - M_ABC) = {delta:.4f} (95% CI [{lo:.4f}, {hi:.4f}]) → **{verdict}**.\n")

    # --- E2 ---
    lines.append("\n## E2 -- two-position decomposition\n")
    for temperature in sorted(e2.temperature.dropna().unique()):
        for pos in ("P_dec", "P_tok"):
            sub = e2[(e2.temperature == temperature) & (e2.domain == "pooled") &
                     (e2.position == pos) & (e2.model == "probe_mlp") & (e2.metric == "auroc")]
            if sub.empty:
                continue
            peak = sub.loc[sub["value"].idxmax()]
            lines.append(f"T={temperature} {pos}: peak AUROC = {peak['value']:.4f} at layer {int(peak['layer'])}")
    lines.append("\nExisting Phase 3 pipeline reads **P_dec** (confirmed by code inspection, Step 0) --"
                  " see NOTES.md for the position-math trace.\n")

    # --- E3 ---
    lines.append("\n## E3 -- drafter-swap transfer\n")
    tr = e3[(e3.metric == "transfer_ratio")]
    if not tr.empty:
        summary = tr.groupby(["domain", "position", "model"])["value"].mean().reset_index()
        header = f"| {'domain':10s} | {'position':8s} | {'model':30s} | {'mean_transfer_ratio':>20s} |"
        sep = f"|{'-'*12}|{'-'*10}|{'-'*32}|{'-'*22}|"
        rows_md = [header, sep]
        for _, r in summary.iterrows():
            rows_md.append(f"| {r['domain']:10s} | {r['position']:8s} | {r['model']:30s} | {r['value']:20.4f} |")
        lines.append("\n".join(rows_md))
    else:
        lines.append("(no E3 transfer_ratio rows found)")

    os.makedirs(os.path.dirname(args.out_verdict), exist_ok=True)
    with open(args.out_verdict, "w") as f:
        f.write("\n".join(str(l) for l in lines))
    print(f"wrote verdict -> {args.out_verdict}")


if __name__ == "__main__":
    main()
