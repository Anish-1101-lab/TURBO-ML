"""
Generate the three Tier 0 deliverable figures from results/tier0_results.parquet.
No notebook state -- everything regenerable by this one script.

Usage: python scripts/22_tier0_plots.py --parquet results/tier0_results.parquet --out-dir figs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def e1_decomposition_fig(df, out_path):
    e1 = df[(df.experiment == "E1") & (df.metric == "auroc")]
    temps = sorted(e1.temperature.dropna().unique())
    domains = ["pooled", "code", "reasoning", "chat"]
    model_order = ["M0", "M_A", "M_AB", "M_ABC", "M_h_mlp", "M_full_mlp", "M_offset_mlp"]
    labels = {"M0": "M0 (base rate)", "M_A": "M_A", "M_AB": "M_AB", "M_ABC": "M_ABC (cheap)",
              "M_h_mlp": "M_h (hidden, MLP)", "M_full_mlp": "M_full (hidden+cheap)",
              "M_offset_mlp": "M_offset (hidden | cheap)"}

    fig, axes = plt.subplots(len(temps), len(domains), figsize=(20, 5 * len(temps)), squeeze=False)
    for i, T in enumerate(temps):
        for j, dom in enumerate(domains):
            ax = axes[i][j]
            sub = e1[(e1.temperature == T) & (e1.domain == dom)]
            for m in model_order:
                s = sub[sub.model == m].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(s.layer, s.value, marker="o", markersize=3, label=labels[m])
            ax.set_title(f"{dom}, T={T}")
            ax.set_xlabel("layer")
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.45, 1.0)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    fig.suptitle("E1: cheap-feature decomposition, AUROC by layer")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def e2_position_fig(df, out_path):
    e2 = df[(df.experiment == "E2") & (df.metric == "auroc") & (df.model == "probe_mlp")]
    temps = sorted(e2.temperature.dropna().unique())
    domains = ["pooled", "code", "reasoning", "chat"]

    fig, axes = plt.subplots(len(temps), len(domains), figsize=(20, 5 * len(temps)), squeeze=False)
    for i, T in enumerate(temps):
        for j, dom in enumerate(domains):
            ax = axes[i][j]
            sub = e2[(e2.temperature == T) & (e2.domain == dom)]
            for pos, style in (("P_dec", "-o"), ("P_tok", "-s")):
                s = sub[sub.position == pos].sort_values("layer")
                if s.empty:
                    continue
                ax.plot(s.layer, s.value, style, markersize=4, label=pos)
            ax.set_title(f"{dom}, T={T}")
            ax.set_xlabel("layer")
            ax.set_ylabel("AUROC")
            ax.set_ylim(0.45, 1.05)
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle("E2: P_dec vs P_tok, AUROC by layer")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def e3_transfer_fig(df, out_path):
    e3 = df[(df.experiment == "E3") & (df.metric == "transfer_ratio")]
    if e3.empty:
        print("no E3 transfer_ratio rows -- skipping figure")
        return
    domains = sorted(e3.domain.unique())
    positions = sorted(e3.position.unique())

    fig, axes = plt.subplots(1, len(domains), figsize=(6 * len(domains), 5), squeeze=False)
    axes = axes[0]
    for j, dom in enumerate(domains):
        ax = axes[j]
        sub = e3[e3.domain == dom]
        for pos, style in (("P_dec", "-o"), ("P_tok", "-s")):
            for model in sorted(sub.model.unique()):
                s = sub[(sub.position == pos) & (sub.model == model)].sort_values("layer")
                if s.empty or "mlp" not in model:
                    continue
                ax.plot(s.layer, s.value, style, markersize=4, label=f"{pos} {model.replace('transfer_ratio_', '')}")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1)
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(dom)
        ax.set_xlabel("layer")
        ax.set_ylabel("transfer_ratio")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("E3: transfer_ratio by layer (drafter A -> B, frozen probe)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="results/tier0_results.parquet")
    ap.add_argument("--out-dir", default="figs")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    os.makedirs(args.out_dir, exist_ok=True)
    e1_decomposition_fig(df, os.path.join(args.out_dir, "e1_decomposition.pdf"))
    e2_position_fig(df, os.path.join(args.out_dir, "e2_position.pdf"))
    e3_transfer_fig(df, os.path.join(args.out_dir, "e3_transfer.pdf"))
    print(f"figures written to {args.out_dir}/")


if __name__ == "__main__":
    main()
