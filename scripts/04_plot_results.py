"""
Phase 3 plots: AUROC-vs-depth, calibration(ECE)-vs-depth, and reliability
diagrams for a few representative layers. Reads results.json/reliability.json
produced by scripts/03_train_probes.py.

Usage: python scripts/04_plot_results.py --config configs/probe_config.yaml
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

DOMAIN_COLORS = {"overall": "black", "code": "#2b8a3e", "reasoning": "#1c7ed6", "chat": "#e8590c"}
PROBE_STYLES = {"linear": "--", "mlp": "-"}


def load_results(out_dir):
    with open(os.path.join(out_dir, "results.json")) as f:
        results = json.load(f)
    with open(os.path.join(out_dir, "reliability.json")) as f:
        reliability = json.load(f)
    return results, reliability


def plot_metric_vs_depth(results, metric, ylabel, out_path):
    layers = sorted(set(r["layer"] for r in results))
    fig, ax = plt.subplots(figsize=(7, 5))
    for probe_type in ["linear", "mlp"]:
        for domain in ["overall", "code", "reasoning", "chat"]:
            ys = [next(r[metric] for r in results if r["layer"] == l and r["probe_type"] == probe_type and r["domain"] == domain)
                  for l in layers]
            ax.plot(layers, ys, marker="o", markersize=4,
                    linestyle=PROBE_STYLES[probe_type], color=DOMAIN_COLORS[domain],
                    label=f"{probe_type}/{domain}",
                    linewidth=2.5 if domain == "overall" else 1.4,
                    alpha=1.0 if domain == "overall" else 0.75)
    ax.set_xlabel("target layer (of 28)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs. depth")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_reliability(reliability, out_path, layers_to_show):
    fig, axes = plt.subplots(2, len(layers_to_show), figsize=(4 * len(layers_to_show), 8), sharex=True, sharey=True)
    for col, layer in enumerate(layers_to_show):
        for row, probe_type in enumerate(["linear", "mlp"]):
            ax = axes[row, col]
            bins = reliability[f"{probe_type}_layer{layer}_overall"]
            confs = [b["conf"] for b in bins if b["conf"] is not None]
            accs = [b["acc"] for b in bins if b["acc"] is not None]
            counts = [b["count"] for b in bins if b["count"] > 0]
            ax.plot([0, 1], [0, 1], linestyle=":", color="gray", linewidth=1)
            ax.scatter(confs, accs, s=[20 + c / 20 for c in counts], color="#1c7ed6", alpha=0.8)
            ax.plot(confs, accs, color="#1c7ed6", alpha=0.5)
            ax.set_title(f"layer {layer}, {probe_type}")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            if row == 1:
                ax.set_xlabel("predicted prob.")
            if col == 0:
                ax.set_ylabel("empirical accept rate")
            ax.grid(alpha=0.3)
    fig.suptitle("Reliability diagrams (overall, test set)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/probe_config.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_dir = cfg["out_dir"]
    plot_dir = os.path.join(out_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    results, reliability = load_results(out_dir)

    plot_metric_vs_depth(results, "test_auroc", "AUROC", os.path.join(plot_dir, "auroc_vs_depth.png"))
    plot_metric_vs_depth(results, "test_ece", "ECE", os.path.join(plot_dir, "ece_vs_depth.png"))

    layers = sorted(set(r["layer"] for r in results))
    layers_to_show = [layers[0], layers[len(layers) // 2], layers[-1]]
    plot_reliability(reliability, os.path.join(plot_dir, "reliability_diagrams.png"), layers_to_show)

    print(f"Saved plots to {plot_dir}")


if __name__ == "__main__":
    main()
