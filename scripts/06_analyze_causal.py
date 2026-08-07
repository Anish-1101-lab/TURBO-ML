"""
Phase 4 analysis: does ablating the probe's layer-24 direction shift the
target's REAL p(x) in a way correlated with the probe's own (pre-ablation)
prediction, more than a random direction of matched ablation mechanics does?

Usage: python scripts/06_analyze_causal.py --records analysis/phase4/causal_records.json
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="analysis/phase4/causal_records.json")
    parser.add_argument("--out", default="analysis/phase4")
    args = parser.parse_args()

    recs = json.load(open(args.records))
    n = len(recs)
    probe_score = np.array([r["probe_score"] for r in recs])
    p_clean = np.array([r["p_clean"] for r in recs])
    p_probe_abl = np.array([r["p_probe_ablated"] for r in recs])
    p_rand_abl = np.array([r["p_random_ablated"] for r in recs])
    domain = np.array([r["domain"] for r in recs])

    delta_probe = p_probe_abl - p_clean
    delta_rand = p_rand_abl - p_clean

    results = {"n_records": n}

    r_probe, pval_probe = pearsonr(probe_score, delta_probe)
    r_rand, pval_rand = pearsonr(probe_score, delta_rand)
    rho_probe, rho_pval_probe = spearmanr(probe_score, delta_probe)
    rho_rand, rho_pval_rand = spearmanr(probe_score, delta_rand)

    results["overall"] = dict(
        pearson_probe_ablation=dict(r=float(r_probe), p=float(pval_probe)),
        pearson_random_ablation=dict(r=float(r_rand), p=float(pval_rand)),
        spearman_probe_ablation=dict(rho=float(rho_probe), p=float(rho_pval_probe)),
        spearman_random_ablation=dict(rho=float(rho_rand), p=float(rho_pval_rand)),
        mean_abs_delta_probe=float(np.mean(np.abs(delta_probe))),
        mean_abs_delta_random=float(np.mean(np.abs(delta_rand))),
        mean_delta_probe=float(np.mean(delta_probe)),
        mean_delta_random=float(np.mean(delta_rand)),
    )

    print(f"n={n}")
    print(f"Pearson r(probe_score, delta_probe_ablation)  = {r_probe:+.4f} (p={pval_probe:.2e})")
    print(f"Pearson r(probe_score, delta_random_ablation) = {r_rand:+.4f} (p={pval_rand:.2e})")
    print(f"Spearman rho(probe_score, delta_probe_ablation)  = {rho_probe:+.4f} (p={rho_pval_probe:.2e})")
    print(f"Spearman rho(probe_score, delta_random_ablation) = {rho_rand:+.4f} (p={rho_pval_rand:.2e})")
    print(f"mean |delta| probe-ablation:  {results['overall']['mean_abs_delta_probe']:.5f}")
    print(f"mean |delta| random-ablation: {results['overall']['mean_abs_delta_random']:.5f}")

    results["by_domain"] = {}
    for dom in ["code", "reasoning", "chat"]:
        mask = domain == dom
        r_p, p_p = pearsonr(probe_score[mask], delta_probe[mask])
        r_r, p_r = pearsonr(probe_score[mask], delta_rand[mask])
        results["by_domain"][dom] = dict(
            n=int(mask.sum()),
            pearson_probe_ablation=dict(r=float(r_p), p=float(p_p)),
            pearson_random_ablation=dict(r=float(r_r), p=float(p_r)),
            mean_abs_delta_probe=float(np.mean(np.abs(delta_probe[mask]))),
            mean_abs_delta_random=float(np.mean(np.abs(delta_rand[mask]))),
        )
        print(f"[{dom}] n={mask.sum()} r_probe={r_p:+.4f} (p={p_p:.2e}) r_random={r_r:+.4f} (p={p_r:.2e})")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "causal_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, delta, title, r, pv in [
        (axes[0], delta_probe, "probe direction", r_probe, pval_probe),
        (axes[1], delta_rand, "random direction (control)", r_rand, pval_rand),
    ]:
        ax.scatter(probe_score, delta, s=4, alpha=0.15, color="#1c7ed6")
        ax.axhline(0, color="gray", linewidth=1, linestyle=":")
        z = np.polyfit(probe_score, delta, 1)
        xs = np.linspace(0, 1, 100)
        ax.plot(xs, np.polyval(z, xs), color="#e8590c", linewidth=2)
        ax.set_title(f"{title}\nPearson r={r:+.3f} (p={pv:.1e})")
        ax.set_xlabel("probe_score (pre-ablation prediction)")
    axes[0].set_ylabel("delta p(x) = p_ablated - p_clean")
    fig.suptitle("Phase 4: does ablating this direction shift real p(x)?")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "causal_scatter.png"), dpi=150)
    plt.close(fig)

    print(f"\nSaved causal_analysis.json and causal_scatter.png to {args.out}")


if __name__ == "__main__":
    main()
