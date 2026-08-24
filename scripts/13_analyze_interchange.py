"""
Phase 4c analysis: aggregate the interchange-patching pairs from
scripts/12_interchange_patching.py.

For each pair, two swap directions (high-into-low, low-into-high) x two
patch directions (probe, random) = 4 shift values, all defined so that
POSITIVE = shift in the direction the causal hypothesis predicts (donor's
outcome pulls the recipient's p(x) toward it). A combined per-pair,
per-patch-direction score (mean of the two swap directions) is used for
the paired statistical test, since that is what makes probe vs. random a
genuinely paired comparison (same pair, same swap directions, only the
patched direction differs).

Usage: python scripts/13_analyze_interchange.py --pairs analysis/phase4c/interchange_pairs.json
"""
import argparse
import json
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="analysis/phase4c/interchange_pairs.json")
    parser.add_argument("--out", default="analysis/phase4c")
    args = parser.parse_args()

    pairs = json.load(open(args.pairs))
    n = len(pairs)
    print(f"n pairs = {n}")

    def arr(key):
        return np.array([p[key] for p in pairs])

    results = dict(n_pairs=n)

    for dir_name in ["probe", "random"]:
        h2l = arr(f"causal_shift_high_into_low_{dir_name}")
        l2h = arr(f"causal_shift_low_into_high_{dir_name}")
        combined = (h2l + l2h) / 2
        results[dir_name] = dict(
            high_into_low=dict(mean=float(h2l.mean()), median=float(np.median(h2l)),
                                frac_correct_direction=float((h2l > 0).mean())),
            low_into_high=dict(mean=float(l2h.mean()), median=float(np.median(l2h)),
                                frac_correct_direction=float((l2h > 0).mean())),
            combined=dict(mean=float(combined.mean()), median=float(np.median(combined)),
                          frac_correct_direction=float((combined > 0).mean())),
        )

    probe_combined = (arr("causal_shift_high_into_low_probe") + arr("causal_shift_low_into_high_probe")) / 2
    random_combined = (arr("causal_shift_high_into_low_random") + arr("causal_shift_low_into_high_random")) / 2

    w_stat, w_p = wilcoxon(probe_combined, random_combined)
    t_stat, t_p = ttest_rel(probe_combined, random_combined)
    results["paired_test_probe_vs_random"] = dict(
        wilcoxon_stat=float(w_stat), wilcoxon_p=float(w_p),
        paired_t_stat=float(t_stat), paired_t_p=float(t_p),
        mean_diff_probe_minus_random=float((probe_combined - random_combined).mean()),
    )

    print(f"\nprobe combined shift:  mean={probe_combined.mean():+.5f}  "
          f"median={np.median(probe_combined):+.5f}  "
          f"frac_correct_direction={(probe_combined > 0).mean():.3f}")
    print(f"random combined shift: mean={random_combined.mean():+.5f}  "
          f"median={np.median(random_combined):+.5f}  "
          f"frac_correct_direction={(random_combined > 0).mean():.3f}")
    print(f"paired Wilcoxon: stat={w_stat:.3f} p={w_p:.4f}")
    print(f"paired t-test:   stat={t_stat:.3f} p={t_p:.4f}")

    results["by_domain"] = {}
    for dom in ["code", "reasoning", "chat"]:
        dom_pairs = [p for p in pairs if p["domain"] == dom]
        if not dom_pairs:
            continue
        dp = np.array([(p[f"causal_shift_high_into_low_probe"] + p[f"causal_shift_low_into_high_probe"]) / 2
                        for p in dom_pairs])
        dr = np.array([(p[f"causal_shift_high_into_low_random"] + p[f"causal_shift_low_into_high_random"]) / 2
                        for p in dom_pairs])
        dom_result = dict(
            n=len(dom_pairs),
            probe_mean=float(dp.mean()), probe_median=float(np.median(dp)),
            random_mean=float(dr.mean()), random_median=float(np.median(dr)),
            probe_frac_correct=float((dp > 0).mean()), random_frac_correct=float((dr > 0).mean()),
            inversion_random_beats_probe=bool(dr.mean() > dp.mean()),
        )
        if len(dom_pairs) >= 5:  # meaningful minimum for a paired test
            dw_stat, dw_p = wilcoxon(dp, dr)
            dom_result["wilcoxon_p"] = float(dw_p)
        results["by_domain"][dom] = dom_result
        print(f"[{dom}] n={len(dom_pairs)} probe_mean={dp.mean():+.5f} random_mean={dr.mean():+.5f} "
              f"inversion(random>probe)={dr.mean() > dp.mean()}")

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "interchange_analysis.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {args.out}/interchange_analysis.json")


if __name__ == "__main__":
    main()
