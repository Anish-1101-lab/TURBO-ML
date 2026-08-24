"""
Phase 4c follow-up analysis (not a rerun -- reads the existing
interchange_pairs.json from scripts/12): breaks the two swap directions
(high-into-low, low-into-high) out separately rather than pooling them,
after noticing the pooled "combined" metric in scripts/13 mixes a
floor-saturated swap direction (high-into-low: most LOW donors' clean
p(x) is already ~0, leaving near-zero room to rise) with a swap direction
that has real dynamic range (low-into-high: HIGH examples' clean p(x) has
real spread down to 0.33). Pooling them diluted the more informative
signal with the uninformative one. This script reports each separately,
with its own paired probe-vs-random test.

Usage: python scripts/14_interchange_swapdir_breakdown.py
"""
import json

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

pairs = json.load(open("analysis/phase4c/interchange_pairs.json"))

for swap in ["high_into_low", "low_into_high"]:
    probe = np.array([p[f"causal_shift_{swap}_probe"] for p in pairs])
    rand = np.array([p[f"causal_shift_{swap}_random"] for p in pairs])
    w_stat, w_p = wilcoxon(probe, rand)
    t_stat, t_p = ttest_rel(probe, rand)
    print(f"=== {swap} (n={len(pairs)}) ===")
    print(f"probe:  mean={probe.mean():+.5f} median={np.median(probe):+.6f} "
          f"frac_correct_direction={(probe > 0).mean():.3f} mean|shift|={np.abs(probe).mean():.5f}")
    print(f"random: mean={rand.mean():+.5f} median={np.median(rand):+.6f} "
          f"frac_correct_direction={(rand > 0).mean():.3f} mean|shift|={np.abs(rand).mean():.5f}")
    print(f"paired wilcoxon: stat={w_stat:.3f} p={w_p:.4f}   paired t-test: stat={t_stat:.3f} p={t_p:.4f}")
    print()

# also break low_into_high (the informative swap direction) out by domain
print("=== low_into_high by domain ===")
for dom in ["code", "reasoning", "chat"]:
    dp = [p for p in pairs if p["domain"] == dom]
    probe = np.array([p["causal_shift_low_into_high_probe"] for p in dp])
    rand = np.array([p["causal_shift_low_into_high_random"] for p in dp])
    w_stat, w_p = wilcoxon(probe, rand) if len(dp) >= 5 else (float("nan"), float("nan"))
    print(f"[{dom}] n={len(dp)} probe_mean={probe.mean():+.5f} (frac_correct={(probe>0).mean():.3f}) "
          f"random_mean={rand.mean():+.5f} (frac_correct={(rand>0).mean():.3f}) "
          f"inversion(random_mean>probe_mean)={rand.mean() > probe.mean()} wilcoxon_p={w_p:.4f}")
