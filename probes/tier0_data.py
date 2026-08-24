"""
Load Tier 0 Pass 1 (events + cheap features) and Pass 2 (per-layer residuals)
data into pandas/numpy for analysis. Pass 1 rows are converted to a DataFrame
(vectorized feature slicing); Pass 2 is loaded ONE LAYER AT A TIME by the
caller (never all layers at once) via load_hs_layer, per the spec's
"cache/load one layer at a time, free it" instruction.
"""
import glob
import os

import numpy as np
import pandas as pd
import torch

from probes.tier0_splits import strip_pass_suffix

LENS_FIELDS = ("p_top1", "entropy", "margin", "logp_draft", "log_rank_draft")

GROUP_B_COLS = ["q_draft", "q_entropy", "q_margin"]
GROUP_C_BOOL_COLS = ["is_punct", "is_leading_space", "is_subword_continuation", "is_digit", "is_newline"]
GROUP_C_NUM_COLS = ["log_unigram_freq", "position_in_draft_window", "log_prefix_length"]
GROUP_C_COLS = GROUP_C_NUM_COLS + GROUP_C_BOOL_COLS


def group_a_cols(layer: int, pos: str = "pdec") -> list:
    return [f"{pos}_L{layer}_{f}" for f in LENS_FIELDS]


def load_pass1_df(pass1_dir: str, domain: str, temperature: float, drafter: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(pass1_dir, f"{domain}_T{temperature}_{drafter}_shard*.pt")))
    assert files, f"no Pass-1 shards for domain={domain} T={temperature} drafter={drafter} in {pass1_dir}"
    rows = []
    for f in files:
        rows.extend(torch.load(f, weights_only=False))
    df = pd.DataFrame(rows)
    for c in GROUP_C_BOOL_COLS:
        df[c] = df[c].astype(float)
    df["accepted"] = df["accepted"].astype(int)
    df["split_key"] = df["prompt_id"].apply(strip_pass_suffix)
    df["shard_order"] = np.arange(len(df))  # preserves Pass-2 alignment order
    return df


def load_hs_layer(pass2_dir: str, domain: str, temperature: float, drafter: str, layer: int) -> dict:
    """Returns {'hs_dec': [N,D] float32, 'hs_tok': [N,D] float32} in the same
    row order as load_pass1_df's output for the identical (domain, temperature,
    drafter) -- both iterate shard files in the same sorted order and preserve
    within-shard row order, so this aligns 1:1 with the DataFrame by position."""
    files = sorted(glob.glob(os.path.join(pass2_dir, f"{domain}_T{temperature}_{drafter}_L{layer}_shard*.pt")))
    assert files, f"no Pass-2 shards for domain={domain} T={temperature} drafter={drafter} L={layer} in {pass2_dir}"
    dec_parts, tok_parts = [], []
    for f in files:
        d = torch.load(f, weights_only=False)
        dec_parts.append(d["hs_dec"].float().numpy())
        tok_parts.append(d["hs_tok"].float().numpy())
    return dict(hs_dec=np.concatenate(dec_parts), hs_tok=np.concatenate(tok_parts))


def load_all_domains_pass1(pass1_dir: str, domains: list, temperature: float, drafter: str) -> pd.DataFrame:
    parts = [load_pass1_df(pass1_dir, d, temperature, drafter) for d in domains]
    return pd.concat(parts, ignore_index=False, keys=domains, names=["domain_group"]).reset_index(drop=True)
