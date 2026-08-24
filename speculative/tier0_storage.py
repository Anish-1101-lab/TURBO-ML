"""
Tier 0 storage: splits generation output into the two passes the spec
calls for.

Pass 1 (events + cheap features): small, kept as flat torch tensors --
one row per drafted-token event, Group A (lens, both positions) / B / C
features flattened into named columns alongside identifiers and the label.

Pass 2 (residual streams): large, cached one layer at a time -- one file
per probed layer, holding that layer's P_dec and P_tok hidden states for
every event, aligned by row index with the Pass 1 table (join key is the
row's integer index, also stored explicitly as event_row_id in Pass 1 for
safety against any future re-ordering).

Both passes are written with torch.save, matching the rest of this repo's
convention (speculative/storage.py) rather than introducing a parquet
dependency at the raw-data stage; parquet is reserved for the tidy
long-format *results* file downstream (results/tier0_results.parquet),
which is a later-stage deliverable, not raw per-event data.
"""
import os

import torch

LENS_FIELDS = ("p_top1", "entropy", "margin", "logp_draft", "log_rank_draft")
SURFACE_BOOL_FIELDS = ("is_punct", "is_leading_space", "is_subword_continuation", "is_digit", "is_newline")


def records_to_pass1_rows(records: list, prompt_id: str, domain: str, drafter_id: str, temperature: float) -> list:
    rows = []
    for event_idx, r in enumerate(records):
        row = dict(
            prompt_id=prompt_id, domain=domain, drafter_id=drafter_id, temperature=temperature,
            event_idx=event_idx, token_id=r["token_id"], p=r["p"], q=r["q"],
            label=r["label"], accepted=r["accepted"],
            position_dec=r["position_dec"], position_tok=r["position_tok"],
        )
        row.update(r["features"])
        for pos_key in ("pdec", "ptok"):
            for L, feats in ((k.split("_L")[1], v) for k, v in r["lens"].items() if k.startswith(pos_key)):
                for f in LENS_FIELDS:
                    row[f"{pos_key}_L{L}_{f}"] = feats[f]
        rows.append(row)
    return rows


def save_pass1(rows: list, out_dir: str, domain: str, temperature: float, drafter_id: str, shard_idx: int = 0):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{domain}_T{temperature}_{drafter_id}_shard{shard_idx:05d}.pt")
    torch.save(rows, path)
    return path


def save_pass2_per_layer(records: list, probed_layers: list, out_dir: str, domain: str,
                          temperature: float, drafter_id: str, shard_idx: int = 0):
    """One file per probed layer: {'hs_dec': [N, D] fp16, 'hs_tok': [N, D] fp16}."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for L in probed_layers:
        hs_dec = torch.stack([r["hs_dec"][L] for r in records])
        hs_tok = torch.stack([r["hs_tok"][L] for r in records])
        path = os.path.join(out_dir, f"{domain}_T{temperature}_{drafter_id}_L{L}_shard{shard_idx:05d}.pt")
        torch.save(dict(hs_dec=hs_dec, hs_tok=hs_tok), path)
        paths[L] = path
    return paths
