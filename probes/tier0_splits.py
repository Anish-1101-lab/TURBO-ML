"""
Tier 0 prompt-level split. Split key is (domain, underlying prompt idx) --
NOT domain+temperature+drafter -- so the same underlying source prompt lands
in the same split across both temperatures and across all drafters (A/B/C).
This matters most for E3: drafters A/B/C are generated with independent
budget-cycling random walks over each domain's prompt list, so they don't
necessarily touch the identical observed subset of prompts. Split assignment
is therefore a DETERMINISTIC HASH of (seed, split_key) -> [0,1), not a
shuffle-and-slice over an observed set -- any generation run assigns the
same underlying prompt to the same split without needing to have seen the
same set of prompts as any other run. This trades slightly-off-70/15/15
exact counts (especially for small pools like chat's 80 MT-Bench prompts)
for cross-run/cross-drafter split consistency, which is the property E3
actually needs.
"""
import hashlib


def strip_pass_suffix(prompt_id: str) -> str:
    return prompt_id.rsplit("_pass", 1)[0]


def _stable_unit_interval(key: str, seed: int) -> float:
    h = hashlib.md5(f"{seed}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 16 ** 12


def assign_split(split_key: str, train_frac: float, val_frac: float, test_frac: float, seed: int) -> str:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    u = _stable_unit_interval(split_key, seed)
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def make_tier0_split(domains_and_idxs: dict, train_frac: float, val_frac: float,
                      test_frac: float, seed: int) -> dict:
    """domains_and_idxs: {domain: iterable of split_keys actually observed in
    this generation run}. Returns {split_key: 'train'|'val'|'test'} via the
    deterministic per-key hash, so it agrees with any other run's call on
    the same keys regardless of what else that run observed."""
    key_to_split = {}
    for _dom, keys in domains_and_idxs.items():
        for k in keys:
            key_to_split[k] = assign_split(k, train_frac, val_frac, test_frac, seed)
    return key_to_split
