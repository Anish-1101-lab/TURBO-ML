"""
Train/val/test split BY PROMPT (split_key), not by token, stratified by
domain so every split has representation from code/reasoning/chat. Splitting
by token instead of prompt would leak: multiple drafted-token positions from
the same generated sequence share context and would let a probe partially
memorize sequence-specific structure rather than learning a genuine
hidden-state -> acceptance mapping.
"""
import random


def make_split(dataset: dict, train_frac: float, val_frac: float, test_frac: float, seed: int) -> list:
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    unique_keys_by_domain = {}
    for dom, key in zip(dataset["domain"], dataset["split_key"]):
        unique_keys_by_domain.setdefault(dom, set()).add(key)

    rng = random.Random(seed)
    key_to_split = {}
    for dom, keys in unique_keys_by_domain.items():
        keys = sorted(keys)
        rng.shuffle(keys)
        n = len(keys)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        for k in keys[:n_train]:
            key_to_split[k] = "train"
        for k in keys[n_train:n_train + n_val]:
            key_to_split[k] = "val"
        for k in keys[n_train + n_val:]:
            key_to_split[k] = "test"

    return [key_to_split[k] for k in dataset["split_key"]]
