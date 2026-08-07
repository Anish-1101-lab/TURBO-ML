"""
Load Phase 2 shard files into unified in-memory tensors for probing. Whole
dataset is ~5GB in fp16, small enough to hold in RAM entirely rather than
streaming -- keeps the rest of the Phase 3 code simple.
"""
import glob
import os

import torch


def load_domain_shards(data_dir: str, domain: str) -> dict:
    files = sorted(glob.glob(os.path.join(data_dir, domain, "*.pt")))
    assert files, f"no shard files found for domain={domain} under {data_dir}"
    hs, label, p, q, accepted, entropy, position, prompt_id = [], [], [], [], [], [], [], []
    layer_indices = None
    for f in files:
        d = torch.load(f, weights_only=False)
        hs.append(d["hidden_states"])
        label.append(d["label"])
        p.append(d["p"])
        q.append(d["q"])
        accepted.append(d["accepted"])
        entropy.append(d["drafter_entropy"])
        position.append(d["position"])
        prompt_id.extend(d["prompt_id"])
        layer_indices = d["layer_indices"]
    return dict(
        hidden_states=torch.cat(hs), label=torch.cat(label), p=torch.cat(p),
        q=torch.cat(q), accepted=torch.cat(accepted), drafter_entropy=torch.cat(entropy),
        position=torch.cat(position), prompt_id=prompt_id, layer_indices=layer_indices,
        domain=domain,
    )


def strip_pass_suffix(prompt_id: str) -> str:
    """Collapse a domain's repeated-pass prompt_id (e.g. 'chat_12_pass1')
    down to the underlying source prompt ('chat_12'). Needed so that two
    passes over the same MT-Bench question (chat domain only, see
    PROGRESS.md Phase 2 caveat) always land in the same split -- otherwise
    near-duplicate content (same question, different sampled completion)
    could leak across train/val/test."""
    return prompt_id.rsplit("_pass", 1)[0]


def load_all_domains(data_dir: str, domains=("code", "reasoning", "chat")) -> dict:
    parts = [load_domain_shards(data_dir, d) for d in domains]
    layer_indices = parts[0]["layer_indices"]
    for part in parts:
        assert torch.equal(part["layer_indices"], layer_indices), "layer_indices mismatch across domain shards"

    domain_labels = []
    for part in parts:
        domain_labels.extend([part["domain"]] * len(part["label"]))

    out = dict(
        hidden_states=torch.cat([pt["hidden_states"] for pt in parts]),
        label=torch.cat([pt["label"] for pt in parts]),
        p=torch.cat([pt["p"] for pt in parts]),
        q=torch.cat([pt["q"] for pt in parts]),
        accepted=torch.cat([pt["accepted"] for pt in parts]),
        drafter_entropy=torch.cat([pt["drafter_entropy"] for pt in parts]),
        position=torch.cat([pt["position"] for pt in parts]),
        prompt_id=sum((pt["prompt_id"] for pt in parts), []),
        domain=domain_labels,
        layer_indices=layer_indices,
    )
    out["split_key"] = [f"{dom}::{strip_pass_suffix(pid)}" for dom, pid in zip(out["domain"], out["prompt_id"])]
    return out
