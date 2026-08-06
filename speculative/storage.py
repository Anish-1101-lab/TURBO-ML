"""
Shard writer for logged drafted-token records. One row per drafted token;
hidden_states stored as a stacked [num_logged_layers, hidden_dim] fp16
tensor per row so Phase 3 can slice out a single layer across all rows
cheaply. prompt_id is carried per-row specifically so Phase 3 can split
train/val/test by prompt rather than by token (avoids leakage across
positions of the same generated sequence).
"""
import json
import os

import torch


class ShardWriter:
    def __init__(self, out_dir: str, domain: str, layer_indices: list, shard_size: int = 2000):
        self.out_dir = out_dir
        self.domain = domain
        self.layer_indices = layer_indices
        self.shard_size = shard_size
        os.makedirs(out_dir, exist_ok=True)
        self.buffer = []
        self.shard_idx = 0
        self.total_written = 0

    def add(self, record: dict):
        self.buffer.append(record)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        hs = torch.stack([
            torch.stack([r["hidden_states"][L] for L in self.layer_indices])
            for r in self.buffer
        ])  # [N, num_layers, D], fp16

        out = {
            "hidden_states": hs,
            "layer_indices": torch.tensor(self.layer_indices),
            "label": torch.tensor([r["label"] for r in self.buffer], dtype=torch.float32),
            "p": torch.tensor([r["p"] for r in self.buffer], dtype=torch.float32),
            "q": torch.tensor([r["q"] for r in self.buffer], dtype=torch.float32),
            "token_id": torch.tensor([r["token_id"] for r in self.buffer], dtype=torch.int64),
            "accepted": torch.tensor([r["accepted"] for r in self.buffer], dtype=torch.bool),
            "drafter_entropy": torch.tensor([r["drafter_entropy"] for r in self.buffer], dtype=torch.float32),
            "position": torch.tensor([r["position"] for r in self.buffer], dtype=torch.int64),
            "prompt_id": [r["prompt_id"] for r in self.buffer],
            "domain": self.domain,
        }
        path = os.path.join(self.out_dir, f"{self.domain}_shard{self.shard_idx:05d}.pt")
        torch.save(out, path)
        self.total_written += len(self.buffer)
        self.shard_idx += 1
        self.buffer = []

    def close(self):
        self.flush()


def write_manifest(out_dir: str, manifest: dict):
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
