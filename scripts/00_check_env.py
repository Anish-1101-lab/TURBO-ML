"""
Phase 0 sanity check: load target + drafter, confirm tokenizer compatibility,
and run one basic (non-speculative) forward pass on each. No SD logic here.

Usage: python scripts/00_check_env.py --config configs/models.yaml
"""
import argparse
import os

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["device"]["cuda_visible_devices"])
    torch.manual_seed(cfg["seed"])

    assert torch.cuda.is_available(), "CUDA not available"
    device = "cuda:0"  # index 0 within the CUDA_VISIBLE_DEVICES-restricted view
    print(f"Visible device: {torch.cuda.get_device_name(0)}")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

    results = {}
    for role in ("target", "drafter"):
        spec = cfg[role]
        print(f"\n=== Loading {role}: {spec['name']} ===")
        tok = AutoTokenizer.from_pretrained(spec["name"], revision=spec["revision"])
        model = AutoModelForCausalLM.from_pretrained(
            spec["name"],
            revision=spec["revision"],
            torch_dtype=dtype_map[spec["dtype"]],
        ).to(device)
        model.eval()

        prompt = "The capital of France is"
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)

        n_layers = len(out.hidden_states) - 1  # exclude embedding layer output
        print(f"vocab_size={tok.vocab_size}, num_hidden_layers={n_layers}")
        print(f"logits shape: {tuple(out.logits.shape)}")
        print(f"hidden_states[1] shape: {tuple(out.hidden_states[1].shape)}")
        top_token = tok.decode(out.logits[0, -1].argmax())
        print(f"greedy next token after '{prompt}': {top_token!r}")

        mem_gb = torch.cuda.memory_allocated(device) / 1e9
        print(f"GPU memory allocated after load: {mem_gb:.2f} GB")

        results[role] = {
            "tokenizer": tok,
            "vocab_size": tok.vocab_size,
            "num_layers": n_layers,
        }
        del model
        torch.cuda.empty_cache()

    # Vocab compatibility check: drafter and target must agree on token ids
    # for the accept/reject rule to be meaningful without remapping.
    t_vocab = results["target"]["tokenizer"].get_vocab()
    d_vocab = results["drafter"]["tokenizer"].get_vocab()
    same_size = results["target"]["vocab_size"] == results["drafter"]["vocab_size"]
    same_map = t_vocab == d_vocab
    print(f"\n=== Vocab compatibility ===")
    print(f"same vocab_size: {same_size}")
    print(f"identical token->id mapping: {same_map}")
    assert same_map, "Tokenizer vocabularies differ between target and drafter"

    print("\nPhase 0 check PASSED.")


if __name__ == "__main__":
    main()
