"""
Tier 0 cheap-feature extraction (Group A logit-lens, Group B drafter,
Group C surface) shared by Pass 1 (event + cheap-feature rows) and used
inline during Pass 2 generation.

Group A features use the LOGIT LENS -- unembedding an intermediate layer's
residual stream through the target's own final norm + LM head -- rather
than a tuned lens (tuned lens is out of scope for Tier 0, see the spec).
This is a known-biased readout at early layers: it understates how much
confidence signal is actually decodable there. That biases *against*
Tier 0's own null hypothesis (that cheap features already explain the
probe's AUROC), which is the safe direction to be biased in.
"""
import math
import re

import torch

PUNCT_RE = re.compile(r"^[^\w\s]+$")
DIGIT_RE = re.compile(r"^\d+$")


def build_unigram_log_freq(tokenizer, vocab_size: int, n_samples: int, seed: int) -> torch.Tensor:
    """Token unigram log-frequency table estimated from a WikiText-103 sample,
    add-1 smoothed over the full vocab (unseen tokens get the smoothing floor,
    not zero probability)."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=g)[:n_samples].tolist()

    counts = torch.ones(vocab_size, dtype=torch.float64)
    for i in idx:
        text = ds[i]["text"]
        if not text.strip():
            continue
        ids = tokenizer(text, add_special_tokens=False).input_ids
        for t in ids:
            if t < vocab_size:
                counts[t] += 1
    return torch.log(counts / counts.sum())


def surface_features(tokenizer, token_id: int) -> dict:
    piece = tokenizer.convert_ids_to_tokens([token_id])[0]
    text = tokenizer.decode([token_id])
    is_leading_space = piece.startswith("Ġ") or piece.startswith("▁") or text.startswith(" ")
    stripped = text.strip()
    is_punct = bool(stripped) and bool(PUNCT_RE.match(stripped))
    is_digit = bool(stripped) and bool(DIGIT_RE.match(stripped))
    is_newline = "\n" in text
    # Heuristic: a piece that doesn't start a new word (no leading space) and
    # isn't a newline is treated as continuing the previous subword. Byte-level
    # BPE (Qwen2/Llama-3) marks word-starts explicitly, so this is a reasonable
    # proxy rather than an exact segmentation-derived flag.
    is_subword_continuation = not is_leading_space and not is_newline
    return dict(
        is_punct=is_punct, is_leading_space=is_leading_space,
        is_subword_continuation=is_subword_continuation,
        is_digit=is_digit, is_newline=is_newline,
    )


def logit_lens_batch(target, hidden_batch: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """hidden_batch: [B, D] (any float dtype/device -- cast to the model's own
    param dtype internally, since norm/lm_head are bf16 and hidden_states may
    be captured in a different dtype). Returns softmax probs [B, vocab_size]
    float32."""
    model_dtype = target.lm_head.weight.dtype
    h = hidden_batch.to(model_dtype).unsqueeze(1)   # [B, 1, D]
    normed = target.model.norm(h)                    # Qwen2ForCausalLM: model.model.norm is the final RMSNorm
    logits = target.lm_head(normed)[:, 0, :vocab_size].float()
    return torch.softmax(logits, dim=-1)


def lens_features(probs_row: torch.Tensor, draft_token_id: int) -> dict:
    p_top1 = probs_row.max().item()
    logp_all = probs_row.clamp_min(1e-12).log()
    entropy = -(probs_row * logp_all).sum().item()
    top2 = torch.topk(probs_row, 2).values
    margin = (top2[0] - top2[1]).item()
    p_draft = probs_row[draft_token_id].item()
    logp_draft = math.log(max(p_draft, 1e-12))
    rank_draft = int((probs_row > p_draft).sum().item()) + 1
    log_rank_draft = math.log(1 + rank_draft)
    return dict(
        p_top1=p_top1, entropy=entropy, margin=margin,
        logp_draft=logp_draft, log_rank_draft=log_rank_draft,
    )
