"""
In silico post-mortem damage simulation — VESTIGE Phase 3.

Applies C→T (5′) and G→A (3′) substitutions at the nucleotide level using
empirical PMD frequencies from damage_profile.npy, then uses the tokenizer's
offset_mapping to record exactly which token position in the original gt_ids
spans each damaged nucleotide.

BPE cascade avoided: we never re-tokenize the damaged sequence. Instead:
  - damage is applied to the raw DNA string
  - offset_mapping on the ORIGINAL tokenization maps nucleotide positions → token positions
  - evaluation masks those specific token positions in gt_ids and asks the model to fill them in

Output: evaluation/damaged_validation.npy
    list of dicts per validation window:
    {
        "gene":            str,
        "dna_gt":          str,           # original nucleotide string
        "dna_damaged":     str,           # after in silico damage
        "nuc_damaged":     list[int],     # 0-based nucleotide positions that were damaged
        "tok_damaged":     list[int],     # corresponding token positions in gt_ids
        "gt_ids":          list[int],     # original token ids (CLS…SEP…PAD)
        "attention_mask":  list[int],
        "offsets":         list[tuple],   # [(start_char, end_char)] per token
    }

Run from project root:
    python evaluation/simulate_damage.py
"""
import sys, types, importlib.util, importlib.machinery

# triton stub removed: it broke transformers 4.57 import; DNABERT-2 falls back to torch attention.

import os  # force torch-only: transformers TF backend breaks with Keras 3 (requirements.txt)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer

DATASET_PATH   = "training/dataset"
PROFILE_PATH   = "damage/damage_profile.npy"
TOKENIZER_NAME = "zhihan1996/DNABERT-2-117M"
OUT_PATH       = "evaluation/damaged_validation.npy"
MAX_TOKENS     = 512
SEED           = 42


def decode_to_dna(token_ids: list[int], tokenizer) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    specials = {
        tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.unk_token,
        tokenizer.mask_token,
    }
    return "".join(
        tok.lstrip("#").lstrip("▁").upper()
        for tok in tokens
        if tok not in specials and tok is not None
    )


def apply_damage(dna: str, ct_5p: np.ndarray, ga_3p: np.ndarray, rng) -> tuple[str, list[int]]:
    """Return (damaged_dna, list_of_damaged_nucleotide_positions)."""
    seq  = list(dna)
    n    = len(seq)
    damaged_positions = []

    for i, base in enumerate(seq):
        if base == "C":
            prob = ct_5p[min(i, len(ct_5p) - 1)]
            if rng.random() < prob:
                seq[i] = "T"
                damaged_positions.append(i)
        elif base == "G":
            dist_3p = n - 1 - i
            prob = ga_3p[min(dist_3p, len(ga_3p) - 1)]
            if rng.random() < prob:
                seq[i] = "A"
                damaged_positions.append(i)

    return "".join(seq), damaged_positions


def nuc_pos_to_token_pos(nuc_pos: int, offsets: list[tuple]) -> int | None:
    """
    Given a 0-based nucleotide position in the raw DNA string,
    return the index into the token list whose character span contains it.
    Returns None if the position is beyond the tokenized range (truncation).
    """
    for tok_idx, (start, end) in enumerate(offsets):
        if start == end:
            continue   # special token with zero-width span
        if start <= nuc_pos < end:
            return tok_idx
    return None


def main():
    rng = np.random.default_rng(SEED)

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    profile   = np.load(PROFILE_PATH, allow_pickle=True).item()
    ct_5p     = profile["ct_5p"]
    ga_3p     = profile["ga_3p"]

    ds  = load_from_disk(DATASET_PATH)
    val = ds["validation"]

    results = []
    n_dam_total = 0

    for row in val:
        dna_gt = decode_to_dna(row["input_ids"], tokenizer)
        if not dna_gt:
            continue

        dna_damaged, nuc_damaged = apply_damage(dna_gt, ct_5p, ga_3p, rng)

        # Tokenize original DNA with offset mapping to locate damaged tokens
        enc = tokenizer(
            dna_gt,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=MAX_TOKENS,
        )
        gt_ids   = enc["input_ids"]
        attn     = enc["attention_mask"]
        offsets  = enc["offset_mapping"]   # list of (start_char, end_char)

        # Map each damaged nucleotide position → the token that spans it
        tok_damaged = []
        nuc_in_range = []
        for npos in nuc_damaged:
            tidx = nuc_pos_to_token_pos(npos, offsets)
            if tidx is not None:
                tok_damaged.append(tidx)
                nuc_in_range.append(npos)

        # Deduplicate token positions (two damaged nucs can share a BPE token)
        tok_damaged_dedup = sorted(set(tok_damaged))

        n_dam_total += len(nuc_in_range)
        results.append({
            "gene":           row["gene"],
            "dna_gt":         dna_gt,
            "dna_damaged":    dna_damaged,
            "nuc_damaged":    nuc_in_range,
            "tok_damaged":    tok_damaged_dedup,
            "gt_ids":         gt_ids,
            "attention_mask": attn,
            "offsets":        offsets,
        })

    out = Path(OUT_PATH)
    out.parent.mkdir(exist_ok=True)
    np.save(out, results, allow_pickle=True)

    n_windows  = len(results)
    n_with_dam = sum(1 for r in results if r["nuc_damaged"])
    avg_nuc    = n_dam_total / n_windows
    avg_tok    = sum(len(r["tok_damaged"]) for r in results) / n_windows

    print(f"Windows total:             {n_windows}")
    print(f"Windows with ≥1 damage:    {n_with_dam}")
    print(f"Total damaged nucleotides: {n_dam_total}")
    print(f"Mean damaged nuc/window:   {avg_nuc:.2f}")
    print(f"Mean damaged tokens/window:{avg_tok:.2f}  (BPE, after dedup)")
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
