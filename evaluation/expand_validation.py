"""
Validation set expansion — VESTIGE Phase 3 extension.

Fetches genomic coordinates for 5 additional genes (not in training data)
from the Asian Elephant reference, extracts 2000 bp windows, tokenizes,
applies PMD damage simulation, and saves to:

    evaluation/damaged_validation_extra.npy

evaluate_scaling.py automatically loads this file (if present) and appends
it to the original 69 validation windows before running the T_END sweep.

New genes (all cold-adaptation / mammoth-relevant, not in training data):
    TRPA1  — cold/pain-sensing TRP channel (same family as TRPV3)
    UCP1   — uncoupling protein 1, thermogenesis
    MC1R   — melanocortin-1 receptor, coat colour (studied in mammoths)
    ADRB3  — beta-3 adrenergic receptor, thermogenesis
    FASN   — fatty acid synthase, lipid metabolism / cold adaptation

Run from project root:
    NCBI_EMAIL=your@email.com python3 evaluation/expand_validation.py
"""
import os
import ssl
import sys
import time
import types
import importlib.util
import importlib.machinery

ssl._create_default_https_context = ssl._create_unverified_context

# triton stub removed: it broke transformers 4.57 import; DNABERT-2 falls back to torch attention.

import os  # force torch-only: transformers TF backend breaks with Keras 3 (requirements.txt)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import numpy as np
from pathlib import Path
from Bio import Entrez, SeqIO
from Bio.Seq import Seq
from transformers import AutoTokenizer

# ── Config ─────────────────────────────────────────────────────────────────────
TOKENIZER_NAME = "zhihan1996/DNABERT-2-117M"
REFERENCE_FA   = Path("Dataset/elephant_reference/elephant_ref.fa")
PROFILE_PATH   = Path("damage/damage_profile.npy")
OUT_PATH       = Path("evaluation/damaged_validation_extra.npy")

WINDOW_BP  = 2000
STRIDE_BP  = 200
MAX_TOKENS = 512
SEED       = 42

Entrez.email = os.environ.get("NCBI_EMAIL", "phoenix@example.com")

EXTRA_GENES = ["TRPA1", "UCP1", "MC1R", "ADRB3", "FASN"]
# ───────────────────────────────────────────────────────────────────────────────


# ── NCBI coordinate fetch ──────────────────────────────────────────────────────

def fetch_gene_coords(gene_name: str) -> dict | None:
    """
    Search NCBI Gene for gene_name in Elephas maximus.
    Returns dict with chr, start, end, strand or None if not found.
    """
    query = f"{gene_name}[Gene Name] AND Elephas maximus[Organism]"
    try:
        handle = Entrez.esearch(db="gene", term=query, retmax=5)
        record = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"  esearch failed for {gene_name}: {e}")
        return None

    if not record["IdList"]:
        # Fallback: broader description search
        query2 = f"{gene_name}[Description] AND txid9783[Organism:exp]"
        try:
            handle = Entrez.esearch(db="gene", term=query2, retmax=5)
            record = Entrez.read(handle)
            handle.close()
        except Exception as e:
            print(f"  fallback esearch failed for {gene_name}: {e}")
            return None

    if not record["IdList"]:
        print(f"  {gene_name}: not found in NCBI Gene for Elephas maximus.")
        return None

    gene_id = record["IdList"][0]
    try:
        handle = Entrez.esummary(db="gene", id=gene_id)
        summary = Entrez.read(handle)
        handle.close()
    except Exception as e:
        print(f"  esummary failed for {gene_name} (id={gene_id}): {e}")
        return None

    doc = summary["DocumentSummarySet"]["DocumentSummary"][0]
    try:
        ginfo = doc["GenomicInfo"][0]
        chrom  = ginfo["ChrAccVer"]
        start  = int(ginfo["ChrStart"]) + 1   # NCBI is 0-based, convert to 1-based
        end    = int(ginfo["ChrStop"])  + 1
        # NCBI stores start < end always; strand inferred from location
        if start > end:
            start, end = end, start
            strand = "-"
        else:
            strand = "+"
        return {"gene": gene_name, "chr": chrom,
                "start": start, "end": end, "strand": strand}
    except (KeyError, IndexError) as e:
        print(f"  {gene_name}: could not parse GenomicInfo — {e}")
        return None


# ── Sequence extraction & tiling ──────────────────────────────────────────────

def extract_seq(ref_index, chrom: str, start: int, end: int, strand: str) -> str:
    if chrom not in ref_index:
        return ""
    seq = str(ref_index[chrom][start - 1:end].seq).upper()
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


def tile(seq: str, window: int, stride: int) -> list[str]:
    windows = []
    for i in range(0, max(1, len(seq) - window + 1), stride):
        windows.append(seq[i:i + window])
    if len(seq) > window and (len(seq) - window) % stride != 0:
        windows.append(seq[-window:])
    return windows


# ── Damage simulation (mirrors simulate_damage.py) ────────────────────────────

def decode_to_dna(token_ids, tokenizer) -> str:
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    specials = {tokenizer.cls_token, tokenizer.sep_token,
                tokenizer.pad_token, tokenizer.unk_token, tokenizer.mask_token}
    return "".join(
        tok.lstrip("#").lstrip("▁").upper()
        for tok in tokens
        if tok not in specials and tok is not None
    )


def apply_damage(dna: str, ct_5p: np.ndarray, ga_3p: np.ndarray,
                 rng) -> tuple[str, list[int]]:
    seq = list(dna)
    n   = len(seq)
    damaged = []
    for i, base in enumerate(seq):
        if base == "C":
            prob = ct_5p[min(i, len(ct_5p) - 1)]
            if rng.random() < prob:
                seq[i] = "T"
                damaged.append(i)
        elif base == "G":
            d3 = n - 1 - i
            prob = ga_3p[min(d3, len(ga_3p) - 1)]
            if rng.random() < prob:
                seq[i] = "A"
                damaged.append(i)
    return "".join(seq), damaged


def nuc_to_tok(nuc_pos: int, offsets) -> int | None:
    for idx, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s <= nuc_pos < e:
            return idx
    return None


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    profile   = np.load(PROFILE_PATH, allow_pickle=True).item()
    ct_5p     = profile["ct_5p"]
    ga_3p     = profile["ga_3p"]

    print(f"Loading reference index from {REFERENCE_FA} ...")
    ref_index = SeqIO.index(str(REFERENCE_FA), "fasta")

    results = []

    for gene_name in EXTRA_GENES:
        print(f"\n── {gene_name} ────────────────────────────────")
        coords = fetch_gene_coords(gene_name)
        time.sleep(0.4)   # NCBI rate limit

        if coords is None:
            print(f"  Skipping {gene_name}.")
            continue

        print(f"  Coords: {coords['chr']}:{coords['start']}-{coords['end']} ({coords['strand']})")
        seq = extract_seq(ref_index, coords["chr"],
                          coords["start"], coords["end"], coords["strand"])
        if not seq:
            print(f"  {gene_name}: chromosome {coords['chr']} not in reference. Skipping.")
            continue

        print(f"  Genomic span: {len(seq):,} bp")
        raw_windows = tile(seq, WINDOW_BP, STRIDE_BP)

        kept = filtered = 0
        for w in raw_windows:
            enc = tokenizer(
                w,
                return_offsets_mapping=True,
                padding="max_length",
                truncation=True,
                max_length=MAX_TOKENS,
            )
            if len(tokenizer(w, truncation=False)["input_ids"]) > MAX_TOKENS:
                filtered += 1
                continue

            gt_ids  = enc["input_ids"]
            attn    = enc["attention_mask"]
            offsets = enc["offset_mapping"]

            dna_gt = decode_to_dna(gt_ids, tokenizer)
            if not dna_gt:
                continue

            dna_dam, nuc_dam = apply_damage(dna_gt, ct_5p, ga_3p, rng)

            tok_dam = []
            nuc_in  = []
            for npos in nuc_dam:
                tidx = nuc_to_tok(npos, offsets)
                if tidx is not None:
                    tok_dam.append(tidx)
                    nuc_in.append(npos)

            results.append({
                "gene":           gene_name,
                "dna_gt":         dna_gt,
                "dna_damaged":    dna_dam,
                "nuc_damaged":    nuc_in,
                "tok_damaged":    sorted(set(tok_dam)),
                "gt_ids":         gt_ids,
                "attention_mask": attn,
                "offsets":        offsets,
            })
            kept += 1

        print(f"  Windows: {len(raw_windows)} raw  →  {kept} kept  ({filtered} filtered >512 tokens)")

    np.save(OUT_PATH, results, allow_pickle=True)

    total_dam = sum(len(r["nuc_damaged"]) for r in results)
    with_dam  = sum(1 for r in results if r["nuc_damaged"])
    print(f"\n── Summary ──────────────────────────────────────")
    print(f"  Extra windows saved : {len(results)}")
    print(f"  With ≥1 damage      : {with_dam}")
    print(f"  Total damaged nuc   : {total_dam}")
    print(f"  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
