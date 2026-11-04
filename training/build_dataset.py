"""
Fine-tuning dataset construction for VESTIGE Phase 2.

Extracts CDS sequences for TRPV3, KCNK9, HBB from the Asian Elephant
reference (elephant_ref.fa), tiles into overlapping windows, tokenizes
with the DNABERT-2 tokenizer, and saves an 80/20 train/val split as a
HuggingFace DatasetDict.

Window / stride parameters
--------------------------
WINDOW_BP = 2000  →  ~400 tokens (well within the 512-token model limit)
STRIDE_BP  = 200  →  ~40-token stride, 10× overlap between consecutive windows

Stride is kept in bp rather than tokens because the DamageAwareDataCollator
uses in-window token positions as proxies for fragment-terminus distance.
Tiling in bp keeps those positions meaningful.

Soft-masking note: elephant_ref.fa uses lowercase letters for RepeatMasker
soft-masked regions. DNABERT-2's BPE tokenizer does not have lowercase
tokens — lowercase input causes character-level fallback and roughly doubles
the token count. All sequences are uppercased before tokenization.

Run from project root:
    python training/build_dataset.py

Outputs:
    training/dataset/          HuggingFace DatasetDict (train + validation)
    training/dataset_stats.txt summary table
"""
import csv
import os
import random
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# ── Config ────────────────────────────────────────────────────────────────────
# Tokenizer + output are env-configurable for the D2 second-backbone study
# (VESTIGE_MODEL / VESTIGE_OUT); defaults reproduce the original DNABERT-2 build.
GENE_COORDS   = Path("damage/gene_coords.csv")
REFERENCE_FA  = Path("Dataset/elephant_reference/elephant_ref.fa")
MODEL_NAME    = os.environ.get("VESTIGE_MODEL", "zhihan1996/DNABERT-2-117M")
OUT_DIR       = Path(os.environ.get("VESTIGE_OUT", "training/dataset"))
STATS_FILE    = OUT_DIR.parent / f"{OUT_DIR.name}_stats.txt"

WINDOW_BP     = 2000
STRIDE_BP     = 200
MAX_TOKENS    = 512
VAL_FRACTION  = 0.20
RANDOM_SEED   = 42
# ──────────────────────────────────────────────────────────────────────────────


def extract_cds(idx, chrom: str, start: int, end: int, strand: str) -> str:
    """Return uppercased CDS string (reverse-complemented if minus strand)."""
    seq = str(idx[chrom][start - 1:end].seq).upper()
    if strand == "-":
        seq = str(Seq(seq).reverse_complement())
    return seq


def tile(seq: str, window: int, stride: int) -> list[str]:
    """Sliding window over seq; last window always included."""
    windows = []
    for i in range(0, max(1, len(seq) - window + 1), stride):
        windows.append(seq[i:i + window])
    # Include tail if not already covered
    if len(seq) > window and (len(seq) - window) % stride != 0:
        windows.append(seq[-window:])
    return windows


def main():
    random.seed(RANDOM_SEED)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    ref_index = SeqIO.index(str(REFERENCE_FA), "fasta")

    # Read gene coordinates
    genes = []
    with open(GENE_COORDS) as fh:
        for row in csv.DictReader(fh):
            genes.append(row)

    all_windows: list[dict] = []
    stats_lines = []
    stats_lines.append(f"{'Gene':<8} {'CDS bp':>8} {'Windows':>8} {'Filtered':>8} {'Kept':>8}")
    stats_lines.append("-" * 46)

    for g in genes:
        gene      = g["gene"]
        chrom     = g["chr"]
        start     = int(g["start"])
        end       = int(g["end"])
        strand    = g["strand"]
        cds_len   = end - start + 1

        cds = extract_cds(ref_index, chrom, start, end, strand)
        raw_windows = tile(cds, WINDOW_BP, STRIDE_BP)

        kept = []
        filtered = 0
        for w in raw_windows:
            ids = tokenizer(w, truncation=False)["input_ids"]
            if len(ids) > MAX_TOKENS:
                filtered += 1
                continue
            kept.append({"sequence": w, "gene": gene})

        stats_lines.append(
            f"{gene:<8} {cds_len:>8,} {len(raw_windows):>8} {filtered:>8} {len(kept):>8}"
        )
        all_windows.extend(kept)

    stats_lines.append("-" * 46)
    stats_lines.append(f"{'TOTAL':<8} {'':>8} {'':>8} {'':>8} {len(all_windows):>8}")

    # ── Train / val split ────────────────────────────────────────────────────
    random.shuffle(all_windows)
    split = int(len(all_windows) * (1 - VAL_FRACTION))
    train_data = all_windows[:split]
    val_data   = all_windows[split:]

    stats_lines.append(f"\nTrain windows : {len(train_data)}")
    stats_lines.append(f"Val   windows : {len(val_data)}")
    stats_lines.append(f"Window bp     : {WINDOW_BP}")
    stats_lines.append(f"Stride bp     : {STRIDE_BP}")
    stats_lines.append(f"Max tokens    : {MAX_TOKENS}")

    # ── Tokenize and build HuggingFace DatasetDict ───────────────────────────
    def tokenize(batch):
        return tokenizer(
            batch["sequence"],
            truncation=True,
            max_length=MAX_TOKENS,
            padding="max_length",
        )

    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True, remove_columns=["sequence"])
    val_ds   = Dataset.from_list(val_data).map(tokenize, batched=True, remove_columns=["sequence"])

    dataset = DatasetDict({"train": train_ds, "validation": val_ds})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(OUT_DIR))

    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATS_FILE.write_text("\n".join(stats_lines) + "\n")

    print("\n".join(stats_lines))
    print(f"\nSaved → {OUT_DIR}")
    print(f"Stats  → {STATS_FILE}")


if __name__ == "__main__":
    main()
