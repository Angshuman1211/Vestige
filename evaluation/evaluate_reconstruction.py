"""
Reconstruction evaluation — VESTIGE Phase 3.

For each of the 69 validation windows:
  1. Mask the token position(s) in gt_ids that span the damaged nucleotide(s).
     One or two tokens at most per window (authentic PMD rates, no BPE cascade).
  2. Run each model; take argmax at masked position(s).
  3. Decode the predicted token to a nucleotide string.
  4. At the damaged nucleotide's offset within the token, compare to ground truth.

This is the correct evaluation: the model sees the original tokenization of an
otherwise unmodified sequence and is asked to fill in the specific token(s) that
overlap a biologically likely damage site. It mirrors the training task exactly —
masked token prediction from context.

Metrics:
  - Nucleotide recovery rate per window (at damaged positions only)
  - Paired t-test + bootstrap 95% CI (DAM vs. MLM, windows with ≥1 damage)
  - BLEU-4 (corpus, full decoded sequence, character-by-character)

Outputs:
  - evaluation/results_table1.json
  - evaluation/table1.txt

Run from project root:
    python evaluation/evaluate_reconstruction.py
"""
import sys, types, importlib.util, importlib.machinery

# triton stub removed: it broke transformers 4.57 import; DNABERT-2 falls back to torch attention.

import os  # force torch-only: transformers TF backend breaks with Keras 3 (requirements.txt)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
import numpy as np
import torch
from pathlib import Path
from scipy import stats
from sacrebleu.metrics import BLEU
from transformers import AutoModelForMaskedLM, AutoTokenizer

DAMAGED_VAL_PATH = "evaluation/damaged_validation.npy"
TOKENIZER_NAME   = "zhihan1996/DNABERT-2-117M"
CHECKPOINTS = {
    "mlm_baseline": "training/checkpoints/mlm-baseline/checkpoint-595",
    "dam_proposed":  "training/checkpoints/dam-proposed/checkpoint-595",
}
OUT_JSON = "evaluation/results_table1.json"
OUT_TXT  = "evaluation/table1.txt"
N_BOOT   = 10_000
SEED     = 42
DEVICE   = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------
# Model loading
# ------------------------------------------------------------------

def load_model(ckpt_path: str, device: str):
    """Load fine-tuned weights into the DNABERT-2 architecture from HuggingFace cache."""
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    state = torch.load(
        Path(ckpt_path) / "pytorch_model.bin",
        map_location="cpu", weights_only=True,
    )
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Weight mismatch — missing: {missing}, unexpected: {unexpected}")
    model.eval()
    model.to(device)
    return model


# ------------------------------------------------------------------
# Inference
# ------------------------------------------------------------------

@torch.no_grad()
def predict_at_damaged_tokens(
    model,
    tokenizer,
    gt_ids:       list[int],
    tok_damaged:  list[int],   # token positions to mask
    attn_mask:    list[int],
    device:       str,
) -> dict[int, int]:
    """
    Mask the token positions in tok_damaged, run the model, return
    {token_position: predicted_token_id} for each masked position.
    """
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    masked  = gt_ids[:]
    for tpos in tok_damaged:
        masked[tpos] = mask_id

    inp  = torch.tensor([masked],   dtype=torch.long, device=device)
    attn = torch.tensor([attn_mask], dtype=torch.long, device=device)
    logits = model(input_ids=inp, attention_mask=attn).logits  # (1, seq_len, vocab)

    return {tpos: int(logits[0, tpos].argmax()) for tpos in tok_damaged}


# ------------------------------------------------------------------
# Nucleotide recovery at damage sites
# ------------------------------------------------------------------

def nucleotide_correct(
    gt_dna:      str,
    nuc_pos:     int,         # 0-based nucleotide position in gt_dna
    tok_pos:     int,         # token position in the original tokenization
    pred_tok_id: int,         # model's predicted token id
    offsets:     list,        # offset_mapping from tokenizer
    tokenizer,
) -> bool:
    """
    Returns True if the predicted token, decoded at the nucleotide sub-offset,
    matches the ground-truth nucleotide.
    """
    pred_token = tokenizer.convert_ids_to_tokens([pred_tok_id])[0]
    if pred_token is None:
        return False
    pred_nuc_str = pred_token.lstrip("#").lstrip("▁").upper()

    tok_start = offsets[tok_pos][0]
    nuc_offset_within_tok = nuc_pos - tok_start  # position within this BPE token

    if nuc_offset_within_tok < 0 or nuc_offset_within_tok >= len(pred_nuc_str):
        return False  # predicted token is shorter than expected

    return pred_nuc_str[nuc_offset_within_tok] == gt_dna[nuc_pos]


def window_recovery_rate(
    gt_dna:     str,
    nuc_damaged: list[int],
    tok_damaged: list[int],
    predictions: dict[int, int],   # tok_pos → pred_token_id
    offsets:    list,
    tokenizer,
) -> float | None:
    """
    Recovery rate for this window at damaged nucleotide positions.
    Returns NaN if no damaged positions.
    """
    if not nuc_damaged:
        return float("nan")

    # Build nuc_pos → tok_pos mapping (tok_damaged may be deduped)
    nuc_to_tok = {}
    for npos, tpos in zip(nuc_damaged, [
        next(t for t in tok_damaged if offsets[t][0] <= npos < offsets[t][1])
        for npos in nuc_damaged
    ]):
        nuc_to_tok[npos] = tpos

    correct = sum(
        nucleotide_correct(gt_dna, npos, nuc_to_tok[npos],
                           predictions[nuc_to_tok[npos]], offsets, tokenizer)
        for npos in nuc_damaged
        if nuc_to_tok[npos] in predictions
    )
    return correct / len(nuc_damaged)


# ------------------------------------------------------------------
# BLEU helper — decode full reconstructed sequence
# ------------------------------------------------------------------

def decode_full(gt_ids: list[int], tok_damaged: list[int], predictions: dict[int, int],
                tokenizer) -> str:
    """Decode the reconstructed token sequence to a space-separated character string."""
    reconstructed_ids = gt_ids[:]
    for tpos, pred_id in predictions.items():
        reconstructed_ids[tpos] = pred_id

    specials = {
        tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.unk_token, tokenizer.mask_token,
    }
    tokens = tokenizer.convert_ids_to_tokens(reconstructed_ids)
    dna = "".join(
        tok.lstrip("#").lstrip("▁").upper()
        for tok in tokens
        if tok not in specials and tok is not None
    )
    return " ".join(dna)   # space-separated chars for sacrebleu BLEU-4


# ------------------------------------------------------------------
# Bootstrap CI
# ------------------------------------------------------------------

def bootstrap_ci(diffs: np.ndarray, n: int, rng) -> tuple[float, float]:
    means = np.array([
        rng.choice(diffs, size=len(diffs), replace=True).mean()
        for _ in range(n)
    ])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows   = np.load(DAMAGED_VAL_PATH, allow_pickle=True).tolist()

    bleu_metric = BLEU(effective_order=True)
    results = {}

    for run_key, ckpt in CHECKPOINTS.items():
        print(f"\nEvaluating {run_key} ({ckpt}) on {DEVICE}...")
        model = load_model(ckpt, DEVICE)

        recovery_rates = []
        ref_strs = []
        hyp_strs = []

        for w in windows:
            tok_dam = w["tok_damaged"]
            nuc_dam = w["nuc_damaged"]
            offsets = [tuple(o) for o in w["offsets"]]   # json loads as list of lists

            if tok_dam:
                preds = predict_at_damaged_tokens(
                    model, tokenizer,
                    w["gt_ids"], tok_dam, w["attention_mask"], DEVICE,
                )
            else:
                preds = {}

            rate = window_recovery_rate(
                w["dna_gt"], nuc_dam, tok_dam, preds, offsets, tokenizer,
            )
            recovery_rates.append(rate)

            ref_strs.append(" ".join(w["dna_gt"]))
            hyp_strs.append(decode_full(w["gt_ids"], tok_dam, preds, tokenizer))

        bleu_score  = bleu_metric.corpus_score(hyp_strs, [ref_strs]).score
        valid_rates = [r for r in recovery_rates if not np.isnan(r)]

        results[run_key] = {
            "recovery_rates_per_window": recovery_rates,
            "valid_windows": len(valid_rates),
            "mean_recovery": float(np.mean(valid_rates))  if valid_rates else float("nan"),
            "std_recovery":  float(np.std(valid_rates))   if valid_rates else float("nan"),
            "bleu4":         float(bleu_score),
        }
        del model

    # Paired statistics
    r_mlm   = np.array(results["mlm_baseline"]["recovery_rates_per_window"])
    r_dam   = np.array(results["dam_proposed"]["recovery_rates_per_window"])
    paired  = ~(np.isnan(r_mlm) | np.isnan(r_dam))

    if paired.sum() >= 2:
        t_stat, p_val = stats.ttest_rel(r_dam[paired], r_mlm[paired])
        diffs         = r_dam[paired] - r_mlm[paired]
        mean_diff     = float(diffs.mean())
        ci_lo, ci_hi  = bootstrap_ci(diffs, N_BOOT, rng)
    else:
        t_stat = p_val = mean_diff = float("nan")
        ci_lo = ci_hi = float("nan")
        print("WARNING: fewer than 2 paired windows — t-test skipped.")

    paired_stats = {
        "paired_windows":          int(paired.sum()),
        "mean_diff_dam_minus_mlm": mean_diff,
        "t_stat":                  float(t_stat),
        "p_value":                 float(p_val),
        "bootstrap_ci_95":         [ci_lo, ci_hi],
    }

    output = {**results, "paired_stats": paired_stats}
    Path(OUT_JSON).write_text(json.dumps(output, indent=2))

    mlm = results["mlm_baseline"]
    dam = results["dam_proposed"]
    ps  = paired_stats

    table = (
        "\nTable 1 — Reconstruction evaluation on 69 validation windows\n"
        + "=" * 62 + "\n"
        + f"{'Metric':<38} {'MLM Baseline':>11} {'DAM Proposed':>11}\n"
        + "-" * 62 + "\n"
        + f"{'Nucleotide recovery rate (mean)':<38} "
        + f"{mlm['mean_recovery']:>11.4f} {dam['mean_recovery']:>11.4f}\n"
        + f"{'Nucleotide recovery rate (SD)':<38} "
        + f"{mlm['std_recovery']:>11.4f} {dam['std_recovery']:>11.4f}\n"
        + f"{'Windows with ≥1 damaged nucleotide':<38} "
        + f"{mlm['valid_windows']:>11} {dam['valid_windows']:>11}\n"
        + f"{'BLEU-4 (corpus, char-level)':<38} "
        + f"{mlm['bleu4']:>11.2f} {dam['bleu4']:>11.2f}\n"
        + "-" * 62 + "\n"
        + f"\nPaired statistics (DAM − MLM, n={ps['paired_windows']} windows):\n"
        + f"  Mean difference:   {ps['mean_diff_dam_minus_mlm']:+.4f}\n"
        + f"  t-statistic:       {ps['t_stat']:.4f}\n"
        + f"  p-value:           {ps['p_value']:.4f}\n"
        + f"  Bootstrap 95% CI:  [{ps['bootstrap_ci_95'][0]:+.4f}, "
        + f"{ps['bootstrap_ci_95'][1]:+.4f}]\n"
        + "=" * 62 + "\n"
    )

    print(table)
    Path(OUT_TXT).write_text(table)
    print(f"Results → {OUT_JSON}")
    print(f"Table   → {OUT_TXT}")


if __name__ == "__main__":
    main()
