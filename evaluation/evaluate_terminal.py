"""
Terminal-position reconstruction evaluation — VESTIGE Phase 3.

The DAM collator concentrates masking at terminal positions: C tokens near the
5' end and G tokens near the 3' end, matching the mapDamage2 deamination
profile. The background-damage evaluation (evaluate_reconstruction.py) found
0/79 damaged nucleotides in the terminal zone because authentic PMD rates
produce mostly background-position events. That evaluation correctly reflects
real-world damage statistics but does not test DAM's intended advantage.

This script evaluates BOTH models specifically at the terminal zone:
  - C tokens in the first T_END nucleotides from the 5' end of each window
  - G tokens in the first T_END nucleotides from the 3' end of each window

These are the positions DAM was specifically trained to reconstruct.

For each such position:
  1. Mask the token that spans it in the original gt_ids.
  2. Run each model; argmax at the masked position.
  3. Check if the predicted token's character at the correct sub-token offset
     matches the ground-truth nucleotide.

No in silico damage is applied — we evaluate the models' ability to fill in
terminal C/G positions from context, which is exactly the DAM training task.

Output:
  - evaluation/results_terminal.json
  - evaluation/table_terminal.txt

Run from project root:
    python evaluation/evaluate_terminal.py
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
from transformers import AutoModelForMaskedLM, AutoTokenizer

DAMAGED_VAL_PATH = "evaluation/damaged_validation.npy"
TOKENIZER_NAME   = "zhihan1996/DNABERT-2-117M"
CHECKPOINTS = {
    "mlm_baseline": "training/checkpoints/mlm-baseline/checkpoint-595",
    "dam_proposed":  "training/checkpoints/dam-proposed/checkpoint-595",
}
OUT_JSON = "evaluation/results_terminal.json"
OUT_TXT  = "evaluation/table_terminal.txt"

T_END  = 10   # evaluate C in first T_END nt from 5', G in last T_END nt from 3'
N_BOOT = 10_000
SEED   = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def load_model(ckpt_path, device):
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    state = torch.load(
        Path(ckpt_path) / "pytorch_model.bin", map_location="cpu", weights_only=True,
    )
    model.load_state_dict(state, strict=False)
    model.eval()
    model.to(device)
    return model


def find_terminal_positions(dna: str, offsets: list) -> tuple[list[tuple[int, int]], list[int]]:
    """
    Return:
      pairs     — list of (nuc_pos, tok_pos) for every terminal C/G nucleotide.
                  Multiple nucleotides in the same BPE token are all included.
      tok_positions — deduplicated list of token positions to mask (one mask per token).

    Terminal = C at nuc_pos < T_END (5') or G at nuc_pos >= len(dna) - T_END (3').
    """
    n = len(dna)
    pairs = []
    seen_tok = set()
    tok_positions = []

    for nuc_pos, base in enumerate(dna):
        is_terminal_C = base == "C" and nuc_pos < T_END
        is_terminal_G = base == "G" and nuc_pos >= n - T_END
        if not (is_terminal_C or is_terminal_G):
            continue

        tok_pos = None
        for t_idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start <= nuc_pos < end:
                tok_pos = t_idx
                break

        if tok_pos is not None:
            pairs.append((nuc_pos, tok_pos))   # all nucleotides, including shared tokens
            if tok_pos not in seen_tok:
                tok_positions.append(tok_pos)
                seen_tok.add(tok_pos)

    return pairs, tok_positions


@torch.no_grad()
def predict_at(model, tokenizer, gt_ids, tok_positions, attn_mask, device):
    """Mask tok_positions in gt_ids, run model, return {tok_pos: pred_token_id}."""
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    masked  = gt_ids[:]
    for tpos in tok_positions:
        masked[tpos] = mask_id

    logits = model(
        input_ids=torch.tensor([masked], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attn_mask], dtype=torch.long, device=device),
    ).logits
    return {tpos: int(logits[0, tpos].argmax()) for tpos in tok_positions}


def nucleotide_correct(gt_dna, nuc_pos, tok_pos, pred_id, offsets, tokenizer):
    pred_tok = tokenizer.convert_ids_to_tokens([pred_id])[0]
    if pred_tok is None:
        return False
    pred_str = pred_tok.lstrip("#").lstrip("▁").upper()
    offset = nuc_pos - offsets[tok_pos][0]
    if offset < 0 or offset >= len(pred_str):
        return False
    return pred_str[offset] == gt_dna[nuc_pos]


def bootstrap_ci(diffs, n, rng):
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows   = np.load(DAMAGED_VAL_PATH, allow_pickle=True).tolist()

    results = {}

    for run_key, ckpt in CHECKPOINTS.items():
        print(f"\nEvaluating {run_key} ({ckpt}) on {DEVICE}...")
        model = load_model(ckpt, DEVICE)

        per_window_rates   = []
        total_correct_all  = 0
        total_sites_all    = 0

        for w in windows:
            dna     = w["dna_gt"]
            offsets = [tuple(o) for o in w["offsets"]]
            pairs, tok_positions = find_terminal_positions(dna, offsets)

            if not pairs:
                per_window_rates.append(float("nan"))
                continue

            preds = predict_at(model, tokenizer, w["gt_ids"], tok_positions,
                               w["attention_mask"], device=DEVICE)

            correct = sum(
                nucleotide_correct(dna, npos, tpos, preds[tpos], offsets, tokenizer)
                for npos, tpos in pairs
            )
            rate = correct / len(pairs)
            per_window_rates.append(rate)
            total_correct_all += correct
            total_sites_all   += len(pairs)

        valid = [r for r in per_window_rates if not np.isnan(r)]
        results[run_key] = {
            "per_window_recovery":  per_window_rates,
            "valid_windows":        len(valid),
            "mean_recovery":        float(np.mean(valid))  if valid else float("nan"),
            "std_recovery":         float(np.std(valid))   if valid else float("nan"),
            "total_correct":        int(total_correct_all),
            "total_sites":          int(total_sites_all),
            "aggregate_recovery":   float(total_correct_all / total_sites_all)
                                    if total_sites_all else float("nan"),
        }
        print(f"  Sites evaluated: {total_sites_all}, "
              f"correct: {total_correct_all}, "
              f"aggregate: {results[run_key]['aggregate_recovery']:.4f}")
        del model

    # Paired statistics
    r_mlm  = np.array(results["mlm_baseline"]["per_window_recovery"])
    r_dam  = np.array(results["dam_proposed"]["per_window_recovery"])
    paired = ~(np.isnan(r_mlm) | np.isnan(r_dam))

    if paired.sum() >= 2:
        t_stat, p_val = stats.ttest_rel(r_dam[paired], r_mlm[paired])
        diffs         = r_dam[paired] - r_mlm[paired]
        mean_diff     = float(diffs.mean())
        ci_lo, ci_hi  = bootstrap_ci(diffs, N_BOOT, rng)
    else:
        t_stat = p_val = mean_diff = float("nan")
        ci_lo = ci_hi = float("nan")

    paired_stats = {
        "paired_windows":          int(paired.sum()),
        "mean_diff_dam_minus_mlm": mean_diff,
        "t_stat":                  float(t_stat),
        "p_value":                 float(p_val),
        "bootstrap_ci_95":         [ci_lo, ci_hi],
    }

    output = {**results, "paired_stats": paired_stats, "terminal_window_nt": T_END}
    Path(OUT_JSON).write_text(json.dumps(output, indent=2))

    mlm = results["mlm_baseline"]
    dam = results["dam_proposed"]
    ps  = paired_stats

    table = (
        f"\nTable 2 — Terminal-position reconstruction ({T_END} nt from 5'/3' ends)\n"
        + "=" * 66 + "\n"
        + f"{'Metric':<42} {'MLM Baseline':>11} {'DAM Proposed':>11}\n"
        + "-" * 66 + "\n"
        + f"{'Nucleotide recovery (mean over windows)':<42} "
        + f"{mlm['mean_recovery']:>11.4f} {dam['mean_recovery']:>11.4f}\n"
        + f"{'Nucleotide recovery (SD)':<42} "
        + f"{mlm['std_recovery']:>11.4f} {dam['std_recovery']:>11.4f}\n"
        + f"{'Aggregate (correct / total sites)':<42} "
        + f"{mlm['aggregate_recovery']:>11.4f} {dam['aggregate_recovery']:>11.4f}\n"
        + f"{'Total sites evaluated':<42} "
        + f"{mlm['total_sites']:>11} {dam['total_sites']:>11}\n"
        + f"{'Windows contributing':<42} "
        + f"{mlm['valid_windows']:>11} {dam['valid_windows']:>11}\n"
        + "-" * 66 + "\n"
        + f"\nPaired statistics (DAM − MLM, n={ps['paired_windows']} windows):\n"
        + f"  Mean difference:   {ps['mean_diff_dam_minus_mlm']:+.4f}\n"
        + f"  t-statistic:       {ps['t_stat']:.4f}\n"
        + f"  p-value:           {ps['p_value']:.4f}\n"
        + f"  Bootstrap 95% CI:  [{ps['bootstrap_ci_95'][0]:+.4f}, "
        + f"{ps['bootstrap_ci_95'][1]:+.4f}]\n"
        + "=" * 66 + "\n"
    )

    print(table)
    Path(OUT_TXT).write_text(table)
    print(f"Results → {OUT_JSON}")
    print(f"Table   → {OUT_TXT}")


if __name__ == "__main__":
    main()
