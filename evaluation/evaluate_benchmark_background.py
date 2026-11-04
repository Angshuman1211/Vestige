"""
Background / authentic-damage reconstruction benchmark across all arms.

Complement to evaluate_benchmark.py (terminal zone). Here I evaluate every arm at
the AUTHENTIC damage positions: mapDamage2 PMD rates place almost all of these in
the sequence interior, not the terminal zone. I also report corpus BLEU-4 over the
fully reconstructed sequence.

The hypothesis this tests: an arm that concentrates its whole mask budget on the
terminal zone (pmd-mask, and to a lesser degree the sharpened DAM variants) should
lose here, where a balanced arm (DAM) holds both zones. If that holds, DAM is the
best terminal/background tradeoff even though it is not the single best at either.

Run from project root:
    python evaluation/evaluate_benchmark_background.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from sacrebleu.metrics import BLEU
from transformers import AutoTokenizer

# Reuse loaders/helpers so the metric matches the originals exactly
from evaluate_terminal import load_model, TOKENIZER_NAME, DAMAGED_VAL_PATH, DEVICE
from evaluate_reconstruction import (
    predict_at_damaged_tokens,
    window_recovery_rate,
    decode_full,
    bootstrap_ci,
)

BENCH_DIR = "training/benchmark_models"
CHECKPOINTS = {
    "mlm":         f"{BENCH_DIR}/mlm-baseline",
    "span":        f"{BENCH_DIR}/span-baseline",
    "pmdmask":     f"{BENCH_DIR}/pmdmask-baseline",
    "dam":         f"{BENCH_DIR}/dam-proposed",
    "dam_noscale": f"{BENCH_DIR}/dam-noscale",
    "dam_atmask":  f"{BENCH_DIR}/dam-atmask",
    "dam_smooth":  f"{BENCH_DIR}/dam-smooth",
    "dam_power":   f"{BENCH_DIR}/dam-power2",
}
REFERENCE = "dam"
N_BOOT = 10_000
SEED   = 42

OUT_JSON = "evaluation/results_benchmark_background.json"
OUT_TXT  = "evaluation/table_benchmark_background.txt"

STRATEGY_ARMS = ["mlm", "span", "pmdmask", "dam"]
ABLATION_ARMS = ["dam", "dam_noscale", "dam_atmask", "dam_smooth", "dam_power"]
LABELS = {
    "mlm": "MLM (uniform 15%)",
    "span": "SpanBERT",
    "pmdmask": "pmd-mask (hard)",
    "dam": "DAM (proposed)",
    "dam_noscale": "DAM, no rescale",
    "dam_atmask": "DAM, A/T baseline",
    "dam_smooth": "DAM, smoothed",
    "dam_power": "DAM, power map",
}


def eval_arm(model, tokenizer, windows, bleu_metric):
    """Authentic-damage recovery per window + corpus BLEU-4 for one model."""
    recovery = []
    ref_strs, hyp_strs = [], []
    for w in windows:
        tok_dam = w["tok_damaged"]
        nuc_dam = w["nuc_damaged"]
        offsets = [tuple(o) for o in w["offsets"]]
        preds = (predict_at_damaged_tokens(model, tokenizer, w["gt_ids"],
                                           tok_dam, w["attention_mask"], DEVICE)
                 if tok_dam else {})
        recovery.append(window_recovery_rate(w["dna_gt"], nuc_dam, tok_dam,
                                             preds, offsets, tokenizer))
        ref_strs.append(" ".join(w["dna_gt"]))
        hyp_strs.append(decode_full(w["gt_ids"], tok_dam, preds, tokenizer))
    bleu = bleu_metric.corpus_score(hyp_strs, [ref_strs]).score
    return recovery, float(bleu)


def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows   = np.load(DAMAGED_VAL_PATH, allow_pickle=True).tolist()
    bleu_metric = BLEU(effective_order=True)

    results = {}
    for key, ckpt in CHECKPOINTS.items():
        if not (Path(ckpt) / "pytorch_model.bin").exists():
            print(f"skip {key}: no weights at {ckpt}")
            continue
        print(f"\nEvaluating {key} ({ckpt}) on {DEVICE}...")
        model = load_model(ckpt, DEVICE)
        recovery, bleu = eval_arm(model, tokenizer, windows, bleu_metric)
        valid = [r for r in recovery if not np.isnan(r)]
        results[key] = {
            "recovery_per_window": recovery,
            "valid_windows":       len(valid),
            "mean_recovery":       float(np.mean(valid)) if valid else float("nan"),
            "std_recovery":        float(np.std(valid))  if valid else float("nan"),
            "bleu4":               bleu,
        }
        print(f"  windows: {len(valid)}, mean recovery: {results[key]['mean_recovery']:.4f}, BLEU-4: {bleu:.2f}")
        del model

    paired = {}
    if REFERENCE in results:
        r_ref = np.array(results[REFERENCE]["recovery_per_window"])
        for key in results:
            if key == REFERENCE:
                continue
            r_arm = np.array(results[key]["recovery_per_window"])
            mask  = ~(np.isnan(r_arm) | np.isnan(r_ref))
            if mask.sum() >= 2:
                diffs = r_ref[mask] - r_arm[mask]        # dam - arm (positive => DAM better)
                t_stat, p_val = stats.ttest_rel(r_ref[mask], r_arm[mask])
                ci_lo, ci_hi  = bootstrap_ci(diffs, N_BOOT, rng)
                paired[key] = {
                    "paired_windows":      int(mask.sum()),
                    "mean_diff_dam_minus": float(diffs.mean()),
                    "t_stat":              float(t_stat),
                    "p_value":             float(p_val),
                    "bootstrap_ci_95":     [ci_lo, ci_hi],
                }

    Path(OUT_JSON).write_text(json.dumps(
        {"arms": results, "paired_vs_dam": paired}, indent=2))

    def row(key):
        r = results[key]
        d = paired.get(key)
        diff = f"{d['mean_diff_dam_minus']:+.4f}" if d else ("  ref" if key == REFERENCE else "   -")
        pv   = f"{d['p_value']:.3g}" if d else ""
        return (f"{LABELS.get(key, key):<22} "
                f"{r['mean_recovery']:>8.4f} {r['std_recovery']:>7.4f} "
                f"{r['bleu4']:>8.2f} {r['valid_windows']:>7} "
                f"{diff:>9} {pv:>8}")

    header = (f"{'Arm':<22} {'Recov':>8} {'SD':>7} {'BLEU-4':>8} {'Wins':>7} "
              f"{'DAM-arm':>9} {'p':>8}")
    lines = ["\nBackground (authentic-damage) reconstruction benchmark",
             "=" * 74, header, "-" * 74, "Masking strategies:"]
    for k in STRATEGY_ARMS:
        if k in results:
            lines.append("  " + row(k))
    lines.append("DAM design ablations:")
    for k in ABLATION_ARMS:
        if k != "dam" and k in results:
            lines.append("  " + row(k))
    lines.append("=" * 74)
    lines.append("DAM-arm > 0 and p < 0.05 means DAM recovers authentic damage better than that arm.")
    table = "\n".join(lines) + "\n"

    print(table)
    Path(OUT_TXT).write_text(table)
    print(f"Results -> {OUT_JSON}")
    print(f"Table   -> {OUT_TXT}")


if __name__ == "__main__":
    main()
