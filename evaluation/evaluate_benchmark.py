"""
Masking-strategy benchmark — terminal-zone reconstruction across all arms.

I evaluate every trained arm on the same terminal C/G reconstruction task used in
evaluate_terminal.py (C in the first T_END nt from the 5' end, G in the last
T_END nt from the 3' end), then rank the arms and report paired statistics of
each arm against DAM.

Arms:
  masking strategies : mlm, span, pmdmask, dam   (pmi added once it is trained)
  DAM design ablations: dam vs dam_noscale, dam_atmask, dam_smooth, dam_power

All weights live flat under BENCH_DIR as <name>/pytorch_model.bin, trained in
one environment (Colab T4) so the arms are comparable.

Run from project root:
    python evaluation/evaluate_benchmark.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoTokenizer

# Reuse the exact terminal-zone machinery so the metric matches evaluate_terminal.py
from evaluate_terminal import (
    load_model,
    find_terminal_positions,
    predict_at,
    nucleotide_correct,
    bootstrap_ci,
    TOKENIZER_NAME,
    DAMAGED_VAL_PATH,
    T_END,
    DEVICE,
)

BENCH_DIR = "training/benchmark_models"
# Only arms whose pytorch_model.bin is present are evaluated; the rest are skipped
# with a warning, so this runs before pmi (or any arm) is finished.
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
REFERENCE = "dam"   # paired stats are computed as (arm - dam)

N_BOOT = 10_000
SEED   = 42

OUT_JSON = "evaluation/results_benchmark.json"
OUT_TXT  = "evaluation/table_benchmark.txt"

# Human-readable grouping for the table
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


def eval_arm(model, tokenizer, windows):
    """Terminal NRR for one model: per-window recovery list + aggregate over sites."""
    per_window   = []
    total_correct = 0
    total_sites   = 0
    for w in windows:
        dna     = w["dna_gt"]
        offsets = [tuple(o) for o in w["offsets"]]
        pairs, tok_positions = find_terminal_positions(dna, offsets)
        if not pairs:
            per_window.append(float("nan"))
            continue
        preds = predict_at(model, tokenizer, w["gt_ids"], tok_positions,
                           w["attention_mask"], device=DEVICE)
        correct = sum(
            nucleotide_correct(dna, npos, tpos, preds[tpos], offsets, tokenizer)
            for npos, tpos in pairs
        )
        per_window.append(correct / len(pairs))
        total_correct += correct
        total_sites   += len(pairs)
    return per_window, total_correct, total_sites


def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows   = np.load(DAMAGED_VAL_PATH, allow_pickle=True).tolist()

    results = {}
    for key, ckpt in CHECKPOINTS.items():
        if not (Path(ckpt) / "pytorch_model.bin").exists():
            print(f"skip {key}: no weights at {ckpt}")
            continue
        print(f"\nEvaluating {key} ({ckpt}) on {DEVICE}...")
        model = load_model(ckpt, DEVICE)
        per_window, tc, ts = eval_arm(model, tokenizer, windows)
        valid = [r for r in per_window if not np.isnan(r)]
        results[key] = {
            "per_window_recovery": per_window,
            "valid_windows":       len(valid),
            "mean_recovery":       float(np.mean(valid)) if valid else float("nan"),
            "std_recovery":        float(np.std(valid))  if valid else float("nan"),
            "total_correct":       int(tc),
            "total_sites":         int(ts),
            "aggregate_recovery":  float(tc / ts) if ts else float("nan"),
        }
        print(f"  sites: {ts}, correct: {tc}, aggregate: {results[key]['aggregate_recovery']:.4f}")
        del model

    # Paired stats of every evaluated arm against the reference (DAM)
    paired = {}
    if REFERENCE in results:
        r_ref = np.array(results[REFERENCE]["per_window_recovery"])
        for key in results:
            if key == REFERENCE:
                continue
            r_arm = np.array(results[key]["per_window_recovery"])
            mask  = ~(np.isnan(r_arm) | np.isnan(r_ref))
            if mask.sum() >= 2:
                diffs = r_ref[mask] - r_arm[mask]     # dam - arm (positive => DAM better)
                t_stat, p_val = stats.ttest_rel(r_ref[mask], r_arm[mask])
                ci_lo, ci_hi  = bootstrap_ci(diffs, N_BOOT, rng)
                paired[key] = {
                    "paired_windows":       int(mask.sum()),
                    "mean_diff_dam_minus":  float(diffs.mean()),
                    "t_stat":               float(t_stat),
                    "p_value":              float(p_val),
                    "bootstrap_ci_95":      [ci_lo, ci_hi],
                }

    output = {"arms": results, "paired_vs_dam": paired, "terminal_window_nt": T_END}
    Path(OUT_JSON).write_text(json.dumps(output, indent=2))

    # ---- table ----
    def row(key):
        r = results[key]
        d = paired.get(key)
        diff = f"{d['mean_diff_dam_minus']:+.4f}" if d else "  ref" if key == REFERENCE else "   -"
        pv   = f"{d['p_value']:.3g}" if d else "" if key == REFERENCE else ""
        return (f"{LABELS.get(key, key):<22} "
                f"{r['aggregate_recovery']:>8.4f} "
                f"{r['mean_recovery']:>8.4f} "
                f"{r['std_recovery']:>7.4f} "
                f"{r['total_sites']:>7} "
                f"{diff:>9} {pv:>8}")

    header = (f"{'Arm':<22} {'Aggr':>8} {'Mean':>8} {'SD':>7} {'Sites':>7} "
              f"{'DAM-arm':>9} {'p':>8}")

    lines = ["\nTerminal-zone reconstruction benchmark "
             f"({T_END} nt from 5'/3' ends)", "=" * 72, header, "-" * 72]
    lines.append("Masking strategies:")
    for k in STRATEGY_ARMS:
        if k in results:
            lines.append("  " + row(k))
    lines.append("DAM design ablations:")
    for k in ABLATION_ARMS:
        if k == "dam":
            continue
        if k in results:
            lines.append("  " + row(k))
    lines.append("=" * 72)
    lines.append("DAM-arm > 0 and p < 0.05 means DAM reconstructs terminal C/G better than that arm.")
    table = "\n".join(lines) + "\n"

    print(table)
    Path(OUT_TXT).write_text(table)
    print(f"Results -> {OUT_JSON}")
    print(f"Table   -> {OUT_TXT}")


if __name__ == "__main__":
    main()
