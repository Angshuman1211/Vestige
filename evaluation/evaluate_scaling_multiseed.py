"""
Multi-seed version of the 626-window T_END sweep (the paper headline).

evaluate_scaling.py establishes the central result (DAM >> MLM at terminal,
p<0.001, with MLM below random at T_END=3) on a SINGLE seed. Before the depth
studies, I confirm that result holds across the 3 training seeds — if the
headline does not survive multi-seed, the whole paper changes.

For each seed I reload mlm-baseline-s<seed> and dam-proposed-s<seed> (from
training/multiseed_models/), rerun the identical terminal-recovery machinery at
every T_END, and report: per-seed Δ(DAM-MLM) and paired p, plus a POOLED
seed-averaged per-window test (windows as the replication unit) for one
defensible p-value per T_END.

Zero-shot and random are seed-independent (fixed references in the paper) and are
not recomputed here; this run is strictly the MLM-vs-DAM robustness check.

Run from project root:
    python evaluation/evaluate_scaling_multiseed.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModelForMaskedLM, AutoTokenizer

# Self-contained copies of the evaluate_scaling.py helpers. I do NOT import that
# module because it pulls in matplotlib at load time, which is broken under the
# numpy-2 migration in this env. The functions below are byte-for-byte the same.
TOKENIZER_NAME    = "zhihan1996/DNABERT-2-117M"
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
T_ENDS = [3, 5, 10, 15, 20, 25]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ckpt, device):
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def find_terminal(dna, offsets, t_end):
    n = len(dna)
    pairs, seen, tok_positions = [], set(), []
    for nuc_pos, base in enumerate(dna):
        if not ((base == "C" and nuc_pos < t_end) or
                (base == "G" and nuc_pos >= n - t_end)):
            continue
        for tok_idx, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s <= nuc_pos < e:
                pairs.append((nuc_pos, tok_idx))
                if tok_idx not in seen:
                    tok_positions.append(tok_idx)
                    seen.add(tok_idx)
                break
    return pairs, tok_positions


@torch.no_grad()
def predict_at(model, tokenizer, gt_ids, tok_positions, attn_mask, device):
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    masked = gt_ids[:]
    for tp in tok_positions:
        masked[tp] = mask_id
    logits = model(
        input_ids=torch.tensor([masked], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attn_mask], dtype=torch.long, device=device),
    ).logits
    preds = {tp: int(logits[0, tp].argmax()) for tp in tok_positions}
    return preds, None


def nuc_correct(dna, nuc_pos, tok_pos, pred_id, offsets, tokenizer):
    tok = tokenizer.convert_ids_to_tokens([pred_id])[0]
    if not tok:
        return False
    s = tok.lstrip("#").lstrip("▁").upper()
    off = nuc_pos - offsets[tok_pos][0]
    return 0 <= off < len(s) and s[off] == dna[nuc_pos]


def bootstrap_ci(diffs, n, rng):
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


MODEL_ROOT = "training/multiseed_models"
SEEDS = [42, 1, 2]
N_BOOT = 10_000
OUT_JSON = "evaluation/scaling_multiseed.json"
OUT_TXT  = "evaluation/table_scaling_multiseed.txt"


def per_window_rates(model, tokenizer, windows, t_end):
    """Per-window terminal recovery at t_end. Window set is seed-independent
    (find_terminal uses only the sequence), so the arrays align across models."""
    rates = []
    for w in windows:
        offsets = [tuple(o) for o in w["offsets"]]
        pairs, tok_positions = find_terminal(w["dna_gt"], offsets, t_end)
        if not pairs:
            continue
        preds, _ = predict_at(model, tokenizer, w["gt_ids"], tok_positions,
                              w["attention_mask"], DEVICE)
        correct = sum(nuc_correct(w["dna_gt"], np_, tp, preds[tp], offsets, tokenizer)
                      for np_, tp in pairs if tp in preds)
        rates.append(correct / len(pairs))
    return np.array(rates)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    print(f"{len(windows)} windows total", flush=True)

    # per_seed_diffs[t_end][seed] = per-window (DAM - MLM) array
    per_seed = {t: {} for t in T_ENDS}
    diffs_by_t = {t: {} for t in T_ENDS}

    for seed in SEEDS:
        print(f"\n=== seed {seed}: loading models ===", flush=True)
        mlm = load_model(f"{MODEL_ROOT}/mlm-baseline-s{seed}", DEVICE)
        dam = load_model(f"{MODEL_ROOT}/dam-proposed-s{seed}", DEVICE)
        rng = np.random.default_rng(42)
        for t in T_ENDS:
            r_mlm = per_window_rates(mlm, tokenizer, windows, t)
            r_dam = per_window_rates(dam, tokenizer, windows, t)
            diffs = r_dam - r_mlm
            t_stat, p2 = stats.ttest_rel(r_dam, r_mlm)
            ci = bootstrap_ci(diffs, N_BOOT, rng)
            per_seed[t][seed] = {
                "mlm": float(r_mlm.mean()), "dam": float(r_dam.mean()),
                "delta": float(diffs.mean()), "t": float(t_stat), "p": float(p2),
                "ci": ci, "n": int(len(r_mlm)),
            }
            diffs_by_t[t][seed] = diffs
            print(f"  seed {seed}  T_END {t:>2}:  Δ={diffs.mean():+.4f}  "
                  f"p={p2:.2g}  (MLM {r_mlm.mean():.3f} / DAM {r_dam.mean():.3f}, n={len(r_mlm)})",
                  flush=True)
        del mlm, dam
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # ---- pooled seed-averaged per-window test, per T_END ----
    pooled = {}
    for t in T_ENDS:
        stacked = np.vstack([diffs_by_t[t][s] for s in SEEDS])   # (n_seeds, n_windows)
        d_avg = stacked.mean(axis=0)                              # seed-averaged per window
        t_stat, p2 = stats.ttest_1samp(d_avg, 0.0)
        try:
            _, wil = stats.wilcoxon(d_avg)
        except Exception:
            wil = float("nan")
        ci = bootstrap_ci(d_avg, N_BOOT, np.random.default_rng(42))
        pooled[t] = {
            "n_windows": int(len(d_avg)), "mean_delta": float(d_avg.mean()),
            "t_p": float(p2), "wilcoxon_p": float(wil), "ci95": ci,
            "per_seed_delta": [per_seed[t][s]["delta"] for s in SEEDS],
            "per_seed_p": [per_seed[t][s]["p"] for s in SEEDS],
            "all_seeds_sig_001": bool(all(per_seed[t][s]["p"] < 0.001 for s in SEEDS)),
            "all_seeds_positive": bool(all(per_seed[t][s]["delta"] > 0 for s in SEEDS)),
        }

    Path(OUT_JSON).write_text(json.dumps(
        {"per_seed": {str(t): per_seed[t] for t in T_ENDS},
         "pooled": {str(t): pooled[t] for t in T_ENDS}}, indent=2))

    # ---- table ----
    L = ["\nMulti-seed 626-window T_END sweep — DAM vs MLM (headline robustness)",
         "=" * 92,
         f"{'T_END':>6} | {'seed42 Δ (p)':>16} {'seed1 Δ (p)':>16} {'seed2 Δ (p)':>16} | "
         f"{'pooled Δ':>9} {'t p':>8} {'wilcox':>8} {'all<.001?':>9}",
         "-" * 92]
    for t in T_ENDS:
        cells = " ".join(f"{per_seed[t][s]['delta']:+.3f}({per_seed[t][s]['p']:.1g})".rjust(16)
                         for s in SEEDS)
        pt = pooled[t]
        L.append(f"{t:>5}n | {cells} | {pt['mean_delta']:>+9.4f} {pt['t_p']:>8.2g} "
                 f"{pt['wilcoxon_p']:>8.2g} {str(pt['all_seeds_sig_001']):>9}")
    L.append("=" * 92)
    L.append("Headline survives iff Δ>0 and significant across ALL seeds, esp. at T_END=3/5/10.")
    table = "\n".join(L) + "\n"
    print(table)
    Path(OUT_TXT).write_text(table)
    print(f"JSON  -> {OUT_JSON}")
    print(f"Table -> {OUT_TXT}")


if __name__ == "__main__":
    main()
