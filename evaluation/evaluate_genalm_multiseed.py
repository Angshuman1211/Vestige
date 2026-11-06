"""
GENA-LM multi-seed eval — terminal AND background, seeds 42/1/2 (P27 final box).

Closes the last GENA-LM caveat: the +0.10 terminal / +0.01 background result was
single-seed (42). This re-runs both metrics across three seeds and reports, per
zone:
  - each seed's Delta(DAM-MLM) (consistency of sign),
  - a POOLED number = the seed-averaged per-window Delta tested against 0 with a
    one-sample t-test + Wilcoxon + 10k-bootstrap CI (windows as the replication
    unit, the same pooling the DNABERT-2 multi-seed benchmark uses).

Window tokenization is model-independent (same GENA-LM tokenizer), so the set of
scored positions is identical across seeds; only the predictions differ. Each
model is loaded once and scored on both zones.

Run from project root:
    python evaluation/evaluate_genalm_multiseed.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from transformers import AutoModel, AutoTokenizer

GENALM = "AIRI-Institute/gena-lm-bert-base-t2t"
MODEL_ROOT = "training/genalm_models"
SEEDS = [42, 1, 2]
ARMS = ["mlm", "dam"]          # folder = f"{arm-or-dam-offset}-genalm-s{seed}"
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
WIDTHS = [3, 5, 10]            # T_END for terminal, half-band for background
N_BOOT = 10_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_genalm_multiseed.json"


def folder(arm, seed):
    stem = "mlm" if arm == "mlm" else "dam-offset"
    return f"{MODEL_ROOT}/{stem}-genalm-s{seed}"


def load_model(ckpt):
    m = AutoModel.from_pretrained(GENALM, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


def find_terminal(dna, offsets, w):
    n = len(dna); pairs = []; seen = set(); toks = []
    for pos, base in enumerate(dna):
        if not ((base == "C" and pos < w) or (base == "G" and pos >= n - w)):
            continue
        for ti, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s <= pos < e:
                pairs.append((pos, ti))
                if ti not in seen:
                    toks.append(ti); seen.add(ti)
                break
    return pairs, toks


def find_background(dna, offsets, w):
    n = len(dna); mid = n // 2; lo, hi = mid - w, mid + w
    pairs = []; seen = set(); toks = []
    for pos, base in enumerate(dna):
        if base not in ("C", "G"):
            continue
        if not (lo <= pos < hi and w <= pos < n - w):
            continue
        for ti, (s, e) in enumerate(offsets):
            if s == e:
                continue
            if s <= pos < e:
                pairs.append((pos, ti))
                if ti not in seen:
                    toks.append(ti); seen.add(ti)
                break
    return pairs, toks


@torch.no_grad()
def predict_at(model, mask_id, ids, toks, attn):
    masked = list(ids)
    for t in toks:
        masked[t] = mask_id
    logits = model(input_ids=torch.tensor([masked], dtype=torch.long, device=DEVICE),
                   attention_mask=torch.tensor([attn], dtype=torch.long, device=DEVICE)).logits
    return {t: int(logits[0, t].argmax()) for t in toks}


def nuc_correct(dna, pos, tok, pid, offsets, tokenizer):
    s = tokenizer.convert_ids_to_tokens([pid])[0]
    if not s:
        return False
    s = s.lstrip("#").lstrip("▁").upper()
    off = pos - offsets[tok][0]
    return 0 <= off < len(s) and s[off] == dna[pos]


def recovery_vector(model, tok, mask_id, enc, finder, w):
    """Per-window recovery rate for one model / zone / width (nan if no positions)."""
    out = []
    for dna, ids, attn, offsets in enc:
        pairs, toks = finder(dna, offsets, w)
        if not pairs:
            out.append(float("nan")); continue
        preds = predict_at(model, mask_id, ids, toks, attn)
        correct = sum(nuc_correct(dna, p, ti, preds[ti], offsets, tok)
                      for p, ti in pairs if ti in preds)
        out.append(correct / len(pairs))
    return np.array(out)


def pooled_stats(per_seed_delta, rng):
    """per_seed_delta: list of per-window Delta arrays (one per seed), same nan pattern.
    Returns seed-averaged per-window Delta stats."""
    stack = np.vstack(per_seed_delta)                 # (seeds, windows)
    mask = ~np.isnan(stack).any(axis=0)
    avg = stack[:, mask].mean(axis=0)                  # seed-averaged per-window Delta
    t_stat, p_t = stats.ttest_1samp(avg, 0.0)
    try:
        _, p_w = stats.wilcoxon(avg)
    except ValueError:
        p_w = float("nan")
    boot = np.array([rng.choice(avg, size=len(avg), replace=True).mean() for _ in range(N_BOOT)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    return {"delta": float(avg.mean()), "t": float(t_stat), "p_t": float(p_t),
            "p_wilcoxon": float(p_w), "ci95": ci, "n": int(mask.sum())}


def main():
    tok = AutoTokenizer.from_pretrained(GENALM, trust_remote_code=True)
    mask_id = tok.convert_tokens_to_ids(tok.mask_token)
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    print(f"{len(windows)} eval windows; re-tokenizing with GENA-LM...", flush=True)
    enc = []
    for wn in windows:
        dna = wn["dna_gt"]
        e = tok(dna, truncation=True, max_length=512, return_offsets_mapping=True)
        enc.append((dna, e["input_ids"], e["attention_mask"],
                    [tuple(o) for o in e["offset_mapping"]]))

    zones = {"terminal": find_terminal, "background": find_background}
    # rec[zone][width][arm][seed] = per-window recovery array
    rec = {z: {w: {a: {} for a in ARMS} for w in WIDTHS} for z in zones}
    for seed in SEEDS:
        for arm in ARMS:
            print(f"scoring {arm} seed {seed} on {DEVICE}...", flush=True)
            model = load_model(folder(arm, seed))
            for z, finder in zones.items():
                for w in WIDTHS:
                    rec[z][w][arm][seed] = recovery_vector(model, tok, mask_id, enc, finder, w)
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    rng = np.random.default_rng(42)
    out = {}
    for z in zones:
        out[z] = {}
        label = "TERMINAL (damaged ends)" if z == "terminal" else "BACKGROUND (central band)"
        print(f"\n=== GENA-LM {label}: DAM-offset - MLM, 3 seeds ===")
        hdr = f"  {'w':>3} " + " ".join(f"{'Δs'+str(s):>8}" for s in SEEDS) + \
              f" {'POOLED Δ':>9} {'p_t':>9} {'p_wilcox':>9} {'95% CI':>20} {'n':>5}"
        print(hdr)
        for w in WIDTHS:
            per_seed_delta = []
            seed_means = []
            for seed in SEEDS:
                d = rec[z][w]["dam"][seed] - rec[z][w]["mlm"][seed]
                per_seed_delta.append(d)
                seed_means.append(np.nanmean(d))
            ps = pooled_stats(per_seed_delta, rng)
            out[z][str(w)] = {"per_seed_delta": [float(x) for x in seed_means], **ps}
            seedcol = " ".join(f"{m:>+8.4f}" for m in seed_means)
            print(f"  {w:>3} {seedcol} {ps['delta']:>+9.4f} {ps['p_t']:>9.2g} "
                  f"{ps['p_wilcoxon']:>9.2g} [{ps['ci95'][0]:+.4f},{ps['ci95'][1]:+.4f}] {ps['n']:>5}")

    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    tt = out["terminal"]; bg = out["background"]
    term_ok = all(tt[str(w)]["delta"] > 0 and tt[str(w)]["p_t"] < 0.05 for w in WIDTHS)
    bg_ok = min(bg[str(w)]["delta"] for w in WIDTHS) > -0.03
    print("\nVERDICT:")
    print(f"  terminal: {'DAM > MLM, pooled & every seed-sign' if term_ok else 'mixed'} "
          f"(pooled Δ @3/5/10 = {tt['3']['delta']:+.3f}/{tt['5']['delta']:+.3f}/{tt['10']['delta']:+.3f})")
    print(f"  background: {'no collapse' if bg_ok else 'cost detected'} "
          f"(pooled Δ @3/5/10 = {bg['3']['delta']:+.3f}/{bg['5']['delta']:+.3f}/{bg['10']['delta']:+.3f})")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
