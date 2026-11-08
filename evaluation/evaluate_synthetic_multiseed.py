"""
D3 synthetic corruption-shape sweep — multi-seed eval (DNABERT-2).

For each arbitrary corruption profile (central, periodic), the DAM-matched arm was
trained to mask C/G by that profile's per-position weight. This eval defines the
profile's CORRUPTION ZONE = positions with w[p] >= THRESH*max(w), masks the C/G
tokens there, and measures recovery of the DAM-matched arm vs the SAME uniform-MLM
baseline (mlm-baseline-s*, reused — uniform masking is profile-agnostic). Pooled
stats identical to the backbone evals (seed-averaged per-window Delta, one-sample t
+ Wilcoxon + 10k-bootstrap CI).

Claim tested: a DAM arm matched to an un-terminal shape still beats uniform MLM at
recovering that shape's damaged positions -> the principle is corruption-shape
-agnostic, not an ends-are-special artifact.

Run from project root:
    python evaluation/evaluate_synthetic_multiseed.py
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

DNABERT2 = "zhihan1996/DNABERT-2-117M"
MODEL_ROOT = "training/multiseed_models"
SEEDS = [42, 1, 2]
PROFILES = ["central", "periodic"]
PROFILE_DIR = "damage/synthetic"
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
THRESH = 0.5                    # corruption zone = w[p] >= THRESH * max(w)
N_BOOT = 10_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_synthetic_multiseed.json"


def folder(arm, profile, seed):
    return f"{MODEL_ROOT}/mlm-baseline-s{seed}" if arm == "mlm" else f"{MODEL_ROOT}/syn-{profile}-s{seed}"


def load_model(ckpt):
    m = AutoModelForMaskedLM.from_pretrained(DNABERT2, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


def find_zone(dna, offsets, zone_pos):
    """C/G positions inside the profile's high-weight zone, and the covering tokens."""
    pairs = []; seen = set(); toks = []
    for pos, base in enumerate(dna):
        if base not in ("C", "G") or pos not in zone_pos:
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


def recovery_vector(model, tok, mask_id, enc, zone_pos):
    out = []
    for dna, ids, attn, offsets in enc:
        pairs, toks = find_zone(dna, offsets, zone_pos)
        if not pairs:
            out.append(float("nan")); continue
        preds = predict_at(model, mask_id, ids, toks, attn)
        correct = sum(nuc_correct(dna, p, ti, preds[ti], offsets, tok)
                      for p, ti in pairs if ti in preds)
        out.append(correct / len(pairs))
    return np.array(out)


def pooled_stats(per_seed_delta, rng):
    stack = np.vstack(per_seed_delta)
    mask = ~np.isnan(stack).any(axis=0)
    avg = stack[:, mask].mean(axis=0)
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
    tok = AutoTokenizer.from_pretrained(DNABERT2, trust_remote_code=True)
    mask_id = tok.convert_tokens_to_ids(tok.mask_token)
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    print(f"{len(windows)} eval windows; tokenizing with DNABERT-2...", flush=True)
    enc = []
    for wn in windows:
        dna = wn["dna_gt"]
        e = tok(dna, truncation=True, max_length=512, return_offsets_mapping=True)
        enc.append((dna, e["input_ids"], e["attention_mask"], [tuple(o) for o in e["offset_mapping"]]))

    rng = np.random.default_rng(42)
    out = {}
    print("\n=== D3 SYNTHETIC SWEEP: DAM-matched - uniform MLM, 3 seeds ===")
    print(f"  {'profile':<9} {'zone n':>7} " + " ".join(f"{'Δs'+str(s):>8}" for s in SEEDS) +
          f" {'POOLED Δ':>9} {'p_t':>9} {'95% CI':>20} {'n':>5}")
    for profile in PROFILES:
        w = np.load(f"{PROFILE_DIR}/{profile}.npy", allow_pickle=True).astype(float).ravel()
        zone_pos = set(np.where(w >= THRESH * w.max())[0].tolist())
        rec = {"mlm": {}, "dam": {}}
        for seed in SEEDS:
            for arm in ("mlm", "dam"):
                model = load_model(folder(arm, profile, seed))
                rec[arm][seed] = recovery_vector(model, tok, mask_id, enc, zone_pos)
                del model
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
        per_seed_delta, seed_means = [], []
        for seed in SEEDS:
            d = rec["dam"][seed] - rec["mlm"][seed]
            per_seed_delta.append(d); seed_means.append(np.nanmean(d))
        ps = pooled_stats(per_seed_delta, rng)
        out[profile] = {"zone_positions": len(zone_pos),
                        "per_seed_delta": [float(x) for x in seed_means], **ps}
        seedcol = " ".join(f"{m:>+8.4f}" for m in seed_means)
        print(f"  {profile:<9} {len(zone_pos):>7} {seedcol} {ps['delta']:>+9.4f} "
              f"{ps['p_t']:>9.2g} [{ps['ci95'][0]:+.4f},{ps['ci95'][1]:+.4f}] {ps['n']:>5}")

    Path(OUT_JSON).write_text(json.dumps(out, indent=2))
    ok = all(out[p]["delta"] > 0 and out[p]["p_t"] < 0.05 for p in PROFILES)
    print("\nVERDICT:", "DAM-matched > MLM on EVERY synthetic shape (corruption-shape-agnostic)"
          if ok else "mixed — inspect per-profile")
    for p in PROFILES:
        print(f"  {p}: pooled Δ {out[p]['delta']:+.4f} (p={out[p]['p_t']:.2g})")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
