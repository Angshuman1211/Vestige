"""
Nucleotide Transformer multi-seed eval — terminal AND background, seeds 42/1/2
(D2 THIRD backbone, "one method across backbones").

Same design and pooled statistics as the DNABERT-2 / GENA-LM multi-seed evals:
per zone (terminal damaged ends, background central band) and width, report each
seed's Delta(DAM-MLM) plus a POOLED number = seed-averaged per-window Delta tested
against 0 with a one-sample t-test + Wilcoxon + 10k-bootstrap CI (windows as the
replication unit).

NT-specific difference: NT's tokenizer is a SLOW (Python) tokenizer, so it does
not expose return_offsets_mapping. NT instead tokenizes greedily into
non-overlapping 6-mers from the left, with any <6 nt remainder split into
single-nucleotide tokens (verified empirically). Because the vocab is pure-ACGT
6-mers + single nucleotides, and every eval window is pure uppercase ACGT (checked:
0/626 windows contain a non-ACGT or lowercase character), the nucleotide span of
each token is EXACT from its string length, and cumsum over token lengths gives the
same absolute positions the fast-tokenizer offset_mapping would. This is the
identical position model the OffsetAwareDamageCollator uses at train time, so eval
and training agree by construction.

Models expected at:
    training/multiseed_models/mlm-nt-s{seed}/pytorch_model.bin
    training/multiseed_models/dam-offset-nt-s{seed}/pytorch_model.bin

Run from project root:
    python evaluation/evaluate_nt_multiseed.py
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

NT = "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species"
MODEL_ROOT = "training/multiseed_models"
SEEDS = [42, 1, 2]
ARMS = ["mlm", "dam"]          # folder = f"{mlm-nt|dam-offset-nt}-s{seed}"
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
WIDTHS = [3, 5, 10]            # T_END for terminal, half-band for background
N_BOOT = 10_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_nt_multiseed.json"


def folder(arm, seed):
    stem = "mlm-nt" if arm == "mlm" else "dam-offset-nt"
    return f"{MODEL_ROOT}/{stem}-s{seed}"


def load_model(ckpt):
    # NT loads cleanly via AutoModelForMaskedLM (EsmForMaskedLM, missing_keys=0,
    # LM head intact) -- no model_class override needed, unlike GENA-LM.
    m = AutoModelForMaskedLM.from_pretrained(NT, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


def _tok_len(s):
    """Nucleotide length of a token string: its pure-ACGT content, else 0 (special)."""
    clean = s.lstrip("#").lstrip("▁").upper() if s else ""
    return len(clean) if clean and all(ch in "ACGT" for ch in clean) else 0


def manual_offsets(ids, tokenizer):
    """Reconstruct (start, end) nucleotide spans for NT's slow tokenizer by cumsum
    over per-token nucleotide lengths. Special tokens get a zero-width span (s==e),
    matching the DNABERT-2/GENA-LM fast-tokenizer offset convention."""
    toks = tokenizer.convert_ids_to_tokens(ids)
    offsets = []
    pos = 0
    for s in toks:
        L = _tok_len(s)
        if L == 0:
            offsets.append((pos, pos))        # special / non-DNA: zero width
        else:
            offsets.append((pos, pos + L))
            pos += L
    return offsets


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
    tok = AutoTokenizer.from_pretrained(NT, trust_remote_code=True)
    mask_id = tok.convert_tokens_to_ids(tok.mask_token)
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    print(f"{len(windows)} eval windows; re-tokenizing with NT (manual 6-mer offsets)...", flush=True)
    enc = []
    for wn in windows:
        dna = wn["dna_gt"]
        e = tok(dna, truncation=True, max_length=512)
        offsets = manual_offsets(e["input_ids"], tok)
        enc.append((dna, e["input_ids"], e["attention_mask"], offsets))

    # Guard: the manual offsets must cover the full window (no <unk> desync). Every
    # window is pure ACGT so the reconstructed span must reach len(dna) exactly.
    bad = [i for i, (dna, _, _, off) in enumerate(enc) if max((e for _, e in off), default=0) != len(dna)]
    if bad:
        raise RuntimeError(f"offset reconstruction did not cover {len(bad)} windows "
                           f"(first: idx {bad[0]}) -- non-ACGT or tokenizer drift")

    zones = {"terminal": find_terminal, "background": find_background}
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
        print(f"\n=== NT {label}: DAM-offset - MLM, 3 seeds ===")
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
