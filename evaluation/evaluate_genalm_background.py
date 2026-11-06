"""
GENA-LM BACKGROUND-recovery eval — the pmd-mask collapse check (P27 caveat).

The terminal eval (evaluate_genalm_terminal.py) shows DAM beats MLM at the
damaged 5'/3' ends. The open question is whether DAM buys that terminal gain by
SACRIFICING interior recovery — the failure mode that sank hard thresholding
(pmd-mask: +0.164 terminal but -0.134 background, P24). DAM is meant to be the
BALANCED operating point: terminal gain WITHOUT a background collapse.

This measures recovery in a CENTRAL band (same half-width as the terminal zone,
but centred on the window), where PMD damage is at baseline for both C and G, so
both arms see an un-damaged, uniform-difficulty task. A DAM-minus-MLM delta near
zero (CI spanning 0) is the desired result: no background cost. A large negative
delta would mean DAM over-specialised to the ends.

Same backbone, tokenizer, metric and models as the terminal eval — only the
position band changes (centre instead of the two ends).

Run from project root:
    python evaluation/evaluate_genalm_background.py
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
ARMS = {"mlm": "mlm-genalm-s42", "dam": "dam-offset-genalm-s42"}
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
# Central-band half-widths, matched to the terminal eval's T_ENDS so the numbers
# are directly comparable (a 3/5/10 nt central band vs a 3/5/10 nt terminal band).
BANDS = [3, 5, 10]
N_BOOT = 10_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_genalm_background.json"


def load_model(ckpt):
    m = AutoModel.from_pretrained(GENALM, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


def find_background(dna, offsets, band):
    # C/G within a central band of half-width `band` around the window midpoint,
    # excluding the terminal zones so a long window never double-counts an end.
    n = len(dna); mid = n // 2
    lo, hi = mid - band, mid + band
    pairs = []; seen = set(); toks = []
    for pos, base in enumerate(dna):
        if base not in ("C", "G"):
            continue
        if not (lo <= pos < hi and band <= pos < n - band):
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
def predict_at(model, tokenizer, ids, toks, attn):
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
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


def bootstrap_ci(diffs, n, rng):
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    tok = AutoTokenizer.from_pretrained(GENALM, trust_remote_code=True)
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    print(f"{len(windows)} eval windows; re-tokenizing with GENA-LM...", flush=True)

    enc = []
    for w in windows:
        dna = w["dna_gt"]
        e = tok(dna, truncation=True, max_length=512, return_offsets_mapping=True)
        enc.append((dna, e["input_ids"], e["attention_mask"],
                    [tuple(o) for o in e["offset_mapping"]]))

    perwin = {}
    for key, name in ARMS.items():
        print(f"evaluating {key} ({name}) on {DEVICE}...", flush=True)
        model = load_model(f"{MODEL_ROOT}/{name}")
        perwin[key] = {b: [] for b in BANDS}
        for b in BANDS:
            for dna, ids, attn, offsets in enc:
                pairs, toks = find_background(dna, offsets, b)
                if not pairs:
                    perwin[key][b].append(float("nan")); continue
                preds = predict_at(model, tok, ids, toks, attn)
                correct = sum(nuc_correct(dna, p, ti, preds[ti], offsets, tok)
                              for p, ti in pairs if ti in preds)
                perwin[key][b].append(correct / len(pairs))
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    rng = np.random.default_rng(42)
    out = {}
    print("\nGENA-LM BACKGROUND recovery (central band; DAM-offset vs MLM):")
    print(f"  {'band':>6} {'MLM':>8} {'DAM':>8} {'Δ(DAM-MLM)':>11} {'p(2-sided)':>11} {'95% CI':>22} {'n':>5}")
    for b in BANDS:
        r_mlm = np.array(perwin["mlm"][b]); r_dam = np.array(perwin["dam"][b])
        mask = ~(np.isnan(r_mlm) | np.isnan(r_dam))
        a, bb = r_dam[mask], r_mlm[mask]
        diffs = a - bb
        t_stat, p2 = stats.ttest_rel(a, bb)
        ci = bootstrap_ci(diffs, N_BOOT, rng)
        out[str(b)] = {"mlm": float(bb.mean()), "dam": float(a.mean()), "delta": float(diffs.mean()),
                       "t": float(t_stat), "p2": float(p2), "ci95": ci, "n": int(mask.sum())}
        print(f"  {b:>5}c {bb.mean():>8.4f} {a.mean():>8.4f} {diffs.mean():>+11.4f} "
              f"{p2:>11.3g} [{ci[0]:+.4f},{ci[1]:+.4f}] {int(mask.sum()):>5}")

    Path(OUT_JSON).write_text(json.dumps({"per_window": perwin, "summary": out}, indent=2))
    worst = min(out[str(b)]["delta"] for b in BANDS)
    print(f"\nVERDICT: {'no background collapse (DAM holds interior) -> DAM is the balanced operating point' if worst > -0.03 else 'background cost detected -> DAM trades interior for terminal, report the tradeoff'}")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
