"""
Phase 3 pmd-mask tradeoff curve — multi-seed eval (DNABERT-2).

Sweeps the DAM concentration knob (offset-aware DAM on the REAL aDNA profile at
power 1 / 2 / 4) between uniform MLM (no concentration) and pmd-mask (hard terminal
concentration), and reports, per arm, the pooled TERMINAL and BACKGROUND C/G recovery
(w=3). Plotting terminal vs background per arm traces the tradeoff: as concentration
rises, terminal recovery climbs and background recovery falls. The claim it supports:
"pmd-mask beats DAM on terminal" is not pmd-mask winning, it is one end of a curve, and
offset-aware DAM (power 1) sits at the balanced knee.

Arms (folder in training/multiseed_models):
    mlm  -> mlm-baseline-s{seed}        (no concentration anchor)
    p1   -> dam-offset-s{seed}          (current DAM, power 1)
    p2   -> dam-offset-p2-s{seed}
    p4   -> dam-offset-p4-s{seed}
    pmd  -> pmdmask-baseline-s{seed}    (hard terminal reference; different collator)

Missing arms are skipped with a warning (so it runs before every arm is trained).

Run from project root:
    python evaluation/evaluate_tradeoff_multiseed.py
"""
import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

DNABERT2 = "zhihan1996/DNABERT-2-117M"
MODEL_ROOT = "training/multiseed_models"
SEEDS = [42, 1, 2]
# ordered soft -> hard; label -> folder stem
ARMS = [("mlm", "mlm-baseline"), ("p1", "dam-offset"), ("p2", "dam-offset-p2"),
        ("p4", "dam-offset-p4"), ("pmd", "pmdmask-baseline")]
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
WIDTHS = [3, 5, 10]
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_tradeoff_multiseed.json"


def load_model(ckpt):
    m = AutoModelForMaskedLM.from_pretrained(DNABERT2, trust_remote_code=True, dtype=torch.float32)
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

    # per arm: seed-averaged per-window recovery, then window-mean, for terminal + background
    curve = {}
    for label, stem in ARMS:
        missing = [s for s in SEEDS if not Path(f"{MODEL_ROOT}/{stem}-s{s}/pytorch_model.bin").exists()]
        if missing:
            print(f"SKIP {label} ({stem}): missing seeds {missing}", flush=True)
            continue
        term = {w: [] for w in WIDTHS}; bg = {w: [] for w in WIDTHS}
        for seed in SEEDS:
            model = load_model(f"{MODEL_ROOT}/{stem}-s{seed}")
            for w in WIDTHS:
                term[w].append(recovery_vector(model, tok, mask_id, enc, find_terminal, w))
                bg[w].append(recovery_vector(model, tok, mask_id, enc, find_background, w))
            del model
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        curve[label] = {
            "terminal": {str(w): float(np.nanmean(np.vstack(term[w]).mean(axis=0))) for w in WIDTHS},
            "background": {str(w): float(np.nanmean(np.vstack(bg[w]).mean(axis=0))) for w in WIDTHS},
        }
        print(f"scored {label}: terminal@3={curve[label]['terminal']['3']:.3f}  "
              f"background@3={curve[label]['background']['3']:.3f}", flush=True)

    # Deltas vs MLM for the curve
    if "mlm" in curve:
        base_t = curve["mlm"]["terminal"]; base_b = curve["mlm"]["background"]
        for label in curve:
            curve[label]["dTerminal"] = {w: curve[label]["terminal"][w] - base_t[w] for w in base_t}
            curve[label]["dBackground"] = {w: curve[label]["background"][w] - base_b[w] for w in base_b}

    Path(OUT_JSON).write_text(json.dumps(curve, indent=2))
    print("\n=== TRADEOFF CURVE (w=3, pooled over seeds) ===")
    print(f"  {'arm':<5} {'terminal':>9} {'background':>11} {'Δterm':>8} {'Δbg':>8}")
    for label, _ in ARMS:
        if label not in curve:
            continue
        c = curve[label]
        dt = c.get("dTerminal", {}).get("3", 0.0); db = c.get("dBackground", {}).get("3", 0.0)
        print(f"  {label:<5} {c['terminal']['3']:>9.3f} {c['background']['3']:>11.3f} "
              f"{dt:>+8.3f} {db:>+8.3f}")
    print("\nExpected shape: terminal rises and background falls as concentration goes "
          "mlm -> p1 -> p2 -> p4 -> pmd; DAM (p1) is the balanced knee.")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
