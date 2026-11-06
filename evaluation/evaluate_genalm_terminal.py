"""
GENA-LM terminal-recovery eval — D2 pay-off test (Test 2).

Does the offset-aware DAM beat MLM at terminal C/G reconstruction on GENA-LM
(a different lab's model, different BPE)? If yes, DAM's advantage is not a
DNABERT-2 artifact and the full rebuild is justified.

Re-tokenizes each eval window's raw dna_gt with the GENA-LM tokenizer (its
offset_mapping maps nucleotides -> tokens), masks the token covering each
terminal C (first T_END nt from 5') or G (last T_END nt from 3'), predicts, and
checks the predicted token's base at the right sub-offset. Same metric as the
DNABERT-2 terminal eval, GENA-LM tokenization.

Run from project root:
    python evaluation/evaluate_genalm_terminal.py
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
T_ENDS = [3, 5, 10]
N_BOOT = 10_000
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OUT_JSON = "evaluation/results_genalm_terminal.json"


def load_model(ckpt):
    # GENA-LM's masked-LM head is reachable only via AutoModel (see train.py note).
    m = AutoModel.from_pretrained(GENALM, trust_remote_code=True, dtype=torch.float32)
    state = torch.load(f"{ckpt}/pytorch_model.bin", map_location="cpu", weights_only=True)
    m.load_state_dict(state, strict=False)
    return m.eval().to(DEVICE)


def find_terminal(dna, offsets, t_end):
    n = len(dna); pairs = []; seen = set(); toks = []
    for pos, base in enumerate(dna):
        if not ((base == "C" and pos < t_end) or (base == "G" and pos >= n - t_end)):
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
        perwin[key] = {t: [] for t in T_ENDS}
        for t in T_ENDS:
            for dna, ids, attn, offsets in enc:
                pairs, toks = find_terminal(dna, offsets, t)
                if not pairs:
                    perwin[key][t].append(float("nan")); continue
                preds = predict_at(model, tok, ids, toks, attn)
                correct = sum(nuc_correct(dna, p, ti, preds[ti], offsets, tok)
                              for p, ti in pairs if ti in preds)
                perwin[key][t].append(correct / len(pairs))
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    rng = np.random.default_rng(42)
    out = {}
    print("\nGENA-LM terminal recovery (DAM-offset vs MLM):")
    print(f"  {'T_END':>6} {'MLM':>8} {'DAM':>8} {'Δ(DAM-MLM)':>11} {'p(2-sided)':>11} {'95% CI':>22} {'n':>5}")
    for t in T_ENDS:
        r_mlm = np.array(perwin["mlm"][t]); r_dam = np.array(perwin["dam"][t])
        mask = ~(np.isnan(r_mlm) | np.isnan(r_dam))
        a, b = r_dam[mask], r_mlm[mask]
        diffs = a - b
        t_stat, p2 = stats.ttest_rel(a, b)
        ci = bootstrap_ci(diffs, N_BOOT, rng)
        out[str(t)] = {"mlm": float(b.mean()), "dam": float(a.mean()), "delta": float(diffs.mean()),
                       "t": float(t_stat), "p2": float(p2), "ci95": ci, "n": int(mask.sum())}
        print(f"  {t:>5}n {b.mean():>8.4f} {a.mean():>8.4f} {diffs.mean():>+11.4f} "
              f"{p2:>11.3g} [{ci[0]:+.4f},{ci[1]:+.4f}] {int(mask.sum()):>5}")

    Path(OUT_JSON).write_text(json.dumps({"per_window": perwin, "summary": out}, indent=2))
    print(f"\nVERDICT: {'DAM > MLM at terminal on GENA-LM -> full rebuild justified' if out['10']['delta']>0 and out['10']['p2']<0.05 else 'weak/negative -> reconsider before full rebuild'}")
    print(f"JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
