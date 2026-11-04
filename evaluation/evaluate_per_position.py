"""
Per-position terminal reconstruction analysis — VESTIGE Phase 3 (extension).

Evaluates MLM vs DAM recovery rate at each individual distance d from the read
end (d = 1 .. 25), rather than binning into a T_END window. This shows the
fine-grained damage gradient: DAM's advantage should peak at d=1 (highest PMD
probability) and decay monotonically toward the window interior.

Positions evaluated:
  - C at nuc_pos = d-1 from the 5' end  (C→T deamination zone)
  - G at nuc_pos = n - d from the 3' end (G→A deamination zone)

Both strands are evaluated separately (5p_C and 3p_G) so the figure can show
whether the advantage is symmetric or strand-biased.

Outputs:
    evaluation/per_position_results.json
    results/figures/fig_per_position.{pdf,png}

Run from project root:
    python3 evaluation/evaluate_per_position.py
"""
import sys, types, importlib.util, importlib.machinery

# triton stub removed: it broke transformers 4.57 import; DNABERT-2 falls back to torch attention.

import os  # force torch-only: transformers TF backend breaks with Keras 3 (requirements.txt)
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from transformers import AutoModelForMaskedLM, AutoTokenizer

DAMAGED_VAL    = "evaluation/damaged_validation.npy"
TOKENIZER_NAME = "zhihan1996/DNABERT-2-117M"
CHECKPOINTS = {
    "MLM baseline": "training/checkpoints/mlm-baseline/checkpoint-595",
    "DAM proposed":  "training/checkpoints/dam-proposed/checkpoint-595",
}
OUT_JSON = Path("evaluation/per_position_results.json")
OUT_FIG  = Path("results/figures/fig_per_position")

MAX_D  = 25
SEED   = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(ckpt, device):
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    state = torch.load(Path(ckpt) / "pytorch_model.bin",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def predict_tok(model, tokenizer, gt_ids, tok_pos, attn_mask, device):
    """Mask a single token position, run model, return predicted token id."""
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    masked  = gt_ids[:]
    masked[tok_pos] = mask_id
    logits = model(
        input_ids=torch.tensor([masked], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attn_mask], dtype=torch.long, device=device),
    ).logits
    return int(logits[0, tok_pos].argmax())


def nuc_correct(dna, nuc_pos, tok_pos, pred_id, offsets, tokenizer) -> bool:
    tok = tokenizer.convert_ids_to_tokens([pred_id])[0]
    if not tok:
        return False
    s   = tok.lstrip("#").lstrip("▁").upper()
    off = nuc_pos - offsets[tok_pos][0]
    return 0 <= off < len(s) and s[off] == dna[nuc_pos]


def find_tok(nuc_pos, offsets):
    """Return token index that contains nuc_pos, or None."""
    for idx, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s <= nuc_pos < e:
            return idx
    return None


def evaluate_position(d, windows, mlm_model, dam_model, tokenizer):
    """
    For distance d (1-indexed from each end), collect per-site results for
    5'-C positions and 3'-G positions separately.
    Returns dict with per-strand lists of (mlm_correct, dam_correct).
    """
    results_5p = []   # (mlm_correct, dam_correct) for each C at pos d-1
    results_3p = []   # (mlm_correct, dam_correct) for each G at pos n-d

    for w in windows:
        dna     = w["dna_gt"]
        n       = len(dna)
        offsets = [tuple(o) for o in w["offsets"]]

        # 5' C at position d-1 (0-indexed)
        nuc_5p = d - 1
        if nuc_5p < n and dna[nuc_5p] == "C":
            tok = find_tok(nuc_5p, offsets)
            if tok is not None:
                mlm_pred = predict_tok(mlm_model, tokenizer, w["gt_ids"], tok,
                                       w["attention_mask"], DEVICE)
                dam_pred = predict_tok(dam_model, tokenizer, w["gt_ids"], tok,
                                       w["attention_mask"], DEVICE)
                results_5p.append((
                    nuc_correct(dna, nuc_5p, tok, mlm_pred, offsets, tokenizer),
                    nuc_correct(dna, nuc_5p, tok, dam_pred, offsets, tokenizer),
                ))

        # 3' G at position n-d (0-indexed)
        nuc_3p = n - d
        if nuc_3p >= 0 and nuc_3p < n and dna[nuc_3p] == "G":
            tok = find_tok(nuc_3p, offsets)
            if tok is not None:
                mlm_pred = predict_tok(mlm_model, tokenizer, w["gt_ids"], tok,
                                       w["attention_mask"], DEVICE)
                dam_pred = predict_tok(dam_model, tokenizer, w["gt_ids"], tok,
                                       w["attention_mask"], DEVICE)
                results_3p.append((
                    nuc_correct(dna, nuc_3p, tok, mlm_pred, offsets, tokenizer),
                    nuc_correct(dna, nuc_3p, tok, dam_pred, offsets, tokenizer),
                ))

    return {"5p": results_5p, "3p": results_3p}


def strand_stats(pairs):
    """Given list of (mlm_correct, dam_correct) bools, return summary dict."""
    if not pairs:
        return {"n": 0, "mlm": float("nan"), "dam": float("nan"),
                "delta": float("nan"), "p_value": float("nan")}
    n = len(pairs)
    mlm_arr = np.array([p[0] for p in pairs], dtype=float)
    dam_arr = np.array([p[1] for p in pairs], dtype=float)
    if n >= 2:
        _, p_val = stats.ttest_rel(dam_arr, mlm_arr)
    else:
        p_val = float("nan")
    return {
        "n":       n,
        "mlm":     float(mlm_arr.mean()),
        "dam":     float(dam_arr.mean()),
        "delta":   float((dam_arr - mlm_arr).mean()),
        "p_value": float(p_val),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    windows   = np.load(DAMAGED_VAL, allow_pickle=True).tolist()

    print("Loading MLM baseline...")
    mlm_model = load_model(CHECKPOINTS["MLM baseline"], DEVICE)
    print("Loading DAM proposed...")
    dam_model = load_model(CHECKPOINTS["DAM proposed"], DEVICE)

    print("\n  d   5'C n   5'C MLM   5'C DAM     5'C D   3'G n   3'G MLM   3'G DAM     3'G D")
    print("-" * 87)

    all_results = {}
    for d in range(1, MAX_D + 1):
        raw = evaluate_position(d, windows, mlm_model, dam_model, tokenizer)
        s5  = strand_stats(raw["5p"])
        s3  = strand_stats(raw["3p"])
        all_results[d] = {"5p": s5, "3p": s3}

        def fmt(v): return f"{v:>8.3f}" if not np.isnan(v) else "     N/A"
        print(f"{d:>3}  {s5['n']:>6}  {fmt(s5['mlm'])}  {fmt(s5['dam'])}  {fmt(s5['delta'])}"
              f"  {s3['n']:>6}  {fmt(s3['mlm'])}  {fmt(s3['dam'])}  {fmt(s3['delta'])}")

    OUT_JSON.write_text(json.dumps({str(k): v for k, v in all_results.items()}, indent=2))

    # ── Figure ────────────────────────────────────────────────────────────────────
    ds = list(range(1, MAX_D + 1))

    def extract(strand):
        mlm = [all_results[d][strand]["mlm"]   for d in ds]
        dam = [all_results[d][strand]["dam"]    for d in ds]
        delta = [all_results[d][strand]["delta"] for d in ds]
        ps  = [all_results[d][strand]["p_value"] for d in ds]
        ns  = [all_results[d][strand]["n"]       for d in ds]
        return mlm, dam, delta, ps, ns

    m5, d5, delta5, p5, n5 = extract("5p")
    m3, d3, delta3, p3, n3 = extract("3p")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(
        "Per-position recovery rate: DAM vs MLM · 69 validation windows · * p < 0.05",
        fontsize=10, color="gray",
    )

    def plot_rates(ax, ds, mlm, dam, ps, ns, title, strand_label):
        ax.plot(ds, mlm, "o-", color="#2980b9", lw=1.8, ms=5, label="MLM baseline")
        ax.plot(ds, dam, "s-", color="#c0392b", lw=1.8, ms=5, label="DAM proposed")
        for d_, p_, dv, mv in zip(ds, ps, dam, mlm):
            if not np.isnan(p_) and p_ < 0.05:
                ax.annotate("*", xy=(d_, max(dv, mv) + 0.03),
                            ha="center", fontsize=12, color="#27ae60", fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Distance from {strand_label} end (nt)", fontsize=9)
        ax.set_ylabel("Recovery rate", fontsize=9)
        ax.set_xticks(range(1, MAX_D + 1, 2))
        ax.legend(fontsize=8, frameon=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # n annotation on secondary axis
        ax2 = ax.twinx()
        ax2.bar(ds, ns, alpha=0.08, color="gray", width=0.4, label="n sites")
        ax2.set_ylabel("Sites (n)", fontsize=8, color="gray")
        ax2.tick_params(axis="y", labelcolor="gray", labelsize=7)
        ax2.spines["top"].set_visible(False)

    def plot_delta(ax, ds, delta, ps, title, strand_label):
        colors = ["#27ae60" if dv > 0 else "#e74c3c" for dv in delta]
        ax.bar(ds, delta, color=colors, alpha=0.7, width=0.6)
        ax.axhline(0, color="black", lw=0.8, ls="--")
        for d_, p_, dv in zip(ds, ps, delta):
            if not np.isnan(p_) and p_ < 0.05:
                offset = 0.01 if dv >= 0 else -0.02
                ax.annotate("*", xy=(d_, dv + offset),
                            ha="center", fontsize=12, color="#27ae60", fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"Distance from {strand_label} end (nt)", fontsize=9)
        ax.set_ylabel("DAM − MLM (Δ)", fontsize=9)
        ax.set_xticks(range(1, MAX_D + 1, 2))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plot_rates(axes[0, 0], ds, m5, d5, p5, n5, "5′ C positions — recovery rate", "5′")
    plot_delta(axes[0, 1], ds, delta5, p5, "5′ C positions — DAM advantage (Δ)", "5′")
    plot_rates(axes[1, 0], ds, m3, d3, p3, n3, "3′ G positions — recovery rate", "3′")
    plot_delta(axes[1, 1], ds, delta3, p3, "3′ G positions — DAM advantage (Δ)", "3′")

    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_FIG) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_FIG) + ".png", dpi=300, bbox_inches="tight")
    print(f"\nFigure → {OUT_FIG}.{{pdf,png}}")
    print(f"JSON   → {OUT_JSON}")


if __name__ == "__main__":
    main()
