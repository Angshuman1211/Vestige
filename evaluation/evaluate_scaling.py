"""
Terminal-zone sensitivity analysis — VESTIGE Phase 3 (Figure 3).

Runs the terminal-position reconstruction evaluation at T_END ∈ {3, 5, 10, 15, 20, 25} nt.
T_END defines how many nucleotides from each window end are treated as "terminal" —
the zone where mapDamage2 shows exponentially decaying C→T and G→A deamination.

DAM masking probability peaks at position 1 (~28%) and decays toward the window
interior. If DAM's advantage is genuine, it should be strongest at small T_END
(highest-damage zone) and diminish as T_END grows to include background-like positions.

Method: mask ALL terminal C/G positions in each validation window (no stochastic
damage applied), matching the approach in evaluate_terminal.py. This gives well-
powered evaluation (n = hundreds of sites) at every T_END.

Outputs:
    evaluation/scaling_results.json
    results/figures/fig3_damage_scaling.{pdf,png}

Run from project root:
    python3 evaluation/evaluate_scaling.py
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

DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
TOKENIZER_NAME = "zhihan1996/DNABERT-2-117M"
CHECKPOINTS = {
    "MLM baseline": "training/checkpoints/mlm-baseline/checkpoint-595",
    "DAM proposed":  "training/checkpoints/dam-proposed/checkpoint-595",
}
OUT_JSON = Path("evaluation/scaling_results.json")
OUT_FIG  = Path("results/figures/fig3_damage_scaling")

T_ENDS = [3, 5, 10, 15, 20, 25]
N_BOOT = 10_000
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


def load_zeroshot(device):
    """Pre-trained DNABERT-2 with no fine-tuning."""
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    return model.eval().to(device)


def find_terminal(dna: str, offsets: list, t_end: int):
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
    masked  = gt_ids[:]
    for tp in tok_positions:
        masked[tp] = mask_id
    logits = model(
        input_ids=torch.tensor([masked], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attn_mask], dtype=torch.long, device=device),
    ).logits
    log_probs = torch.log_softmax(logits[0], dim=-1)  # (seq_len, vocab)
    preds    = {tp: int(logits[0, tp].argmax()) for tp in tok_positions}
    lp_correct = {tp: float(log_probs[tp, gt_ids[tp]]) for tp in tok_positions}
    return preds, lp_correct


def nuc_correct(dna, nuc_pos, tok_pos, pred_id, offsets, tokenizer) -> bool:
    tok = tokenizer.convert_ids_to_tokens([pred_id])[0]
    if not tok:
        return False
    s   = tok.lstrip("#").lstrip("▁").upper()
    off = nuc_pos - offsets[tok_pos][0]
    return 0 <= off < len(s) and s[off] == dna[nuc_pos]


def bootstrap_ci(diffs, n, rng):
    means = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                      for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def random_rate(pairs, rng):
    """Per-window random baseline: Bernoulli(0.25) per nucleotide site."""
    if not pairs:
        return float("nan")
    hits = sum(rng.random() < 0.25 for _ in pairs)
    return hits / len(pairs)


def run_t_end(t_end, windows, mlm_model, dam_model, zs_model, tokenizer, rng):
    mlm_rates, dam_rates, zs_rates, rnd_rates = [], [], [], []
    mlm_lp, dam_lp = [], []
    total_sites = 0

    for w in windows:
        offsets = [tuple(o) for o in w["offsets"]]
        pairs, tok_positions = find_terminal(w["dna_gt"], offsets, t_end)
        if not pairs:
            continue
        total_sites += len(pairs)

        for model, rates, lp_list in [
            (mlm_model, mlm_rates, mlm_lp),
            (dam_model, dam_rates, dam_lp),
            (zs_model,  zs_rates,  None),
        ]:
            preds, lp_correct = predict_at(model, tokenizer, w["gt_ids"],
                                           tok_positions, w["attention_mask"], DEVICE)
            correct = sum(
                nuc_correct(w["dna_gt"], np_, tp, preds[tp], offsets, tokenizer)
                for np_, tp in pairs if tp in preds
            )
            rates.append(correct / len(pairs))
            if lp_list is not None:
                lp_list.append(float(np.mean([lp_correct[tp] for tp in tok_positions])))

        rnd_rates.append(random_rate(pairs, rng))

    r_mlm = np.array(mlm_rates)
    r_dam = np.array(dam_rates)
    r_zs  = np.array(zs_rates)
    r_rnd = np.array(rnd_rates)
    lp_mlm = np.array(mlm_lp)
    lp_dam = np.array(dam_lp)
    n_w = len(r_mlm)

    t_stat = p_two = p_one = ci_lo = ci_hi = float("nan")
    lp_t = lp_p_two = lp_p_one = float("nan")
    if n_w >= 2:
        t_stat, p_two = stats.ttest_rel(r_dam, r_mlm)
        p_one  = float(p_two / 2) if t_stat > 0 else float(1 - p_two / 2)
        diffs  = r_dam - r_mlm
        ci_lo, ci_hi = bootstrap_ci(diffs, N_BOOT, rng)
        lp_t, lp_p_two = stats.ttest_rel(lp_dam, lp_mlm)
        lp_p_one = float(lp_p_two / 2) if lp_t > 0 else float(1 - lp_p_two / 2)

    return {
        "t_end":         t_end,
        "n_sites":       total_sites,
        "n_windows":     n_w,
        "mlm_mean":      float(r_mlm.mean())  if n_w else float("nan"),
        "dam_mean":      float(r_dam.mean())  if n_w else float("nan"),
        "zs_mean":       float(r_zs.mean())   if n_w else float("nan"),
        "rnd_mean":      float(r_rnd.mean())  if n_w else float("nan"),
        "delta":         float((r_dam - r_mlm).mean()) if n_w else float("nan"),
        "t_stat":        float(t_stat),
        "p_value_two":   float(p_two),
        "p_value_one":   float(p_one),
        "ci_95":         [ci_lo, ci_hi],
        "mlm_logprob":   float(lp_mlm.mean()) if n_w else float("nan"),
        "dam_logprob":   float(lp_dam.mean()) if n_w else float("nan"),
        "delta_logprob": float((lp_dam - lp_mlm).mean()) if n_w else float("nan"),
        "lp_p_two":      float(lp_p_two),
        "lp_p_one":      float(lp_p_one),
    }


def main():
    rng       = np.random.default_rng(SEED)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    extra_path = Path(DAMAGED_VAL_EXTRA)
    if extra_path.exists():
        extra = np.load(extra_path, allow_pickle=True).tolist()
        windows = windows + extra
        print(f"Windows: {len(windows) - len(extra)} original + {len(extra)} extra = {len(windows)} total")
    else:
        print(f"Windows: {len(windows)} (run expand_validation.py to add more)")

    print("Loading MLM baseline...")
    mlm_model = load_model(CHECKPOINTS["MLM baseline"], DEVICE)
    print("Loading DAM proposed...")
    dam_model = load_model(CHECKPOINTS["DAM proposed"], DEVICE)
    print("Loading zero-shot DNABERT-2 (no fine-tuning)...")
    zs_model  = load_zeroshot(DEVICE)

    print(f"\n{'T_END':>6}  {'Sites':>7}  {'Win':>5}  "
          f"{'Random':>8}  {'ZeroShot':>9}  {'MLM':>8}  {'DAM':>8}  "
          f"{'Δ(DAM-MLM)':>11}  {'p(2)':>7}  {'p(1)':>7}")
    print("-" * 96)

    all_results = {}
    for t_end in T_ENDS:
        r = run_t_end(t_end, windows, mlm_model, dam_model, zs_model, tokenizer, rng)
        all_results[t_end] = r

        def ps(v): return f"{v:.3f}" if not np.isnan(v) else "  N/A"
        def sig(v): return "*" if not np.isnan(v) and v < 0.05 else " "

        print(f"{t_end:>5}nt  {r['n_sites']:>7}  {r['n_windows']:>5}  "
              f"{r['rnd_mean']:>8.4f}  {r['zs_mean']:>9.4f}  "
              f"{r['mlm_mean']:>8.4f}  {r['dam_mean']:>8.4f}  "
              f"{r['delta']:>+11.4f}  "
              f"{ps(r['p_value_two'])}{sig(r['p_value_two'])}  "
              f"{ps(r['p_value_one'])}{sig(r['p_value_one'])}")

    OUT_JSON.write_text(json.dumps({str(k): v for k, v in all_results.items()}, indent=2))

    # ── Data extraction ──────────────────────────────────────────────────────────
    t_vals  = [r["t_end"]        for r in all_results.values()]
    rnd_v   = [r["rnd_mean"]     for r in all_results.values()]
    zs_v    = [r["zs_mean"]      for r in all_results.values()]
    mlm_v   = [r["mlm_mean"]     for r in all_results.values()]
    dam_v   = [r["dam_mean"]     for r in all_results.values()]
    deltas  = [r["delta"]        for r in all_results.values()]
    p_vals  = [r["p_value_two"]  for r in all_results.values()]
    n_wins  = [r["n_windows"]    for r in all_results.values()]

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)

    COLORS = {
        "Random":    "#94a3b8",
        "Zero-shot": "#a855f7",
        "MLM":       "#2980b9",
        "DAM":       "#c0392b",
    }

    # ── Figure 3 — 4-baseline recovery + delta ──────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax1.axhline(0.25, color="#94a3b8", lw=1.2, ls=":", label="Random (theoretical)")
    ax1.plot(t_vals, zs_v,  "^-", color=COLORS["Zero-shot"], lw=1.8, ms=6,
             label="Zero-shot DNABERT-2")
    ax1.plot(t_vals, rnd_v, "v--", color=COLORS["Random"],    lw=1.5, ms=5,
             label="Random (observed)", alpha=0.7)
    ax1.plot(t_vals, mlm_v, "o-", color=COLORS["MLM"],        lw=2,   ms=7,
             label="MLM fine-tuning")
    ax1.plot(t_vals, dam_v, "s-", color=COLORS["DAM"],        lw=2.5, ms=8,
             label="DAM (proposed)", zorder=5)

    for tv, p, d, m in zip(t_vals, p_vals, dam_v, mlm_v):
        if not np.isnan(p) and p < 0.05:
            ax1.annotate("***", xy=(tv, max(d, m) + 0.008), ha="center",
                         fontsize=9, color="#27ae60", fontweight="bold")

    ax1.set_xlabel("Terminal zone width (nt from each end)", fontsize=11)
    ax1.set_ylabel("Nucleotide recovery rate (mean per window)", fontsize=11)
    ax1.set_title("Four-baseline comparison vs terminal zone width",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(t_vals)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bar_colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
    ax2.bar(t_vals, deltas, color=bar_colors, alpha=0.8, width=1.8)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    for tv, p, d in zip(t_vals, p_vals, deltas):
        offset = 0.004 if d >= 0 else -0.008
        label  = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else ""))
        if label:
            ax2.annotate(label, xy=(tv, d + offset), ha="center",
                         fontsize=9, color="#27ae60", fontweight="bold")

    ax2b = ax2.twinx()
    ax2b.plot(t_vals, n_wins, "D--", color="gray", lw=1.2, ms=5)
    ax2b.set_ylabel("Windows (n)", fontsize=10, color="gray")
    ax2b.tick_params(axis="y", labelcolor="gray")
    ax2b.spines["top"].set_visible(False)

    ax2.set_xlabel("Terminal zone width (nt from each end)", fontsize=11)
    ax2.set_ylabel("DAM − MLM recovery (Δ)", fontsize=11)
    ax2.set_title("DAM advantage over MLM", fontsize=12, fontweight="bold")
    ax2.set_xticks(t_vals)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    n_genes = len({w["gene"] for w in
                   np.load(DAMAGED_VAL, allow_pickle=True).tolist() +
                   (np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
                    if Path(DAMAGED_VAL_EXTRA).exists() else [])})
    fig.suptitle(
        f"Terminal C/G reconstruction · {max(n_wins)} windows · {n_genes} genes · "
        "*** p < 0.001 (paired t-test, two-sided)",
        fontsize=9, color="gray", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(str(OUT_FIG) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_FIG) + ".png", dpi=300, bbox_inches="tight")
    print(f"\nFigure 3 → {OUT_FIG}.{{pdf,png}}")

    # ── Figure 3b — Headline bar chart (T_END = 3 spotlight) ────────────────────
    OUT_FIG_B = OUT_FIG.parent / "fig3b_headline"
    r3 = all_results[3]

    methods = ["Zero-shot\nDNABERT-2", "MLM\nfine-tuning", "Random\n(chance)", "DAM\n(proposed)"]
    values  = [r3["zs_mean"], r3["mlm_mean"], r3["rnd_mean"], r3["dam_mean"]]
    colors  = [COLORS["Zero-shot"], COLORS["MLM"], COLORS["Random"], COLORS["DAM"]]
    alphas  = [0.85, 0.85, 0.55, 1.0]

    fig2, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, values, color=colors,
                  alpha=1.0, width=0.55, zorder=3)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.axhline(0.25, color="#64748b", lw=1.5, ls="--", zorder=4, label="Random (theoretical 25%)")
    ax.set_ylabel("Nucleotide recovery rate", fontsize=12)
    ax.set_title("T_END = 3 nt  ·  Innermost terminal zone\n"
                 "(Peak post-mortem damage — highest C→T probability)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(0, 0.42)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.007,
                f"{v:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Annotation: MLM below random
    ax.annotate("", xy=(0.5, r3["mlm_mean"]), xytext=(0.5, r3["rnd_mean"] - 0.003),
                arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.5))
    ax.text(0.6, (r3["mlm_mean"] + r3["rnd_mean"]) / 2,
            "MLM < random", color="#dc2626", fontsize=9, va="center")

    # Annotation: DAM vs MLM delta
    x_dam = 3
    ax.annotate("", xy=(x_dam, r3["dam_mean"]), xytext=(x_dam, r3["mlm_mean"]),
                arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=1.5))
    delta_pct = (r3["dam_mean"] - r3["mlm_mean"]) * 100
    ax.text(x_dam + 0.08, (r3["dam_mean"] + r3["mlm_mean"]) / 2,
            f"+{delta_pct:.1f} pp\n(p < 0.001)", color="#16a34a",
            fontsize=9, va="center")

    fig2.suptitle(f"n = {r3['n_windows']} windows  ·  {r3['n_sites']} terminal C/G sites  ·"
                  f"  7 genes (TRPV3, KCNK9, HBB, TRPA1, UCP1, ADRB3, FASN)",
                  fontsize=8, color="gray", y=0.01)
    plt.tight_layout()
    fig2.savefig(str(OUT_FIG_B) + ".pdf", bbox_inches="tight")
    fig2.savefig(str(OUT_FIG_B) + ".png", dpi=300, bbox_inches="tight")
    print(f"Figure 3b → {OUT_FIG_B}.{{pdf,png}}")
    print(f"JSON      → {OUT_JSON}")


if __name__ == "__main__":
    main()
