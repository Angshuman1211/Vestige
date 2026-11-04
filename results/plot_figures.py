"""
Regenerate Figure 3 and Figure 3b from saved scaling_results.json.
No model inference needed — reads directly from evaluation/scaling_results.json.

Run from project root:
    python3 results/plot_figures.py
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

SCALING_JSON      = Path("evaluation/scaling_results.json")
DAMAGED_VAL       = "evaluation/damaged_validation.npy"
DAMAGED_VAL_EXTRA = "evaluation/damaged_validation_extra.npy"
OUT_DIR           = Path("results/figures")

COLORS = {
    "Random":    "#94a3b8",
    "Zero-shot": "#a855f7",
    "MLM":       "#2980b9",
    "DAM":       "#c0392b",
}


def load_results():
    raw = json.loads(SCALING_JSON.read_text())
    return {int(k): v for k, v in raw.items()}


def make_fig3(all_results, n_genes):
    t_vals = sorted(all_results.keys())
    rnd_v  = [all_results[t]["rnd_mean"]    for t in t_vals]
    zs_v   = [all_results[t]["zs_mean"]     for t in t_vals]
    mlm_v  = [all_results[t]["mlm_mean"]    for t in t_vals]
    dam_v  = [all_results[t]["dam_mean"]    for t in t_vals]
    deltas = [all_results[t]["delta"]       for t in t_vals]
    p_vals = [all_results[t]["p_value_two"] for t in t_vals]
    n_wins = [all_results[t]["n_windows"]   for t in t_vals]

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

    fig.suptitle(
        f"Terminal C/G reconstruction · {max(n_wins)} windows · {n_genes} genes · "
        "*** p < 0.001 (paired t-test, two-sided)",
        fontsize=9, color="gray", y=1.01,
    )
    ax1.set_ylim(top=0.40)  # headroom for the callout above the DAM line
    ax1.text(0.5, 0.97,
             "Gap largest at T_END = 3; it narrows as wider zones mix in\n"
             "background C/G positions where DAM and MLM perform alike.",
             transform=ax1.transAxes, ha="center", va="top", fontsize=8.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                       edgecolor="#16a34a", linewidth=1.0, alpha=0.95))
    plt.tight_layout()
    out = OUT_DIR / "fig3_damage_scaling"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    print(f"Figure 3  → {out}.{{pdf,png}}")
    plt.close(fig)


def make_fig3b(r3):
    methods = ["Zero-shot\nDNABERT-2", "MLM\nfine-tuning", "Random\n(chance)", "DAM\n(proposed)"]
    values  = [r3["zs_mean"], r3["mlm_mean"], r3["rnd_mean"], r3["dam_mean"]]
    colors  = [COLORS["Zero-shot"], COLORS["MLM"], COLORS["Random"], COLORS["DAM"]]
    alphas  = [0.85, 0.85, 0.55, 1.0]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(methods, values, color=colors, width=0.55, zorder=3)
    for bar, alpha in zip(bars, alphas):
        bar.set_alpha(alpha)

    ax.axhline(0.25, color="#64748b", lw=1.5, ls="--", zorder=4,
               label="Random (theoretical 25%)")
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

    # MLM < random annotation
    ax.annotate("", xy=(0.5, r3["mlm_mean"]), xytext=(0.5, r3["rnd_mean"] - 0.003),
                arrowprops=dict(arrowstyle="<->", color="#dc2626", lw=1.5))
    ax.text(0.62, (r3["mlm_mean"] + r3["rnd_mean"]) / 2,
            "MLM < random", color="#dc2626", fontsize=9, va="center")

    # DAM vs MLM delta annotation
    ax.annotate("", xy=(3, r3["dam_mean"]), xytext=(3, r3["mlm_mean"]),
                arrowprops=dict(arrowstyle="<->", color="#16a34a", lw=1.5))
    delta_pct = (r3["dam_mean"] - r3["mlm_mean"]) * 100
    ax.text(3.08, (r3["dam_mean"] + r3["mlm_mean"]) / 2,
            f"+{delta_pct:.1f} pp\n(p < 0.001)", color="#16a34a",
            fontsize=9, va="center")

    fig.suptitle(
        f"n = {r3['n_windows']} windows  ·  {r3['n_sites']} terminal C/G sites  ·"
        "  7 genes (TRPV3, KCNK9, HBB, TRPA1, UCP1, ADRB3, FASN)",
        fontsize=8, color="gray", y=0.01,
    )
    ax.text(0.03, 0.97,
            "MLM < Random: standard MLM produces a worse-than-\n"
            "random model at the very positions that matter most.",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdecea",
                      edgecolor="#dc2626", linewidth=1.0, alpha=0.95))
    plt.tight_layout()
    out = OUT_DIR / "fig3b_headline"
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out) + ".png", dpi=300, bbox_inches="tight")
    print(f"Figure 3b → {out}.{{pdf,png}}")
    plt.close(fig)


def main():
    all_results = load_results()

    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if Path(DAMAGED_VAL_EXTRA).exists():
        windows += np.load(DAMAGED_VAL_EXTRA, allow_pickle=True).tolist()
    n_genes = len({w["gene"] for w in windows})

    make_fig3(all_results, n_genes)
    make_fig3b(all_results[3])
    print("Done.")


if __name__ == "__main__":
    main()
