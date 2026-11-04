"""
Fig 8 — per-residue pLDDT profiles for TRPV3, KCNK9, HBB.
Reads: protein/structures/plddt_scores.json
Writes: results/figures/fig_structural.{pdf,png}
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

DATA = Path("protein/structures/plddt_scores.json")
OUT  = Path("results/figures/fig_structural")

GENES = ["TRPV3", "KCNK9", "HBB"]
LENGTHS = {"TRPV3": 400, "KCNK9": 347, "HBB": 162}
NOTES = {
    "TRPV3": "N-terminal domain (aa 1–400)",
    "KCNK9": "full length (347 aa)",
    "HBB":   "full length (162 aa)",
}

REF_COLOR = "#444444"
MLM_COLOR = "#2166ac"
DAM_COLOR = "#d6604d"

with open(DATA) as f:
    data = json.load(f)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
fig.suptitle(
    "Per-residue pLDDT: reference vs. reconstructed sequences · ESMFold",
    fontsize=10, y=1.01
)
fig.text(0.5, 1.11,
         "Extreme-damage stress test: TM-score > 0.95 even at 30× the authentic "
         "PMD rate - reconstructions remain fold-correct.",
         ha="center", va="bottom", fontsize=8.5,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef7ff",
                   edgecolor="#2166ac", linewidth=1.0, alpha=0.95))

for ax, gene in zip(axes, GENES):
    ref = np.array(data[f"{gene}_reference"]["plddt_per_res"])
    mlm = np.array(data[f"{gene}_mlm"]["plddt_per_res"])
    dam = np.array(data[f"{gene}_dam"]["plddt_per_res"])
    x   = np.arange(1, len(ref) + 1)

    ax.plot(x, ref, color=REF_COLOR, lw=1.2, label="Reference", zorder=3)
    ax.plot(x, mlm, color=MLM_COLOR, lw=1.0, ls="--", label="MLM recon", zorder=2)
    ax.plot(x, dam, color=DAM_COLOR, lw=1.0, ls=":",  label="DAM recon", zorder=2)

    # pLDDT confidence bands
    ax.axhspan(90, 100, alpha=0.06, color="green",  zorder=0)
    ax.axhspan(70,  90, alpha=0.06, color="limegreen", zorder=0)
    ax.axhspan(50,  70, alpha=0.06, color="orange", zorder=0)
    ax.axhspan(0,   50, alpha=0.06, color="red",    zorder=0)

    # TM-score annotation
    tm_mlm = {"TRPV3": 0.9808, "KCNK9": 0.9520, "HBB": 0.9728}[gene]
    tm_dam = {"TRPV3": 0.9808, "KCNK9": 0.9520, "HBB": 0.9714}[gene]
    ax.text(0.97, 0.04,
            f"TM (MLM) = {tm_mlm:.4f}\nTM (DAM) = {tm_dam:.4f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))

    ax.set_title(f"{gene}  ·  {NOTES[gene]}", fontsize=9)
    ax.set_xlabel("Residue position", fontsize=8)
    ax.set_xlim(1, len(ref))
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    ax.tick_params(labelsize=8)

axes[0].set_ylabel("pLDDT", fontsize=8)
axes[0].legend(fontsize=7.5, loc="upper right", framealpha=0.85)

fig.text(0.5, -0.04,
         "pLDDT bands: very high ≥90 (green) · high 70–90 · low 50–70 · very low <50 · "
         "damage applied at 10–30× authentic PMD rates",
         ha="center", fontsize=7, color="#555555")

plt.tight_layout()
fig.savefig(str(OUT) + ".pdf", bbox_inches="tight", dpi=300)
fig.savefig(str(OUT) + ".png", bbox_inches="tight", dpi=150)
print(f"Structural figure → {OUT}.{{pdf,png}}")
