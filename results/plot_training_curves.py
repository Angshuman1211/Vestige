"""
Figure 2: Training and validation loss curves for MLM baseline and DAM proposed runs.

Run from project root after both training runs complete:
    python results/plot_training_curves.py

Reads trainer_state.json from the final checkpoint of each run.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt

RUNS = {
    "MLM baseline": "training/checkpoints/mlm-baseline/checkpoint-700/trainer_state.json",
    "DAM proposed": "training/checkpoints/dam-proposed/checkpoint-700/trainer_state.json",  # final epoch; best is 595 but 700 has full history
}
COLORS = {
    "MLM baseline": {"train": "#2980b9", "eval": "#1a5276"},
    "DAM proposed": {"train": "#c0392b", "eval": "#922b21"},
}
OUT = Path("results/figures")


def parse_history(path):
    with open(path) as f:
        state = json.load(f)
    train, eval_ = {}, {}
    for entry in state["log_history"]:
        ep = round(entry.get("epoch", 0), 1)
        if "loss" in entry:
            train[ep] = entry["loss"]
        if "eval_loss" in entry:
            eval_[ep] = entry["eval_loss"]
    best = state["best_metric"]
    best_ckpt = state["best_model_checkpoint"]
    return train, eval_, best, best_ckpt


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)

for run_name, state_path in RUNS.items():
    p = Path(state_path)
    if not p.exists():
        print(f"Skipping {run_name} — {state_path} not found")
        continue

    train, eval_, best, best_ckpt = parse_history(state_path)
    c = COLORS[run_name]

    train_epochs = sorted(train); train_vals = [train[e] for e in train_epochs]
    eval_epochs  = sorted(eval_);  eval_vals  = [eval_[e]  for e in eval_epochs]

    ax1.plot(train_epochs, train_vals, color=c["train"], linewidth=1.5, label=run_name)
    ax2.plot(eval_epochs,  eval_vals,  color=c["eval"],  linewidth=1.5, label=f"{run_name}  (best={best:.4f})")

for ax, title, xlabel in [
    (ax1, "Training Loss", "Epoch"),
    (ax2, "Validation Loss", "Epoch"),
]:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Loss", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=9, frameon=False)

ax2.legend(fontsize=9, frameon=False, loc="lower left")  # move legend clear of the callout

# Callout (verified from trainer_state.json: DAM leads at every epoch; gap grows to ~epoch 8 then holds)
ax2.text(
    0.97, 0.94,
    "DAM's validation loss is lower at every epoch;\n"
    "the advantage widens as the empirical prior\n"
    "gains signal, then holds steady.",
    transform=ax2.transAxes, ha="right", va="top", fontsize=8,
    bbox=dict(boxstyle="round,pad=0.4", facecolor="#fdf2e9",
              edgecolor="#c0392b", linewidth=1.0, alpha=0.95),
)

fig.suptitle(
    "DNABERT-2 fine-tuning · TRPV3 / KCNK9 / HBB · Elephas maximus reference",
    fontsize=9, color="gray", y=1.01,
)
plt.tight_layout()

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "fig2_training_curves.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig2_training_curves.png", dpi=300, bbox_inches="tight")
print("Saved results/figures/fig2_training_curves.{pdf,png}")
