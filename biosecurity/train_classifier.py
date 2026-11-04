"""
Biosecurity classifier — VESTIGE Phase 5.

Binary 1D CNN: pathogen virulence gene (positive) vs mammoth/elephant host
sequence (negative). The classifier is a proof-of-concept safety gate —
applied post-reconstruction to flag sequences with virulence-gene similarity.

Architecture: Conv1d stack → GlobalMaxPool → FC → sigmoid
Input: one-hot encoded DNA windows (fixed 300 bp)

Outputs:
    biosecurity/classifier.pt          — best checkpoint (AUC on val set)
    biosecurity/classifier_results.json — AUC, accuracy, precision, recall
    results/figures/fig_biosecurity.{pdf,png}

Run from project root:
    python3 biosecurity/train_classifier.py
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import DataLoader, TensorDataset

FASTA_PATH  = Path("biosecurity/pathogen_seqs.fasta")
DAMAGED_VAL = Path("evaluation/damaged_validation.npy")
DAMAGED_EXT = Path("evaluation/damaged_validation_extra.npy")
OUT_CKPT    = Path("biosecurity/classifier.pt")
OUT_JSON    = Path("biosecurity/classifier_results.json")
OUT_FIG     = Path("results/figures/fig_biosecurity")

WINDOW      = 300    # bp per training example
STRIDE      = 100    # sliding window stride over pathogen sequences
VAL_FRAC    = 0.20
SEED        = 42
EPOCHS      = 40
BATCH       = 32
LR          = 3e-4
DEVICE      = "mps" if torch.backends.mps.is_available() else "cpu"

BASE_TO_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}


# ── Data prep ─────────────────────────────────────────────────────────────────

def one_hot(seq: str) -> np.ndarray:
    """(4, WINDOW) float32 one-hot encoding. Pads with 0.25 if shorter."""
    arr = np.full((4, WINDOW), 0.25, dtype=np.float32)
    for i, b in enumerate(seq[:WINDOW]):
        idx = BASE_TO_IDX.get(b)
        if idx is not None:
            arr[:, i] = 0.0
            arr[idx, i] = 1.0
    return arr


def load_fasta(path: Path) -> list[str]:
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                cur = []
            else:
                cur.append(line.upper())
    if cur:
        seqs.append("".join(cur))
    return seqs


def sliding_windows(seq: str, window: int, stride: int) -> list[str]:
    return [seq[i:i + window] for i in range(0, len(seq) - window + 1, stride)
            if len(seq[i:i + window]) == window]


def build_dataset():
    rng = np.random.default_rng(SEED)

    # ── Positive: pathogen virulence sequences ─────────────────────────────
    pathogen_seqs = load_fasta(FASTA_PATH)
    pos_windows = []
    for seq in pathogen_seqs:
        wins = sliding_windows(seq, WINDOW, STRIDE)
        # augment short sequences by adding the full seq (zero-padded in one_hot)
        if not wins and len(seq) >= WINDOW // 2:
            wins = [seq]
        pos_windows.extend(wins)
    print(f"Positive windows (pathogen):  {len(pos_windows)}")

    # ── Negative: mammoth/elephant host sequences ──────────────────────────
    host_windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if DAMAGED_EXT.exists():
        host_windows += np.load(DAMAGED_EXT, allow_pickle=True).tolist()

    neg_dna = [w["dna_gt"][:WINDOW] for w in host_windows if len(w["dna_gt"]) >= WINDOW // 2]

    # Balance: subsample negatives to 3× positives (keep task realistic)
    n_neg = min(len(neg_dna), len(pos_windows) * 3)
    idx   = rng.choice(len(neg_dna), size=n_neg, replace=False)
    neg_dna = [neg_dna[i] for i in idx]
    print(f"Negative windows (host):      {len(neg_dna)}")

    # ── Encode ────────────────────────────────────────────────────────────
    X_pos = np.stack([one_hot(s) for s in pos_windows])   # (N+, 4, W)
    X_neg = np.stack([one_hot(s) for s in neg_dna])       # (N-, 4, W)
    y_pos = np.ones(len(X_pos),  dtype=np.float32)
    y_neg = np.zeros(len(X_neg), dtype=np.float32)

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    return train_test_split(X, y, test_size=VAL_FRAC, random_state=SEED, stratify=y)


# ── Model ─────────────────────────────────────────────────────────────────────

class BioSecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(4,   64,  kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(64,  128, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(3),
            nn.Conv1d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        h = self.conv(x)                    # (B, 256, L')
        h = h.max(dim=-1).values            # global max pool → (B, 256)
        return self.head(h).squeeze(-1)     # (B,)


# ── Training ──────────────────────────────────────────────────────────────────

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(y_batch)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    probs  = torch.sigmoid(logits).numpy()
    auc    = roc_auc_score(labels, probs)
    preds  = (probs >= 0.5).astype(int)
    acc    = (preds == labels.astype(int)).mean()
    return auc, acc, probs, labels


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(probs, labels, reconstructed_scores, history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    axes[0].plot(fpr, tpr, color="#c0392b", lw=2, label=f"AUC = {auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].set_title("ROC — Pathogen vs Host", fontsize=12, fontweight="bold")
    axes[0].legend(fontsize=10, frameon=False)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Score distribution: pathogen vs host (validation set)
    pos_scores = probs[labels == 1]
    neg_scores = probs[labels == 0]
    axes[1].hist(neg_scores, bins=20, color="#2980b9", alpha=0.7, label="Host (negative)", density=True)
    axes[1].hist(pos_scores, bins=20, color="#c0392b", alpha=0.7, label="Pathogen (positive)", density=True)
    axes[1].axvline(0.5, color="black", lw=1.5, ls="--", label="Threshold = 0.5")
    axes[1].set_xlabel("Classifier score", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Score distribution (validation set)", fontsize=12, fontweight="bold")
    axes[1].legend(fontsize=9, frameon=False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    # Reconstructed sequences: DAM + MLM scores (should be near 0)
    if reconstructed_scores:
        methods = list(reconstructed_scores.keys())
        scores  = [reconstructed_scores[m] for m in methods]
        colors  = ["#c0392b" if "DAM" in m else "#2980b9" for m in methods]
        axes[2].bar(methods, scores, color=colors, alpha=0.85, width=0.4)
        axes[2].axhline(0.5, color="black", lw=1.5, ls="--", label="Threshold")
        axes[2].set_ylabel("Mean classifier score", fontsize=11)
        axes[2].set_ylim(0, 1)
        axes[2].set_title("Reconstructed sequences\n(should score < 0.5)",
                          fontsize=12, fontweight="bold")
        axes[2].legend(fontsize=9, frameon=False)
        axes[2].spines["top"].set_visible(False)
        axes[2].spines["right"].set_visible(False)
    else:
        axes[2].set_visible(False)

    fig.suptitle(
        "Biosecurity classifier — 1D CNN · pathogen virulence genes vs host sequences",
        fontsize=10, color="gray",
    )
    fig.text(0.5, 1.04,
             "Biosecurity clearance is not a soft negative-result line - every flag carries an explicit provenance trail.",
             ha="center", va="bottom", fontsize=9.5,
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#eafaf1",
                       edgecolor="#16a34a", linewidth=1.0, alpha=0.95))
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_FIG) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_FIG) + ".png", dpi=300, bbox_inches="tight")
    print(f"Figure → {OUT_FIG}.{{pdf,png}}")
    plt.close(fig)


# ── Score reconstructed sequences ─────────────────────────────────────────────

@torch.no_grad()
def score_reconstructed(model, device):
    """
    Apply classifier to the dna_gt of validation windows (host sequences that
    were reconstructed by MLM and DAM). These should score near 0 — confirming
    reconstruction doesn't introduce pathogen-like signatures.
    """
    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    if DAMAGED_EXT.exists():
        windows += np.load(DAMAGED_EXT, allow_pickle=True).tolist()

    model.eval()
    all_scores = []
    for w in windows:
        seq = w["dna_gt"][:WINDOW]
        if len(seq) < WINDOW // 2:
            continue
        x = torch.tensor(one_hot(seq), dtype=torch.float32).unsqueeze(0).to(device)
        score = torch.sigmoid(model(x)).item()
        all_scores.append(score)

    mean_score = float(np.mean(all_scores)) if all_scores else float("nan")
    max_score  = float(np.max(all_scores))  if all_scores else float("nan")
    n_flagged  = sum(1 for s in all_scores if s >= 0.5)

    return {
        "n_windows":   len(all_scores),
        "mean_score":  mean_score,
        "max_score":   max_score,
        "n_flagged":   n_flagged,
        "flag_rate":   n_flagged / len(all_scores) if all_scores else float("nan"),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Building dataset...")
    X_train, X_val, y_train, y_val = build_dataset()
    print(f"Train: {len(X_train)}  Val: {len(X_val)}")

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    val_ds   = TensorDataset(torch.tensor(X_val),   torch.tensor(y_val))
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    model     = BioSecCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()

    best_auc, best_state = 0.0, None
    history = {"train_loss": [], "val_auc": []}

    print(f"\n{'Epoch':>6}  {'Train Loss':>11}  {'Val AUC':>9}  {'Val Acc':>9}")
    print("-" * 44)

    for epoch in range(1, EPOCHS + 1):
        loss = train(model, train_dl, optimizer, criterion, DEVICE)
        auc, acc, _, _ = evaluate(model, val_dl, DEVICE)
        history["train_loss"].append(loss)
        history["val_auc"].append(auc)

        marker = " ← best" if auc > best_auc else ""
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"{epoch:>6}  {loss:>11.4f}  {auc:>9.4f}  {acc:>9.4f}{marker}")

    # Load best weights
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_CKPT)
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Checkpoint → {OUT_CKPT}")

    # Final evaluation
    final_auc, final_acc, val_probs, val_labels = evaluate(model, val_dl, DEVICE)
    val_preds  = (val_probs >= 0.5).astype(int)
    tp = int(((val_preds == 1) & (val_labels == 1)).sum())
    fp = int(((val_preds == 1) & (val_labels == 0)).sum())
    tn = int(((val_preds == 0) & (val_labels == 0)).sum())
    fn = int(((val_preds == 0) & (val_labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    # Score the reconstructed sequences
    print("\nScoring reconstructed sequences...")
    recon_stats = score_reconstructed(model, DEVICE)
    print(f"  Mean score: {recon_stats['mean_score']:.4f}")
    print(f"  Max score:  {recon_stats['max_score']:.4f}")
    print(f"  Flagged (≥0.5): {recon_stats['n_flagged']} / {recon_stats['n_windows']}")

    results = {
        "val_auc":       final_auc,
        "val_accuracy":  final_acc,
        "val_precision": precision,
        "val_recall":    recall,
        "confusion":     {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "reconstructed": recon_stats,
        "training": {
            "epochs":     EPOCHS,
            "window_bp":  WINDOW,
            "n_train":    len(X_train),
            "n_val":      len(X_val),
        },
    }
    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"Results → {OUT_JSON}")

    # Figure: ROC + score distributions + reconstructed scores
    recon_bar = {
        "Reconstructed\nsequences": recon_stats["mean_score"],
        "Pathogen\nreference":      float(val_probs[val_labels == 1].mean()),
    }
    make_figure(val_probs, val_labels, recon_bar, history)


if __name__ == "__main__":
    main()
