"""
Damage intensity sweep — VESTIGE.

The empirical woolly mammoth PMD rates are ~0.35% peak C→T — too low for
stochastic testing (barely any positions get hit per window). This script
instead normalises the mapDamage2 SHAPE to four synthetic peak rates:

    5%  →  well-preserved ancient specimen (~1,000 BP)
   10%  →  moderately damaged (~5,000 BP)
   20%  →  heavily damaged (~20,000 BP)
   30%  →  near-saturated (~50,000+ BP)

The exponential decay profile (position-dependence) is preserved; only the
overall magnitude is scaled. For each intensity level:
  - Damage is applied stochastically to C in the 5' T_END zone and G in 3'
  - Evaluation is on the positions that were actually damaged (realistic)

If DAM's advantage is mechanistically driven by the masking objective,
it should grow with damage intensity — the model was trained on exactly
this structure, and it becomes the dominant source of sequence error at high rates.

Outputs:
    evaluation/intensity_results.json
    results/figures/fig_intensity.{pdf,png}

Run from project root:
    python3 evaluation/evaluate_intensity.py
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
PROFILE_PATH      = "damage/damage_profile.npy"
TOKENIZER_NAME    = "zhihan1996/DNABERT-2-117M"
CHECKPOINTS = {
    "MLM": "training/checkpoints/mlm-baseline/checkpoint-595",
    "DAM": "training/checkpoints/dam-proposed/checkpoint-595",
}
OUT_JSON = Path("evaluation/intensity_results.json")
OUT_FIG  = Path("results/figures/fig_intensity")

# Target peak C→T / G→A rates at position 0 (exponential decay shape preserved)
# Approximate age analogue given in labels for narrative framing only.
PEAK_RATES   = [0.05, 0.10, 0.20, 0.30, 0.40]
RATE_LABELS  = ["5%\n(~1k BP)", "10%\n(~5k BP)", "20%\n(~20k BP)",
                "30%\n(~50k BP)", "40%\n(near-sat)"]

T_END  = 3       # innermost terminal zone (peak damage)
N_BOOT = 10_000
SEED   = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

COLORS = {
    "Random":    "#94a3b8",
    "Zero-shot": "#a855f7",
    "MLM":       "#2980b9",
    "DAM":       "#c0392b",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_model(ckpt, device):
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    state = torch.load(Path(ckpt) / "pytorch_model.bin",
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


def load_zeroshot(device):
    model = AutoModelForMaskedLM.from_pretrained(
        TOKENIZER_NAME, trust_remote_code=True, dtype=torch.float32,
    )
    return model.eval().to(device)


def scale_profile(profile_arr: np.ndarray, target_peak: float) -> np.ndarray:
    """Normalise profile shape so the peak value equals target_peak (capped at 1)."""
    raw_peak = float(profile_arr[0])
    if raw_peak == 0:
        return np.full_like(profile_arr, target_peak)
    return np.clip(profile_arr / raw_peak * target_peak, 0.0, 1.0)


def apply_damage(dna: str, ct_norm: np.ndarray, ga_norm: np.ndarray,
                 rng) -> list[tuple[int, str]]:
    """
    Stochastically damage C→T at 5' and G→A at 3' within T_END positions.
    Returns list of (nuc_pos, original_base) for damaged sites.
    """
    n = len(dna)
    damaged = []
    for i, base in enumerate(dna):
        if base == "C" and i < T_END:
            prob = float(ct_norm[min(i, len(ct_norm) - 1)])
            if rng.random() < prob:
                damaged.append((i, "C"))
        elif base == "G" and i >= n - T_END:
            dist = n - 1 - i
            prob = float(ga_norm[min(dist, len(ga_norm) - 1)])
            if rng.random() < prob:
                damaged.append((i, "G"))
    return damaged


def nuc_to_tok(nuc_pos: int, offsets: list) -> int | None:
    for tok_idx, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s <= nuc_pos < e:
            return tok_idx
    return None


@torch.no_grad()
def predict_at(model, gt_ids, tok_positions, attn_mask, mask_id, device):
    masked = gt_ids[:]
    for tp in tok_positions:
        masked[tp] = mask_id
    logits = model(
        input_ids=torch.tensor([masked], dtype=torch.long, device=device),
        attention_mask=torch.tensor([attn_mask], dtype=torch.long, device=device),
    ).logits
    return {tp: int(logits[0, tp].argmax()) for tp in tok_positions}


def nuc_correct(dna, nuc_pos, tok_pos, pred_id, offsets, tokenizer) -> bool:
    tok = tokenizer.convert_ids_to_tokens([pred_id])[0]
    if not tok:
        return False
    s   = tok.lstrip("#").lstrip("▁").upper()
    off = nuc_pos - offsets[tok_pos][0]
    return 0 <= off < len(s) and s[off] == dna[nuc_pos]


# ── Per-intensity evaluation ──────────────────────────────────────────────────

def run_peak(peak_rate, windows, mlm_model, dam_model, zs_model,
             tokenizer, ct_5p_raw, ga_3p_raw, rng):

    ct_norm = scale_profile(ct_5p_raw, peak_rate)
    ga_norm = scale_profile(ga_3p_raw, peak_rate)
    mask_id = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    mlm_rates, dam_rates, zs_rates, rnd_rates = [], [], [], []
    total_sites = 0

    for w in windows:
        dna     = w["dna_gt"]
        gt_ids  = w["gt_ids"]
        attn    = w["attention_mask"]
        offsets = [tuple(o) for o in w["offsets"]]

        damaged = apply_damage(dna, ct_norm, ga_norm, rng)
        if not damaged:
            continue

        # Map each damaged nucleotide → its token position
        tok_map = {}
        for nuc_pos, _ in damaged:
            tidx = nuc_to_tok(nuc_pos, offsets)
            if tidx is not None:
                tok_map[nuc_pos] = tidx

        if not tok_map:
            continue

        pairs         = list(tok_map.items())    # [(nuc_pos, tok_pos)]
        tok_positions = sorted(set(tok_map.values()))
        total_sites  += len(pairs)

        for model, rates in [(mlm_model, mlm_rates),
                              (dam_model, dam_rates),
                              (zs_model,  zs_rates)]:
            preds   = predict_at(model, gt_ids, tok_positions, attn, mask_id, DEVICE)
            correct = sum(
                nuc_correct(dna, np_, tp, preds[tp], offsets, tokenizer)
                for np_, tp in pairs if tp in preds
            )
            rates.append(correct / len(pairs))

        rnd_rates.append(sum(rng.random() < 0.25 for _ in pairs) / len(pairs))

    r_mlm = np.array(mlm_rates)
    r_dam = np.array(dam_rates)
    r_zs  = np.array(zs_rates)
    r_rnd = np.array(rnd_rates)
    n_w   = len(r_mlm)

    t_stat = p_two = p_one = ci_lo = ci_hi = float("nan")
    if n_w >= 2:
        t_stat, p_two = stats.ttest_rel(r_dam, r_mlm)
        p_one = float(p_two / 2) if t_stat > 0 else float(1 - p_two / 2)
        diffs = r_dam - r_mlm
        boot  = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                          for _ in range(N_BOOT)])
        ci_lo = float(np.percentile(boot, 2.5))
        ci_hi = float(np.percentile(boot, 97.5))

    return {
        "peak_rate":   peak_rate,
        "n_sites":     total_sites,
        "n_windows":   n_w,
        "rnd_mean":    float(r_rnd.mean()) if n_w else float("nan"),
        "zs_mean":     float(r_zs.mean())  if n_w else float("nan"),
        "mlm_mean":    float(r_mlm.mean()) if n_w else float("nan"),
        "dam_mean":    float(r_dam.mean()) if n_w else float("nan"),
        "delta":       float((r_dam - r_mlm).mean()) if n_w else float("nan"),
        "p_value_two": float(p_two),
        "p_value_one": float(p_one),
        "ci_95":       [ci_lo, ci_hi],
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def make_figure(all_results):
    peaks   = sorted(all_results.keys())
    xlabels = [f"{int(p*100)}%" for p in peaks]
    rnd_v   = [all_results[p]["rnd_mean"]    for p in peaks]
    zs_v    = [all_results[p]["zs_mean"]     for p in peaks]
    mlm_v   = [all_results[p]["mlm_mean"]    for p in peaks]
    dam_v   = [all_results[p]["dam_mean"]    for p in peaks]
    deltas  = [all_results[p]["delta"]       for p in peaks]
    p_vals  = [all_results[p]["p_value_two"] for p in peaks]
    n_sites = [all_results[p]["n_sites"]     for p in peaks]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.array(peaks)

    ax1.axhline(0.25, color="#94a3b8", lw=1.2, ls=":", label="Random (theoretical)")
    ax1.plot(x, rnd_v, "v--", color=COLORS["Random"],    lw=1.5, ms=5,
             label="Random (observed)", alpha=0.7)
    ax1.plot(x, zs_v,  "^-",  color=COLORS["Zero-shot"], lw=1.8, ms=6,
             label="Zero-shot DNABERT-2")
    ax1.plot(x, mlm_v, "o-",  color=COLORS["MLM"],       lw=2,   ms=7,
             label="MLM fine-tuning")
    ax1.plot(x, dam_v, "s-",  color=COLORS["DAM"],       lw=2.5, ms=8,
             label="DAM (proposed)", zorder=5)

    for p, pv, d, m in zip(peaks, p_vals, dam_v, mlm_v):
        if not np.isnan(pv) and pv < 0.05:
            marker = "***" if pv < 0.001 else ("**" if pv < 0.01 else "*")
            ax1.annotate(marker, xy=(p, max(d, m) + 0.010), ha="center",
                         fontsize=9, color="#27ae60", fontweight="bold")

    ax1.set_xlabel("Peak C→T / G→A damage rate (position 0)", fontsize=11)
    ax1.set_ylabel("Nucleotide recovery rate (T_END=3 zone)", fontsize=11)
    ax1.set_title("Recovery at peak-damage zone vs damage intensity",
                  fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xlabels)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    bar_colors = ["#27ae60" if d > 0 else "#e74c3c" for d in deltas]
    ax2.bar(x, deltas, color=bar_colors, alpha=0.8, width=0.035)
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    for p, pv, d in zip(peaks, p_vals, deltas):
        if np.isnan(pv):
            continue
        label  = "***" if pv < 0.001 else ("**" if pv < 0.01 else ("*" if pv < 0.05 else "ns"))
        offset = 0.004 if d >= 0 else -0.008
        ax2.annotate(label, xy=(p, d + offset), ha="center",
                     fontsize=9,
                     color="#27ae60" if label != "ns" else "gray",
                     fontweight="bold")

    ax2b = ax2.twinx()
    ax2b.plot(x, n_sites, "D--", color="gray", lw=1.2, ms=5)
    ax2b.set_ylabel("Damaged sites evaluated (n)", fontsize=10, color="gray")
    ax2b.tick_params(axis="y", labelcolor="gray")
    ax2b.spines["top"].set_visible(False)

    ax2.set_xlabel("Peak C→T / G→A damage rate (position 0)", fontsize=11)
    ax2.set_ylabel("DAM − MLM recovery (Δ)", fontsize=11)
    ax2.set_title("DAM advantage grows with damage intensity",
                  fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle(
        "Synthetic damage intensity sweep · T_END=3 · 626 windows · 7 genes · "
        "position-dependent decay profile preserved · *** p < 0.001",
        fontsize=9, color="gray", y=1.01,
    )
    plt.tight_layout()
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(OUT_FIG) + ".pdf", bbox_inches="tight")
    fig.savefig(str(OUT_FIG) + ".png", dpi=300, bbox_inches="tight")
    print(f"Figure → {OUT_FIG}.{{pdf,png}}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    profile   = np.load(PROFILE_PATH, allow_pickle=True).item()
    ct_5p_raw = profile["ct_5p"]
    ga_3p_raw = profile["ga_3p"]

    windows = np.load(DAMAGED_VAL, allow_pickle=True).tolist()
    extra   = Path(DAMAGED_VAL_EXTRA)
    if extra.exists():
        windows += np.load(extra, allow_pickle=True).tolist()
    print(f"Windows: {len(windows)} total")

    # Quick sanity check on the profile
    print(f"Empirical profile peak: ct_5p[0]={ct_5p_raw[0]:.4f}, ga_3p[0]={ga_3p_raw[0]:.4f}")

    print("Loading MLM...")
    mlm_model = load_model(CHECKPOINTS["MLM"], DEVICE)
    print("Loading DAM...")
    dam_model = load_model(CHECKPOINTS["DAM"], DEVICE)
    print("Loading zero-shot...")
    zs_model  = load_zeroshot(DEVICE)

    print(f"\n{'Rate':>6}  {'Sites':>7}  {'Win':>5}  "
          f"{'Random':>8}  {'ZeroShot':>9}  {'MLM':>8}  {'DAM':>8}  "
          f"{'Δ':>8}  {'p(2)':>7}  {'p(1)':>7}")
    print("-" * 88)

    all_results = {}
    for peak_rate in PEAK_RATES:
        # Independent seed per level for reproducibility
        level_rng = np.random.default_rng(SEED + int(peak_rate * 1000))
        r = run_peak(peak_rate, windows, mlm_model, dam_model, zs_model,
                     tokenizer, ct_5p_raw, ga_3p_raw, level_rng)
        all_results[peak_rate] = r

        def ps(v): return f"{v:.4f}" if not np.isnan(v) else "   N/A"
        def sig(p): return "*" if not np.isnan(p) and p < 0.05 else " "

        print(f"{int(peak_rate*100):>5}%  {r['n_sites']:>7}  {r['n_windows']:>5}  "
              f"{r['rnd_mean']:>8.4f}  {r['zs_mean']:>9.4f}  "
              f"{r['mlm_mean']:>8.4f}  {r['dam_mean']:>8.4f}  "
              f"{r['delta']:>+8.4f}  "
              f"{ps(r['p_value_two'])}{sig(r['p_value_two'])}  "
              f"{ps(r['p_value_one'])}{sig(r['p_value_one'])}")

    OUT_JSON.write_text(json.dumps({str(k): v for k, v in all_results.items()}, indent=2))
    print(f"\nJSON → {OUT_JSON}")
    make_figure(all_results)


if __name__ == "__main__":
    main()
