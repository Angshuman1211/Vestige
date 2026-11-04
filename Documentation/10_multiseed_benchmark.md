# 10 — Multi-Seed Masking-Strategy Benchmark

Response to the TCBB editorial reject ("needs more comparison with existing methods" + "more depth"). I benchmark DAM against the masking strategies it is actually competing with, and I confirm every claim across three training seeds so no result rests on one lucky initialisation.

This is a **separate, lower-powered evaluation** from the 626-window terminal sweep in `06_evaluation.md` / the README (3 training genes, validation split, T_END=10, authentic PMD rates). That sweep stands unchanged. This benchmark adds the baselines and the seed-robustness.

## Setup

- **Arms (5):** MLM (uniform 15%), DAM linear (proposed), DAM smoothed, DAM power (freq²), pmd-mask (hard top-k on the most-damaged C/G — the direct aDNA prior art).
- **Seeds:** 42, 1, 2. All retrained on Colab T4 (one environment, so a cross-seed difference cannot be a hardware artefact).
- **Weights:** `training/multiseed_models/<arm>-s<seed>/pytorch_model.bin`.
- **Eval:** `evaluation/evaluate_multiseed.py` — terminal C/G recovery (69 windows, T_END=10) and authentic-damage background recovery (43 windows), identical metric code to the single-seed scripts.
- **Stats:** per-seed paired t-test, then a **pooled across-seed test** — seed-averaged per-window difference, one-sample t-test + Wilcoxon + bootstrap CI, with windows as the replication unit (seed is training noise to average out). A MixedLM was wired as a confirmatory but is blocked by a statsmodels 0.14 / scipy 1.17 incompatibility; the seed-averaged test is the cleaner primary analysis regardless.
- **Outputs:** `evaluation/results_multiseed.json`, `evaluation/table_multiseed.txt`.

## Pooled result (all vs MLM)

**Terminal C/G (n = 69 windows):**

| Arm | mean Δ | t-test p | Wilcoxon p | 95% CI |
|-----|--------|----------|------------|--------|
| DAM (linear) | +0.040 | 0.049 | 0.032 | [+0.004, +0.080] |
| DAM smoothed | +0.043 | 0.034 | 0.016 | [+0.007, +0.082] |
| DAM power | +0.047 | 0.015 | 0.012 | [+0.012, +0.084] |
| pmd-mask (hard) | +0.164 | 3.7e-5 | 3.6e-5 | [+0.093, +0.238] |

**Background authentic-damage (n = 43 windows):**

| Arm | mean Δ | t-test p | 95% CI |
|-----|--------|----------|--------|
| DAM (linear) | −0.027 | 0.49 (ns) | [−0.103, +0.044] |
| DAM power | +0.008 | 0.82 (ns) | [−0.061, +0.081] |
| pmd-mask (hard) | −0.134 | 0.025 | [−0.245, −0.025] |

## What this establishes

1. **DAM > MLM on terminal is seed-robust** — positive on all three seeds for every variant; pooled p = 0.049 (t) / 0.032 (Wilcoxon) for linear DAM, CI excludes zero. Real, but **modest** (~4 points) and borderline on this small authentic test. The single-seed seed-42 result that made linear DAM look dead (p = 0.44) was a fluctuation.

2. **DAM pays no background cost** — all DAM variants are statistically indistinguishable from MLM at background positions.

3. **pmd-mask beats DAM on terminal but collapses on background** — the hard-threshold baseline wins the terminal metric by a wide margin (+0.164) because it concentrates even more masking there, and pays for it with a significant background loss (−0.134). DAM is therefore not the best terminal reconstructor; it is the **balanced** point on the soft→hard masking-concentration spectrum. The paper's contribution is best framed as characterising that tradeoff, with DAM as the operating point that improves terminal recovery without sacrificing general infilling.

4. **The mapping shape does not matter** — linear, smoothed, and power-mapped DAM are equivalent within noise (+0.040 / +0.043 / +0.047). Linear DAM is the method; smoothed and power are reported as an insensitivity ablation. There is no "power map is special" effect (that was seed-42 noise).

## To reconcile before resubmission

The fresh-retrained DAM here recovers 36.3% at terminal on seed 42, while `06_evaluation.md` / the README report the paper's DAM at 38.6% (a different training run / hardware / checkpoint-595). The multi-seed models are fresh retrains used for the robustness study. Decide which set of numbers the resubmission reports, and keep it consistent.
