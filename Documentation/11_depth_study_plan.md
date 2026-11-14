# 11 — Depth-Study Program (TCBB resubmission)

**Goal:** convert the reviewers' recurring verdict ("simple trick, insufficient depth") into a demonstrated **general principle** — damage-aware masking helps whenever training-time masking matches a position-dependent corruption distribution, independent of architecture and corruption domain. Three independence axes carry the argument.

**Scope agreed (2026-11-05):** both backbones (NT + GENA-LM), both D3 domains (synthetic + FFPE), paid compute allowed (RunPod ~$0.30/hr) for the heavier backbone runs.

**Status key:** ☐ todo · ◐ in progress · ☑ done

---

## Phase 0 — Headline robustness (gate) ☑ PASSED (2026-11-05)
The 626-window T_END sweep (p<0.001, MLM-below-random at T_END=3) was single-seed. Multi-seeded it across 42/1/2.
- ☑ `evaluation/evaluate_scaling_multiseed.py` → `table_scaling_multiseed.txt`. **Headline SURVIVES decisively**: Δ(DAM−MLM) positive and p<0.001 at EVERY T_END on EVERY seed. T_END=3 pooled Δ=+0.103, p=3.6e-13; all T_END `all_seeds_sig_001=True`. The earlier "borderline +0.040, p=0.049" was only the 69-window authentic subset (underpowered); on the full 626-window eval the effect is larger AND hugely significant. Foundation is solid.
- ☐ Reconcile the 38.6% vs 36.3% DAM terminal mismatch (paper's checkpoint-595 vs the fresh benchmark retrain). Pick one model set, report it consistently. (Deferred to Phase 4 rigor pass — the multiseed models are the self-consistent set to standardize on.)
**Gate verdict:** PASS → proceed to D2/D3.

## Phase 1 — D2, second + third backbone (architecture independence) ☑ COMPLETE (THREE backbones: DNABERT-2 + GENA-LM + NT)
Retrain MLM vs DAM on models from other labs/tokenizers. DAM beating MLM at terminal on all three → not a DNABERT-2 artifact.
- ☑ `build_dataset.py` parametrized by tokenizer (env `VESTIGE_MODEL`/`VESTIGE_OUT`). GENA-LM dataset built (`training/dataset_genalm`, 275/69, identical windows).
- ☑ **P25 discovery + fix:** the original collator's "first base" test is a DNABERT-2 accident (73% C/G-first there vs 4% on GENA-LM) → built `masking/collator_dam_offset.py` (offset-aware, tokenizer-agnostic). **Pay-off test 1 PASSED**: fair ~0.145 budget + identical terminal gradient on both backbones.
- ☑ **Pay-off test 2 (gate for the full rebuild) PASSED (2026-11-05, P27):** 1-seed GENA-LM (MLM vs DAM-offset, loaded via the P26 `AutoModel` fix) + terminal eval. DAM beats MLM at the terminal: Δ **+0.101 / +0.080 / +0.039** at 3/5/10 nt, all p<1e-12, CIs exclude 0; edge largest at the most-damaged extreme; equal overall eval_loss (5.795 vs 5.786). Architecture independence demonstrated → gate cleared. `evaluate_genalm_terminal.py` → `results_genalm_terminal.json`. Single-seed + terminal-only: full rollout must add multi-seed and background NRR (caveats in P27).
- ◐ Full rollout (gate cleared): multi-seed GENA-LM (seeds 1,2) + background NRR; **re-verify/retrain DNABERT-2 DAM with the offset-aware collator** so all backbones share one principled method; then NT.
  - ☑ **GENA-LM background check DONE (seed 42, `evaluate_genalm_background.py`):** no collapse — Δ(DAM−MLM) = +0.009/+0.010/+0.011 on a 3/5/10 nt central band (vs pmd-mask's −0.134). DAM improves C/G recovery everywhere, most at the ends → "balanced reconstructor" confirmed on GENA-LM. Second caveat (terminal-only) removed.
  - ☑ **Multi-seed GENA-LM DONE (2026-11-06, seeds 42/1/2, `evaluate_genalm_multiseed.py`):** every seed same sign; pooled terminal Δ +0.110/+0.087/+0.048 (p 2e-17…1e-18), pooled background Δ +0.015/+0.015/+0.014 (p 6e-4…9e-9, positive not negative). **GENA-LM CLOSED — paper-ready 2nd backbone** (terminal win + no background cost + multi-seed + principled collator).
  - ☑ **Re-verify DNABERT-2 DAM under the offset-aware collator DONE (2026-11-07, P28, seeds 42/1/2, `results_dnabert_offset_multiseed.json`).** Pooled terminal Δ **+0.067/+0.045/+0.034** (p 4.9e-12…1.9e-13) + pooled background Δ **+0.029/+0.026/+0.021** (p 7.6e-7…8.2e-11), every seed same sign, no collapse. The original DNABERT-2 result is NOT a first-base artifact (P25) — it survives the principled collator. Both backbones now share ONE method and show the same balanced-reconstructor pattern. Nuance: offset-aware terminal edge (+0.067 @3nt) < first-base (+0.103) but adds the background gain — standardize the paper on the offset-aware set in Phase 4. Models in `multiseed_models/dam-offset-s{42,1,2}/`; MLM arm reused from `mlm-baseline-s*`.
  - ☑ **Nucleotide Transformer (3rd backbone) DONE 2026-11-08 (P29, seeds 42/1/2, `results_nt_multiseed.json`).** `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species` (ESM-style, non-overlapping 6-mer tokenizer), all 6 arms trained fresh (no reusable MLM). Pooled TERMINAL Δ **+0.233/+0.173/+0.112** (p 1e-44…3e-52) + pooled BACKGROUND Δ **+0.018/+0.035/+0.050** (all positive, no collapse), every seed same sign. Largest terminal edge of the three backbones (a terminal base sits inside a fixed 6-mer so uniform MLM rarely targets it; DAM does). Loads via `AutoModelForMaskedLM`→`EsmForMaskedLM` (no `model_class` override). Offsets rebuilt by cumsum over 6-mer token-string lengths (exact: all windows pure ACGT, coverage guard enforced). Models `training/multiseed_models/{mlm-nt,dam-offset-nt}-s{42,1,2}/` (gitignored). **Three-backbone table = Phase-4 headline (see P29).**

## Phase 2 — D3, second corruption domain (domain + principle independence) ☑ COMPLETE (synthetic P30 + FFPE P31)
Same backbone (DNABERT-2), same eval — only the damage profile changes. Biggest payoff, cheapest.
- ☑ **Synthetic corruption-shape sweep DONE (2026-11-08, P30, `results_synthetic_multiseed.json`).** `PositionProfileCollator` masks C/G by an arbitrary per-position weight; two deliberately UN-terminal shapes (central Gaussian bump, 50 bp periodic), DAM-matched vs reused uniform MLM, seeds 42/1/2. Pooled Δ(DAM−MLM) in the corruption zone: **central +0.0437** (p=5.1e-45), **periodic +0.0248** (p=4.1e-105), every seed positive, CIs clear of 0. Principle is corruption-shape-agnostic → the domain/corruption-shape independence axis to pair with 3-backbone architecture independence.
- ☑ **FFPE real clinical domain DONE (2026-11-09, P31, `results_ffpe_multiseed.json`).** FFPE deamination is C→T like aDNA but NEAR-UNIFORM along the read (not terminal) — `make_ffpe_profile.py` → `damage/synthetic/ffpe.npy`, fed to the same `PositionProfileCollator`. Pooled Δ(DAM−MLM) **+0.0655** (p=3e-130, CI [+0.061,+0.070], n=626), every seed positive — the LARGEST D3 effect. DAM generalises to a real clinical sequencing domain with no positional structure → strongest "not just ancient DNA" evidence. Caveat: literature-SHAPED profile (near-uniform), not a specific FFPE-BAM mapDamage run (shape is what matters; real-BAM is the stronger version on request).

## Phase 3 — Tradeoff characterization (own the pmd-mask result) ☑ DONE 2026-11-08 (P30) — OUTCOME NEGATIVE, reframed
- ☑ Swept the DAM concentration knob (power 1/2/4) soft→hard vs MLM and pmd-mask (`results_tradeoff_multiseed.json`). **The knob is INERT** — p1≈p2≈p4 on both terminal and background; no continuum. Mechanism (verified): susceptible termini are <1% of tokens, so even power 4 puts only 3.7% of the 15% budget at the ends → soft-DAM can't reach pmd-mask's terminal-only regime. **AND P24 is REVISED: pmd-mask does NOT collapse on background** under the offset-aware multiseed protocol (pmd bg 0.390 ≈/> DAM 0.401 @3), so the "DAM is the balanced champion vs pmd" framing is dead. **Decision (user 2026-11-08): pmd-mask = REFERENCE POINT, not rival.** DAM stands on DAM>MLM (3 backbones + corruption-shape); pmd is a domain-specific terminal-only reference that can't address the D3 synthetic shapes. Drop all "pmd collapses on background" claims. Tradeoff curve is NOT a paper figure as a continuum. (DAM>MLM itself is untouched — p1 Δterm +0.067/Δbg +0.029 @3.)

## Phase 4 — Rigor pass ◐ (audit DONE 2026-11-08 → `Documentation/12_phase4_rigor.md`; manuscript pending, user-led)
- ☑ Audit complete. One consistent model set frozen (offset-aware multiseed, all 3 backbones); every headline number traced to a reproducible JSON; first-base `dam-proposed` set RETIRED, which resolves the 38.6/36.3 mismatch. Authoritative numbers + manuscript change-list = `Documentation/12_phase4_rigor.md`.
- ⚠ **Key finding:** the absolute "DAM beats random at all T_END" claim does NOT survive the switch to the offset-aware set — DNABERT-2 DAM@3 = 0.273 ≈ random 0.277 (−0.004), clears random only at T_END≥5. Δ(DAM−MLM) untouched (+0.067, p<1e-11). Reframe (✅ user 2026-11-08, KEEP reframed): lead on Δ + architecture/corruption-shape independence; keep "MLM below random at the damaged terminus" as the motivation; state DAM restores terminal recovery to ≈random at T_END=3 and above random at T_END≥5. Do NOT claim DAM beats chance at every T_END. Locked in rigor doc §4.
- ☐ Multi-seed + pooled throughout = already satisfied by the canonical set (P27–P30). Remaining = apply the §7 change-list to the manuscript (user writes prose, blueprints first).

---

## Reserved for Angshuman (hands-on builds)
- Phase 2 synthetic-corruption generator (`evaluation/` + a profile factory).
- Phase 3 masking-concentration sweep + tradeoff-curve figure.

## Compute notes
- DNABERT-2 runs: Colab/Kaggle T4 (~16 min/run) as before.
- NT / larger backbones: RunPod A10G or similar for speed + no disconnects. Scope NT to a Colab-feasible size (50M/100M) only if paid compute is unavailable that day.

## Prior-work references to add (from the resubmission plan)
Dynamic/RoBERTa masking, SpanBERT, adaptive/PMI masking, pmd-mask (already benchmarked, P24), plus the second-backbone citations (Nucleotide Transformer, GENA-LM) and an FFPE-damage reference for Phase 2.
