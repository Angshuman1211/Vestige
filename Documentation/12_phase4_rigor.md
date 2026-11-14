# Phase 4 — Rigor Pass (authoritative numbers + manuscript change-list)

Purpose: freeze ONE self-consistent model set, state every headline number with its
reproducible source, reconcile the old single-seed / first-base numbers the current
manuscript still carries, and list exactly what the manuscript must change. No GPU.
All depth-study GPU work is complete (P24–P30). The manuscript is written LAST, from
this document.

Status key: ✅ settled · ⚠ decision needed · ☐ to apply in the manuscript

---

## 1. Canonical model set (✅ decided, P28)

Standardize EVERYTHING on the **offset-aware multiseed set** — the `OffsetAwareDamageCollator`
(nucleotide-principled, tokenizer-agnostic), seeds 42/1/2, pooled stats (seed-averaged
per-window Δ, one-sample t + Wilcoxon + 10k-bootstrap CI, windows = replication unit).

| backbone | MLM arm | DAM arm | eval script | result JSON |
|---|---|---|---|---|
| DNABERT-2 | `mlm-baseline-s{42,1,2}` | `dam-offset-s{42,1,2}` | `evaluate_dnabert_offset_multiseed.py` | `results_dnabert_offset_multiseed.json` |
| GENA-LM | `genalm_models/mlm-s*` | `genalm_models/dam-s*` | `evaluate_genalm_multiseed.py` | `results_genalm_multiseed.json` |
| NT | `mlm-nt-s{42,1,2}` | `dam-offset-nt-s{42,1,2}` | `evaluate_nt_multiseed.py` | `results_nt_multiseed.json` |
| DNABERT-2 D3 synth | `mlm-baseline-s*` (reused) | `syn-{central,periodic}-s*` | `evaluate_synthetic_multiseed.py` | `results_synthetic_multiseed.json` |
| DNABERT-2 D3 FFPE | `mlm-baseline-s*` (reused) | `ffpe-s*` | `evaluate_ffpe_multiseed.py` | `results_ffpe_multiseed.json` |
| DNABERT-2 tradeoff | `mlm-baseline-s*` | `dam-offset{,-p2,-p4}-s*`, `pmdmask-baseline-s*` | `evaluate_tradeoff_multiseed.py` | `results_tradeoff_multiseed.json` |

RETIRED from the main results (do NOT cite as headline): the first-base `dam-proposed-s*`
set (`evaluate_scaling_multiseed.py` / `scaling_results.json`), `results_multiseed.json`
(69-window authentic subset, underpowered), and the paper's checkpoint-595 numbers. These
were the DNABERT-2-BPE first-base-of-token proxy (P25), not a general method. They may be
cited once, in a methods/ablation note, as "the original first-base proxy, superseded."

---

## 2. Headline table — architecture independence (✅, the Phase-4 lead)

Pooled Δ(DAM − MLM), three backbones, identical offset-aware collator. Absolute magnitudes
are NOT comparable across backbones (different tokenizers / vocab / pretraining) — only the
PATTERN is. Report the DELTA, not cross-backbone absolutes.

| backbone (lab / tokenizer / arch) | term Δ@3 (p, 95% CI) | term Δ@5 | term Δ@10 | bg Δ@3 | bg Δ@5 | bg Δ@10 |
|---|---|---|---|---|---|---|
| DNABERT-2 (Zhihan / BPE / MosaicBERT) | **+0.067** (4.9e-12, [+0.049,+0.086]) | +0.045 | +0.034 | +0.029 | +0.026 | +0.021 |
| GENA-LM (AIRI / BPE / pre-LN BERT) | **+0.110** (2.2e-17, [+0.086,+0.135]) | +0.087 | +0.048 | +0.015 | +0.015 | +0.014 |
| NT (InstaDeep / 6-mer / ESM) | **+0.233** (1.1e-44, [+0.204,+0.263]) | +0.173 | +0.112 | +0.018 | +0.035 | +0.050 |

Every seed same sign; terminal Δ largest at 3 nt and decaying inward (the damage gradient);
background Δ positive everywhere (no collapse). n = 486/565/619 (terminal 3/5/10),
600/622/626 (background 3/5/10).

> These tables regenerate verbatim from the JSONs via `evaluation/make_results_tables.py`
> → `evaluation/phase4_tables.md`. That generated file is the single source for the
> manuscript — copy numbers from it, never hand-type them.

## 3. D3 — corruption-shape AND domain independence (✅, second independence axis)

Pooled Δ(DAM-matched − uniform MLM) inside each profile's corruption zone (w[p] ≥ 0.5·max).
Two synthetic shapes (deliberately un-terminal) + one REAL clinical domain (FFPE).

| profile | zone n | per-seed Δ (42/1/2) | pooled Δ | p | 95% CI |
|---|---|---|---|---|---|
| central (synthetic, mid-window Gaussian) | 392 | +0.029/+0.062/+0.041 | **+0.0437** | 5.1e-45 | [+0.038,+0.049] |
| periodic (synthetic, 50 bp sin²) | 1000 | +0.023/+0.024/+0.027 | **+0.0248** | 4.1e-105 | [+0.023,+0.027] |
| FFPE (REAL clinical, near-uniform) | 2000 | +0.032/+0.093/+0.072 | **+0.0655** | 3.0e-130 | [+0.061,+0.070] |

DAM matched to a deliberately UN-terminal synthetic shape, AND to a REAL clinical FFPE
deamination domain whose damage is near-uniform (no positional structure), still beats uniform
MLM at recovering the damaged positions → the principle is corruption-shape- and domain-
agnostic, not ends-are-special. FFPE (P31) is the largest D3 effect: when damage is uniform,
"match the mask to the corruption distribution" reduces to "mask the susceptible C/G bases
everywhere", which beats uniform MLM (that wastes budget on A/T). FFPE caveat: the profile is
literature-SHAPED (near-uniform), not a specific FFPE-BAM mapDamage run — shape is what matters
(rescaled to the 15% budget); a real-BAM profile is the stronger version on reviewer request.

---

## 4. ⚠ FRAMING FINDING — the absolute "DAM beats random" claim does NOT survive (DNABERT-2)

The terminal `find_terminal` zone is byte-identical across the first-base and offset-aware
evals (same windows, same C-at-5'/G-at-3' rule), so these absolutes are directly comparable:

| T_END | MLM | DAM first-base (old paper) | DAM offset-aware (canonical) | random | zero-shot | DAM_offset − random |
|---|---|---|---|---|---|---|
| 3 | 0.205 | 0.308 | **0.273** | 0.277 | 0.155 | **−0.004** |
| 5 | 0.240 | 0.312 | 0.287 | 0.257 | 0.192 | +0.030 |
| 10 | 0.265 | 0.322 | 0.300 | 0.250 | 0.228 | +0.050 |

- Old headline ("MLM 20.5% < random 27.7% < DAM 30.8%, DAM exceeds random at ALL T_END")
  held ONLY because first-base DAM@3 = 0.308 > random.
- Offset-aware DAM@3 = 0.273 ≈ random 0.277 (−0.004) — it matches random at the extreme
  terminus, clearing it only from T_END ≥ 5. MLM stays well below random at T_END 3–5.
- The Δ(DAM − MLM) is identical in spirit and overwhelming (+0.067 @3, p<1e-11); only the
  ABSOLUTE "above chance" claim weakens.

Honest, defensible reframing (recommended):
- LEAD on Δ: "DAM significantly and consistently improves terminal C/G recovery over MLM
  across three architectures" (the Table-2 headline) — untouched, strong.
- Keep "MLM collapses below random at the damaged terminus" as the MOTIVATION (true: MLM@3
  0.206 ≪ random 0.277) — this is the problem DAM addresses.
- State the DAM-vs-random fact precisely: DAM restores terminal recovery to ≈ random at the
  extreme terminus (T_END=3) and above random inward (T_END ≥ 5). Do NOT claim DAM exceeds
  chance at every T_END.
✅ DECISION (user 2026-11-08): KEEP the "below random" story, reframed honestly. Lead the
results on Δ(DAM−MLM) + architecture/corruption-shape independence. Keep "MLM collapses
below random at the damaged terminus" (MLM@3 0.206 ≪ random 0.277) as the MOTIVATING problem.
State precisely that DAM restores terminal recovery to ≈ random at the extreme terminus
(T_END=3, 0.273 vs 0.277) and ABOVE random inward (T_END≥5: +0.030 @5, +0.050 @10). Do NOT
claim DAM exceeds chance at every T_END. (Rejected: demoting below-random entirely — the
"MLM is worse than guessing" motivation is kept.)

---

## 5. pmd-mask = reference point, not rival (✅, P30 + user 2026-11-08)

- The DAM power knob is INERT (p1≈p2≈p4): susceptible termini are <1% of tokens, so even
  power 4 puts only 3.7% of the 15% budget at the ends → soft-DAM cannot reach pmd-mask's
  terminal-only regime. No soft→hard continuum; the tradeoff curve is NOT a paper figure.
- P24 REVISED: under the offset-aware multiseed protocol, pmd-mask does NOT collapse on
  background (pmd bg 0.390 ≈/> DAM 0.401 @3; pmd terminal 0.514 ≫ DAM 0.273). DROP every
  "pmd-mask collapses on background" statement.
- Position pmd-mask as a domain-specific terminal-only reference built from the known aDNA
  damage profile — it specializes on the terminal at the cost of generality and cannot
  address the D3 synthetic shapes. DAM's contribution rests on DAM > MLM (§2, §3), NOT on
  beating pmd-mask.

## 6. 38.6% vs 36.3% checkpoint mismatch (✅ resolved)

Both were first-base numbers (paper's checkpoint-595 = 38.6%; fresh first-base retrain =
36.3%). Resolved by RETIRING the first-base set entirely (§1). The canonical DNABERT-2
terminal absolute is the offset-aware 0.273 @3 (§4). No mismatch remains once first-base is
dropped.

---

## 7. Manuscript change-list (☐ to apply, user writes prose — blueprint points only)

1. ☐ Swap every DNABERT-2 headline number from first-base (checkpoint-595 / scaling) to the
   offset-aware canonical set: terminal absolute 0.273 @3 (not 0.308/0.386), Δ +0.067.
2. ☐ Replace the single-backbone headline with the 3-backbone architecture-independence
   table (§2). This is the new lead result and the direct answer to the TCBB depth reject.
3. ☐ Add the D3 result (§3) as the second independence axis — now corruption-shape (synthetic
   central/periodic) AND domain (REAL FFPE clinical deamination, the largest effect +0.0655).
   FFPE replaces the paper's current "generalises to FFPE/clinical" assertion with evidence.
4. ☐ Reframe the "below random" claim per §4 (motivation = MLM below random; DAM restores to
   ≈random at T_END=3, above by T_END≥5). Remove "DAM exceeds random at all T_END".
5. ☐ Reframe pmd-mask as a reference point (§5); delete all "pmd collapses on background".
   Remove / do not include the tradeoff curve as a continuum figure.
6. ☐ Re-describe the method as the nucleotide-principled offset-aware collator (reconstructs
   each token's real C/G + absolute position); mention the first-base proxy only as the
   superseded original (P25).
7. ☐ Report the collator-independence nuance honestly (P28): offset-aware terminal edge
   (+0.067) < first-base (+0.103) but adds a background gain — balanced reconstructor.
8. ☐ State seeds (42/1/2) and pooling method once, apply to every number.

## 8. Reproducibility map (number → script → JSON) ✅

§2 deltas → the three `*_multiseed.py` scripts → the three `results_*_multiseed.json`.
§3 → `evaluate_synthetic_multiseed.py` → `results_synthetic_multiseed.json`.
§4 offset absolutes → `evaluate_tradeoff_multiseed.py` (mlm, p1) → `results_tradeoff_multiseed.json`;
references (random/zero-shot) → `scaling_results.json`. §4 first-base → `scaling_results.json`
+ `table_scaling_multiseed.txt` (retired, shown for reconciliation only).

## 9. Open / optional

- ✅ DONE `evaluation/make_results_tables.py` — regenerates all four tables (§2–§5) straight
  from the JSONs → `evaluation/phase4_tables.md`, the single reproducible source for the
  manuscript (no hand-typed numbers). Re-run it after any eval changes.
- ☐ Random / zero-shot references in §4 are from the single-seed first-base run under the
  identical terminal zone; they are seed-independent baselines, but for full consistency
  they could be recomputed under the offset-aware eval harness (low priority — the zone def
  is identical, so they carry over).
- ✅ DONE FFPE (real clinical deamination domain) — P31, `evaluate_ffpe_multiseed.py` →
  `results_ffpe_multiseed.json`, pooled Δ +0.0655 (p=3e-130). Now a row in Table 3 (§3). The
  only residual is that the FFPE profile is literature-shaped, not from a specific FFPE BAM via
  mapDamage2 — add the real-BAM version only if a reviewer asks.
- GENA-LM / NT absolute recovery is deliberately NOT tabulated — cross-backbone comparison
  is delta-based (§2); absolutes across different tokenizers are not comparable.
