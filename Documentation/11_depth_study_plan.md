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

## Phase 1 — D2, second + third backbone (architecture independence) ◐ (GENA-LM gate PASSED; full rollout running)
Retrain MLM vs DAM on models from other labs/tokenizers. DAM beating MLM at terminal on all three → not a DNABERT-2 artifact.
- ☑ `build_dataset.py` parametrized by tokenizer (env `VESTIGE_MODEL`/`VESTIGE_OUT`). GENA-LM dataset built (`training/dataset_genalm`, 275/69, identical windows).
- ☑ **P25 discovery + fix:** the original collator's "first base" test is a DNABERT-2 accident (73% C/G-first there vs 4% on GENA-LM) → built `masking/collator_dam_offset.py` (offset-aware, tokenizer-agnostic). **Pay-off test 1 PASSED**: fair ~0.145 budget + identical terminal gradient on both backbones.
- ☑ **Pay-off test 2 (gate for the full rebuild) PASSED (2026-11-05, P27):** 1-seed GENA-LM (MLM vs DAM-offset, loaded via the P26 `AutoModel` fix) + terminal eval. DAM beats MLM at the terminal: Δ **+0.101 / +0.080 / +0.039** at 3/5/10 nt, all p<1e-12, CIs exclude 0; edge largest at the most-damaged extreme; equal overall eval_loss (5.795 vs 5.786). Architecture independence demonstrated → gate cleared. `evaluate_genalm_terminal.py` → `results_genalm_terminal.json`. Single-seed + terminal-only: full rollout must add multi-seed and background NRR (caveats in P27).
- ◐ Full rollout (gate cleared): multi-seed GENA-LM (seeds 1,2) + background NRR; **re-verify/retrain DNABERT-2 DAM with the offset-aware collator** so all backbones share one principled method; then NT.
  - ☑ **GENA-LM background check DONE (seed 42, `evaluate_genalm_background.py`):** no collapse — Δ(DAM−MLM) = +0.009/+0.010/+0.011 on a 3/5/10 nt central band (vs pmd-mask's −0.134). DAM improves C/G recovery everywhere, most at the ends → "balanced reconstructor" confirmed on GENA-LM. Second caveat (terminal-only) removed.
  - ☑ **Multi-seed GENA-LM DONE (2026-11-06, seeds 42/1/2, `evaluate_genalm_multiseed.py`):** every seed same sign; pooled terminal Δ +0.110/+0.087/+0.048 (p 2e-17…1e-18), pooled background Δ +0.015/+0.015/+0.014 (p 6e-4…9e-9, positive not negative). **GENA-LM CLOSED — paper-ready 2nd backbone** (terminal win + no background cost + multi-seed + principled collator).
  - ◐ Re-verify/retrain DNABERT-2 DAM under the offset-aware collator (one method across backbones). **STAGED 2026-11-06:** configs `config_dam_offset_s{42,1,2}.yaml` (collator dam_offset, linear 0.15), eval `evaluation/evaluate_dnabert_offset_multiseed.py` (DNABERT-2 supports offset_mapping → direct port of the GENA-LM eval; smoke-tested on mlm-baseline = terminal 0.240 / bg 0.373), bundle `vestige_dnabert_offset.zip` (Drive-free, downloads models direct from Colab). Reuses existing `multiseed_models/mlm-baseline-s*` (MLM collator-invariant) → only the DAM-offset arm × 3 seeds trains. AWAITING user Colab run.
  - ☐ Nucleotide Transformer (3rd backbone) — needs NT dataset build (`VESTIGE_MODEL`=NT), NT configs, and an NT eval with manual 6-mer offsets (NT tokenizer likely has no offset_mapping).
- ☐ **Nucleotide Transformer** (InstaDeep, MLM, 6-mer): deterministic 6-mer offsets feed the same offset-aware collator; terminal eval needs manual offsets (no offset_mapping).

## Phase 2 — D3, second corruption domain (domain + principle independence) ☐
Same backbone (DNABERT-2), same eval — only the damage profile changes. Biggest payoff, cheapest.
- ☐ **Synthetic corruption-shape sweep** [ANGSHUMAN'S BUILD]: define 2–3 arbitrary corruption profiles (terminal / central / periodic); for each, corrupt data per that profile, train DAM-matched vs uniform MLM, show matched masking recovers it better. Proves the principle is corruption-agnostic. No new data.
- ☐ **FFPE** (real clinical deamination): source an FFPE damage profile (public FFPE BAM via mapDamage2, or literature rates); retrain DAM vs MLM; same C→T mechanism, different domain → "not just ancient DNA, also clinical sequencing."

## Phase 3 — Tradeoff characterization (own the pmd-mask result) ☐
- ☐ [ANGSHUMAN'S BUILD] Sweep the masking-concentration knob soft→hard (DAM → pmd-mask), plot the terminal↔background tradeoff curve. Reframes "pmd-mask beats DAM" into "DAM is the balanced operating point on a curve we characterize."

## Phase 4 — Rigor pass ☐
- ☐ Multi-seed every headline number; pooled stats throughout; one consistent model set across all tables; update the manuscript figures/claims last.

---

## Reserved for Angshuman (hands-on builds)
- Phase 2 synthetic-corruption generator (`evaluation/` + a profile factory).
- Phase 3 masking-concentration sweep + tradeoff-curve figure.

## Compute notes
- DNABERT-2 runs: Colab/Kaggle T4 (~16 min/run) as before.
- NT / larger backbones: RunPod A10G or similar for speed + no disconnects. Scope NT to a Colab-feasible size (50M/100M) only if paid compute is unavailable that day.

## Prior-work references to add (from the resubmission plan)
Dynamic/RoBERTa masking, SpanBERT, adaptive/PMI masking, pmd-mask (already benchmarked, P24), plus the second-backbone citations (Nucleotide Transformer, GENA-LM) and an FFPE-damage reference for Phase 2.
