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

## Phase 1 — D2, second + third backbone (architecture independence) ◐
Retrain MLM vs DAM on models from other labs/tokenizers. DAM beating MLM at terminal on all three → not a DNABERT-2 artifact.
- ☑ `build_dataset.py` parametrized by tokenizer (env `VESTIGE_MODEL`/`VESTIGE_OUT`). GENA-LM dataset built (`training/dataset_genalm`, 275/69, identical windows).
- ☑ **P25 discovery + fix:** the original collator's "first base" test is a DNABERT-2 accident (73% C/G-first there vs 4% on GENA-LM) → built `masking/collator_dam_offset.py` (offset-aware, tokenizer-agnostic). **Pay-off test 1 PASSED**: fair ~0.145 budget + identical terminal gradient on both backbones.
- ◐ **Pay-off test 2 (gate for the full rebuild):** short 1-seed GENA-LM train (MLM vs DAM-offset) + terminal eval. Needs: wire `dam_offset` into `train.py build_collator`; GENA-LM configs; GENA-LM terminal eval (re-tokenize eval windows + offset-aware recovery); run on GPU.
- ☐ Full rollout (only if test 2 shows a terminal edge): multi-seed GENA-LM + NT, and **re-verify/retrain DNABERT-2 DAM with the offset-aware collator** so all backbones share one principled method.
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
