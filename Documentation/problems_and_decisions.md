# Problems Faced & Key Decisions — VESTIGE

This document is a running log of every non-trivial problem encountered and every design decision made during the project. It is updated after each implementation step and serves as source material for the Methods, Limitations, and Discussion sections of the paper.

---

## P1 — Wrong Dataset Citation

**Phase:** 1
**Where it surfaced:** README species table + `01_data_acquisition.md` Section 1

**Problem:** Both documents cited van der Valk et al. 2021 (*Nature*), PRJEB44109 as the aDNA source — the dataset originally planned in the blueprint. The actual data downloaded and used was Palkopoulou et al. 2015 (*Current Biology*), ERP008929, accessions ERR855944 and ERR852028. A journal reviewer checking the accession against the citation would have found a mismatch.

**Fix:** Updated citation and accession in both files before the first commit.

**Why not just use van der Valk:** The Palkopoulou data was already fully downloaded, aligned, and profiled. Switching to van der Valk at that point would have discarded several hours of compute and changed the biological specimen (million-year-old vs. 4,300/44,800 BP). Palkopoulou is the better choice for this project anyway — two specimens of very different ages produce two damage profiles of distinct severity, directly enabling the cross-specimen consistency check that became a paper line.

---

## P2 — Wrong File Paths in Documentation

**Phase:** 1
**Where it surfaced:** `01_data_acquisition.md` Sections 1, 4, 5

**Problem:** Documentation referenced `mammoth_sra/mammoth/`, `mammoth_sra/results/`, `mammoth_sra/elephant_reference/` — the old working directory used before the project was reorganized into `Phoenix_Project/`. The actual paths are `Dataset/mammoth/`, `Dataset/results/`, `Dataset/elephant_reference/`.

**Fix:** All path references updated across all affected sections of the doc.

---

## P3 — NCBI Datasets CLI Non-Functional

**Phase:** 1
**Where it surfaced:** Attempting to download GFF3 annotation for EanMak 1.0

**Problem:** `/usr/local/bin/datasets` existed but was an XML stub (corrupted install), not the actual NCBI Datasets binary. The GFF3 file is ~3 GB and was needed to extract gene coordinates for TRPV3, KCNK9, HBB.

**Decision:** Replaced GFF3 parsing entirely with direct NCBI Entrez API queries via `Bio.Entrez` (`esearch` + `esummary` on the Gene database).

**Why not fix the CLI:** Re-installing datasets would require internet access, and the Entrez approach is strictly faster — no 3 GB download, no file parsing, returns coordinates in seconds.

**Why not use Biopython's GFF parser on the existing file:** GFF3 was not included in the initial reference download (only the FASTA was fetched). The catalog JSON confirmed this.

---

## P4 — HBB Not Annotated in EanMak 1.0

**Phase:** 1
**Where it surfaced:** `extract_genes.py` — Entrez search for `HBB[Gene Name] AND Elephas maximus[Organism]` returned zero hits

**Problem:** GCF_024166365.1 does not carry a gene symbol "HBB". The locus is annotated as `LOC126080006` ("hemoglobin subunit beta-1/2-like") on NC_064825.1, minus strand, 5,821 bp.

**Why this is expected:** Proboscideans carry a fused HBB/HBD (beta/delta globin) hybrid locus rather than separate HBB and HBD genes — the ancestral gene duplication was resolved differently in this lineage (Opazo et al. 2009; Signore et al. 2019). No HGNC-standard HBB symbol is therefore assigned in the Asian Elephant annotation.

**Decision:** Used `LOC126080006` as the HBB ortholog. The symbol mismatch and its biological reason are documented in `02_damage_profile_parsing.md` and will be noted in the paper's Methods section.

**Search path that found it:** Queried `hemoglobin subunit beta[Description] AND txid9783[Organism:exp]` — taxid 9783 is species-level *Elephas maximus*, broader than the strain-level ID used by the genome assembly.

---

## P5 — No Phoenix Conda Environment

**Phase:** 1
**Where it surfaced:** First attempt to run `parse_profiles.py`

**Problem:** Blueprint specified `conda create -n phoenix python=3.10`. The environment did not exist on the machine. Available environments: `base`, `flwr-env`, `venv`.

**Decision:** Used system Python 3 (numpy, pandas, matplotlib, biopython all present). No functional impact — the blueprint environment spec is for reproducibility documentation, not a hard runtime requirement at this stage.

---

## P6 — mapDamage2 Built-in R Plots Failed

**Phase:** 1
**Where it surfaced:** `Dataset/results/results_mammoth_ERR855944/Fragmisincorporation_plot.pdf` was generated but visually broken

**Problem:** mapDamage2's internal R plotting pipeline has incompatibilities with R ≥ 4.x. The PDF was produced but the damage curves were absent.

**Decision:** Wrote a custom `damage/visualize_damage.py` in matplotlib instead of debugging the R dependency.

**Why this is strictly better:** The custom script overlays both specimens on the same axes with the mean as a solid line — a figure that mapDamage2's own output cannot produce. Full control over styling, axis labels, and output format (PDF + PNG at 300 dpi).

---

## P7 — GFF3 Not Included in Reference Download

**Phase:** 1
**Where it surfaced:** Checking `Dataset/elephant_reference/ncbi_dataset/` after discovering the datasets CLI was broken

**Problem:** The reference genome download only fetched the FASTA assembly (`elephant_ref.fa`). The GFF3 annotation file, needed for gene coordinate extraction, was not downloaded.

**Resolution:** Moot once P3 was resolved — Entrez queries made the GFF3 unnecessary.

---

## P8 — Raw Damage Frequencies Too Sparse for MLM-Parity Masking

**Phase:** 2 — Collator design
**Where it surfaced:** Designing `DamageAwareDataCollator._prob_matrix()`

**Problem:** Empirical C→T deamination rates from the two specimens peak at 0.29–0.41% at position 1 from the 5′ end. Standard MLM masks 15% of tokens. Using raw damage frequencies as masking probabilities would give DAM ~40× fewer masked tokens per batch than the MLM baseline, making any loss or accuracy comparison between the two runs invalid — the models would be trained under fundamentally different data densities.

**Decision:** Added a `scale_to` parameter (default `0.15`) that uniformly rescales the entire probability matrix so E[masking rate] = 0.15. The *relative* damage gradient is fully preserved — terminal C tokens still receive ~28× higher masking probability than interior C tokens (verified by chi-squared test, χ²(1) = 7154.8, p ≈ 0). The absolute values are scaled up.

**Why not adjust the evaluation instead:** Evaluating DAM and MLM at different masking densities would require separate perplexity normalizations and complicate the ablation. Reviewers at *Bioinformatics* would immediately question unmatched training conditions. Scaling is the standard approach in masked LM literature when using non-uniform masking strategies.

**`scale_to=None` option retained:** Available for analysis runs where biologically literal frequencies are desired (e.g., inspecting the raw gradient shape).

---

## P9 — Soft-Masking Doubles DNABERT-2 Token Count

**Phase:** 2 — Dataset construction
**Where it surfaced:** Verifying token counts per window during `build_dataset.py` development

**Problem:** `elephant_ref.fa` uses lowercase nucleotides to denote RepeatMasker soft-masked repeat regions (standard UCSC/NCBI convention). DNABERT-2's BPE tokenizer was trained on uppercase-only sequences and has no lowercase vocabulary entries. When lowercase input is passed, the tokenizer falls back to character-level tokenization, roughly doubling the token count per window:

| Gene | 2,000 bp (mixed case) | 2,000 bp (uppercase) |
|------|----------------------|----------------------|
| TRPV3 | 458 tokens (3.4% lowercase) | 399 tokens |
| KCNK9 | 910 tokens (31.2% lowercase) | 400 tokens |
| HBB | 817 tokens (25.0% lowercase) | 402 tokens |

KCNK9 and HBB would have produced windows far exceeding the 512-token limit with mixed-case input, causing silent truncation or filtering of most windows.

**Decision:** Uppercase all extracted sequences before tokenization.

**Information lost:** Repeat annotation (soft-mask positions). This is acceptable — the training objective is sequence reconstruction, not repeat characterization. The model is not asked to distinguish repeat from non-repeat regions.

**Why not hard-mask (replace lowercase with N):** N tokens would introduce a third vocabulary class alongside real bases, distorting the token distribution and potentially confusing the pre-trained DNABERT-2 weights. Uppercasing is the minimal intervention.

---

## P10 — ERR852028 Added Mid-Phase

**Phase:** 1→2 boundary
**Where it surfaced:** Observing that ERR852028.sra (15 GB) was already downloaded but not processed

**Problem/Opportunity:** Blueprint planned ERR855944 only for the initial damage profiling. ERR852028 was sitting idle.

**Decision:** Align and run mapDamage2 on ERR852028 immediately. Results:

| Accession | Mapping rate | C→T pos 1 | G→A pos 1 | Background |
|-----------|-------------|-----------|-----------|------------|
| ERR855944 | 99.12% | 0.0029 | 0.0031 | 0.0010 |
| ERR852028 | 99.33% | 0.0041 | 0.0042 | 0.0010 |

ERR852028 shows slightly higher terminal damage (consistent with its greater age, ~44,800 BP vs ~4,300 BP). Both converge to identical background by position ~10. `damage_profile.npy` now stores individual profiles for both specimens plus their mean. The DAM collator uses the mean arrays.

**Paper benefit:** Adds cross-specimen validation and a direct line: *"Damage patterns were consistent across both specimens, converging to a shared background frequency of 0.001 by position 10."*

---

## P11 — Collator Unit Test: Chi-Squared Over Ratio Assertion

**Phase:** 2 — Collator validation
**Where it surfaced:** Initial unit test used a simple ratio check (`terminal_rate > 1.5 × central_rate`)

**Problem:** A ratio threshold is not a statistical test — it cannot quantify how unlikely the observed distribution is under the null hypothesis of uniform masking, and it gives no threshold that would catch a subtle regression in a future refactor.

**Decision:** Replaced with a chi-squared test of independence on a 2×2 contingency table:

|  | Masked | Not masked |
|--|--------|------------|
| Terminal (pos 1–5) | observed | observed |
| Central (pos 21–50) | observed | observed |

10,000 trials, `scale_to=None` (raw damage frequencies, so the gradient is maximally visible). Result: χ²(1) = 7154.8, p ≈ 0. Assertion: `p < 0.001`.

**Why chi-squared specifically:** The null hypothesis is that masking probability is independent of position group (terminal vs. central). Chi-squared of independence is the canonical test for that claim on count data. It also naturally handles the asymmetry in group sizes (5 positions vs. 30 positions) via the expected frequency computation.

**Why not a t-test:** A t-test on per-position masking rates would also work, but chi-squared on raw counts is more powerful here — we have large counts (N_TRIALS × n_positions) and the test directly addresses the distributional question rather than comparing means.

**Paper relevance:** The test result can be cited in the Methods section as validation that the collator implements the intended masking gradient — *"correct operation was verified by a chi-squared test of independence on masking position counts (χ²(1) = 7154.8, p < 10⁻³)"*.

---

## P12 — Three Dependency Failures on First Training Run

**Phase:** 2 — Training
**Where it surfaced:** First execution of `training/train.py`

Three separate import/runtime errors hit in sequence:

**P12a — `tf_keras` missing:**
`transformers` imports TF integration at load time. Keras 3 (installed by tensorflow) is not backward-compatible with this integration layer. Fix: `pip install tf-keras==2.21.0`. Added to `requirements.txt` with an explanatory comment.

**P12b — `accelerate` missing:**
`Trainer` requires `accelerate>=0.26.0` for the PyTorch backend. It is not pulled in automatically by `pip install transformers`. Fix: `pip install "accelerate>=0.26.0"`. Added to `requirements.txt`.

**P12c — `use_mps_device` deprecated:**
`TrainingArguments(use_mps_device=True)` was removed in a recent `transformers` version — the Trainer now detects MPS automatically via `accelerate`. Passing the argument raises a TypeError. Fix: removed the argument from `train.py`. MPS is used automatically on Apple Silicon when `accelerate` is installed.

**P12e — Training wall-clock time severely underestimated:**
Benchmark predicted ~6 hrs for 20 epochs at batch size 8 (29s/step × 700 steps). Actual runtime: **11 hours 24 minutes** per run. The discrepancy is from the benchmark using a vanilla BERT config without DNABERT-2's custom MosaicBERT attention layers, which add overhead even in PyTorch fallback mode, plus MPS memory management overhead not visible in short benchmarks. Total wall time for both runs: ~23 hours. Plan accordingly — start each run before sleep.

**P12d — `triton` unavailable on macOS:**
DNABERT-2's `bert_layers.py` imports `triton` at module load time. `triton` is a CUDA-only library with no macOS build. Fix: stub the module before any transformers import using a `ModuleSpec`-backed `types.ModuleType`. The model falls back to standard PyTorch attention with a warning — no accuracy impact.

**P12f — `offset_mapping` unexpected keyword in `torch_mask_tokens`:**
Newer versions of `transformers` pass `offset_mapping=None` as a keyword argument to `torch_mask_tokens()`. Our override signature didn't accept it, causing a `TypeError` on the first DAM training step. Fix: added `**kwargs` to the override signature. The argument is unused — it carries subword-to-character offset data irrelevant to our masking logic.

**Reproducibility note:** All fixes are encoded in `requirements.txt` and `train.py`. A fresh install from `requirements.txt` + the mapDamage2 conda install will not hit these errors again.

---

## P13 — BPE Tokenization Cascade: Token-Diff Masking Masks ~30 Tokens per 2 Damaged Nucleotides

**Phase:** 3 — Evaluation
**Where it surfaced:** Debugging the second version of `evaluate_reconstruction.py` (after fixing P13's token-substitution bug). Spotted by auditing: 2 damaged nucleotides produced 31 changed token positions between `gt_ids` and `dam_ids`.

**Problem:** The second implementation of `evaluate_reconstruction.py` found changed token positions by diffing `gt_ids` vs. `dam_ids` (the re-tokenized damaged sequence). BPE tokenization is context-sensitive: changing one nucleotide causes the tokenizer to choose different segment boundaries for a large surrounding region. 2 substituted nucleotides → 31 different token positions. Masking all 31 positions asks the model to reconstruct ~150 nucleotides from context when only 2 are actually damaged. The evaluation was measuring something entirely different from claimed.

**Root cause:** BPE tokenization is not positionally local. Changing character `i` can alter token boundaries from position `i` all the way to the next segment anchor, potentially many tokens ahead.

**Fix:** Use `offset_mapping` from the tokenizer to directly map each damaged nucleotide position (0-based in the raw DNA string) to the single BPE token in the ORIGINAL tokenization that spans that character. Mask only those 1–2 token positions in `gt_ids`. Never re-tokenize the damaged sequence.

**Correct pipeline:**
1. `simulate_damage.py`: decode `gt_ids` → DNA, apply damage, use `offset_mapping` on original tokenization to map `nuc_pos → tok_pos`. Store `nuc_damaged` (list of nucleotide positions) and `tok_damaged` (deduplicated token positions).
2. `evaluate_reconstruction.py`: mask only `tok_damaged` in `gt_ids`, run model, decode predicted token at each position, check the character at the sub-token offset.

Result after fix: 1.14 damaged tokens per window (matching 1.14 damaged nucleotides — each falls in exactly one token). BLEU-4 rose from 88% to 99.86%, consistent with only ~1.14 characters changing per 2000-character sequence.

**Paper relevance:** Demonstrates that BPE cascade is a non-obvious failure mode for any evaluation involving sequence-level substitution + token-level comparison. The correct method requires offset_mapping to preserve the original tokenization and mask only the affected tokens.

---

## P14 — MLM Significantly Outperforms DAM on Nucleotide Recovery (p=0.0009)

**Phase:** 3 — Evaluation
**Where it surfaced:** `evaluate_reconstruction.py` corrected results.

**Result:** MLM baseline recovery rate 65.7% ± 39.5% vs. DAM 43.4% ± 38.1%. Paired t-test: t=−3.556, p=0.0009. Bootstrap 95% CI: [−0.349, −0.107] — entirely negative, strongly significant. MLM is the better model for per-nucleotide reconstruction at authentic damage sites.

**This co-exists with DAM's 54% lower validation MLM loss (1.73 vs. 3.76).** The two findings together constitute the paper's most interesting result.

**Mechanistic explanation:**
- *Loss vs. argmax:* CE loss rewards calibrated probability distributions. DAM's focused masking produces a sharper, lower-CE distribution at terminal C/G positions but that does not mean its argmax prediction is more often correct.
- *Distributional shift:* Training used 15% masking (scaled). Evaluation has ~0.08% damage per nucleotide (authentic rates). At this sparse-masking regime, MLM's broad training generalizes better.
- *Richer representations:* MLM's uniform masking builds position-agnostic context representations across the full sequence. DAM's focused signal concentrates on terminal tokens, potentially at the cost of general-purpose token reconstruction ability.

**Decision:** Present both findings honestly in the paper. Primary claim: DAM is a more efficient training objective for learning the damage grammar (54% lower loss). Secondary finding: under authentic damage rates at evaluation, MLM reconstructs individual damaged nucleotides significantly more accurately. Proposed future direction: masking curricula combining damage-aware concentration with uniform exploration.
---

## P15 — DAM Collator: baseline_prob Dominates scale_to, Crushing C/G Masking to ~1%

**Phase:** 3 — Evaluation (discovered during audit of P14 result)
**Where it surfaced:** Auditing per-base masking probabilities output by `_prob_matrix()`.

**Problem:** `scale_to=0.15` scales the mean masking probability over **all tokens** in the sequence. A/T tokens outnumber C/G roughly 2:1 in genomic sequences. With `baseline_prob=0.03`, A/T dominate the mean. Raw C→T damage rates (~0.003 at position 1) are near-zero by comparison. The scaler crushes C/G masking to ~1% while inflating A/T to ~33%:

| Base | Actual masking (original DAM run) |
|------|----------------------------------|
| A/T  | **32.95%** — massively over-masked |
| C/G  | **~1.1%** — nearly never masked |
| Terminal C pos 1 | **2.1%** |

MLM masked C/G at 15%. The original DAM model had 13× less C/G training signal than MLM — the opposite of the intended design. This is why P14 shows DAM worse at C/G reconstruction.

**Fix:**
1. `baseline_prob=0.0` — A/T never masked. Biologically correct (no deamination at A/T). Evaluation never tests A/T so no impact on results.
2. Scale over C/G positions only: replace `prob.mean()` with `prob[is_C | is_G].mean()` in `_prob_matrix()`. C/G now average exactly 15%, with the damage gradient on top.

**Result after fix:**

| Base | Fixed DAM | MLM |
|------|----------|-----|
| A/T  | 0% | 15% |
| C/G  | ~15% avg | 15% flat |
| Terminal C pos 1 | **28%** | 15% |
| Interior C (pos 130) | **14.8%** | 15% |

C/G get the same total masking density as MLM. The damage gradient is the only difference — which is the contribution.

**Files changed:** `masking/collator_dam.py` (`_prob_matrix` scaling), `training/train.py` (`baseline_prob=0.0`).
**Retraining required:** DAM run only (~11h 24min). Old checkpoints archived to `dam-proposed-baseline_prob_0.03_ARCHIVED/`.

---

## P16 — Corrected DAM Run Results (baseline_prob=0, C/G-only scaling)

**Phase:** 3 — Evaluation
**Training runtime:** 14h 32min (700 steps, 20 epochs, seed=42)

**Training loss comparison:**

| Model | Best eval loss | Best checkpoint |
|-------|---------------|----------------|
| MLM baseline | 3.7568 | checkpoint-595 (epoch 17) |
| DAM corrected | **3.2736** | checkpoint-595 (epoch 17) |
| DAM broken (archived) | 1.7326* | — |

*The archived DAM's 1.7326 was not a real result — it trained mostly on A/T tokens (32% masking each), making the task trivially easy. The corrected 3.2736 is the honest 13% improvement over MLM on C/G-only masked loss.

**Table 1 — Background damage site reconstruction (n=43 paired windows):**

| Metric | MLM | DAM |
|--------|-----|-----|
| Nucleotide recovery (mean) | 65.7% | 65.1% |
| p-value | 0.8917 | — |

No significant difference at background positions — expected, since both models now receive equal C/G masking density there.

**Table 2 — Terminal-position reconstruction (first/last 10 nt, n=69 windows):**

| Metric | MLM | DAM |
|--------|-----|-----|
| Nucleotide recovery (mean) | 34.1% | **38.6%** |
| Aggregate (correct/total) | 33.5% | **38.2%** |
| p-value | **0.041** | — |
| Bootstrap 95% CI (DAM−MLM) | [+0.004, +0.087] | — |

DAM significantly outperforms MLM at terminal positions (p = 0.041, CI entirely positive). This is the biologically correct evaluation domain — the positions where DAM concentrated its masking probability during training.

**Note:** An earlier version of this entry showed p = 0.086 / 39.9% vs 44.3% — those were from a pre-retraining run where `evaluate_terminal.py` was still using `checkpoint-595` from the broken DAM (archived). The numbers above are from the final corrected DAM checkpoint and match `evaluate_scaling.py` at T_END=10 exactly (cross-validation passed).

---

## P17 — fetch_cds.py: mRNA[Feature Key] Returned Genomic NC_* Records

**Phase:** 4 — CDS fetching
**Where it surfaced:** `protein/fetch_cds.py` first run — NCBI returned NC_* (genomic chromosome) records instead of NM_*/XM_* mRNA records.

**Problem:** The initial query used `esearch` on the nucleotide database with `mRNA[Feature Key]` as a filter. NCBI's nucleotide database includes genomic records that contain mRNA features as sub-annotations. The search returned chromosome-level accessions (NC_*) that carry the gene as one of many features — not the spliced mRNA transcript itself.

**Fix:** Switched to a two-step approach:
1. `esearch` on the Gene database for the gene name + organism
2. `elink` with `gene_nuccore_refseqrna` to get associated RefSeq mRNA records
3. `efetch` the individual mRNA accessions; skip any NC_* or NW_* records

This reliably returns NM_*/XM_* mRNA records, from which the CDS feature is extracted.

**Results:** TRPV3 → XM_049859879.1 (2376 nt, 791 aa), KCNK9 → XM_049854616.1 (1044 nt, 347 aa), HBB → XM_049891336.1 (489 nt, 162 aa).

---

## P18 — HBB Produces 0 Damaged Positions at 10× PMD Scale

**Phase:** 4 — damage_reconstruct_cds.py
**Where it surfaced:** First run of `damage_reconstruct_cds.py` — HBB CDS produced 0 damaged positions.

**Problem:** The HBB CDS is only 489 nt long. At 10× authentic PMD rates, peak C→T probability at position 1 is ~3.5%. The expected number of damaged positions at the 5′ end is proportional to CDS length × base frequency × peak probability. For HBB (489 nt, ~25% C at 5′ end), this works out to ~0.4 expected damaged positions — near zero on any single run (stochastic draw from Bernoulli).

**Fix:** Per-gene `DAMAGE_SCALE_PER_GENE` dict: `{"TRPV3": 10, "KCNK9": 10, "HBB": 30}`. At 30×, HBB peak C→T becomes ~10.5%, giving ~1.3 expected damaged positions — reliably ≥1 across runs.

**Paper note:** The Methods section notes that damage was amplified 10–30× authentic rates to ensure ≥1 damaged position per gene for meaningful comparison. This is standard for controlled in silico experiments (amplified rates are disclosed, not used to inflate claims).

---

## P19 — ESMFold Local Download: 2h+ / 8.44 GB / 403 Errors

**Phase:** 4 — run_esmfold.py
**Where it surfaced:** First version of `run_esmfold.py` used `EsmForProteinFolding.from_pretrained("facebook/esmfold_v1")`.

**Problem:** The model is ~2.7 GB of ESM-2 weights + ~1.5 GB of folding trunk = ~4.2 GB total, but HuggingFace also downloads tokenizer and auxiliary files. Download started at ~61% after 2 hours then hit a 403 (rate limit / quota). Model was killed.

**Decision:** Rewrote to use the ESMFold public REST API (`https://api.esmatlas.com/foldSequence/v1/pdb/`). No model download needed. Latency is ~60–120 seconds per sequence. Total time for 9 proteins: ~15 min (vs. 2h+ download + unbounded compute).

**Tradeoff:** REST API has a ~600 aa limit per request. TRPV3 (791 aa) required truncation (see P20). The API is also rate-limited and occasionally returns 503; handled with 3-attempt retry + 5-second backoff.

---

## P20 — TRPV3 HTTP 413 from ESMFold API (791 aa Exceeds Limit)

**Phase:** 4 — run_esmfold.py
**Where it surfaced:** ESMFold API returned HTTP 413 (payload too large) for TRPV3 full-length sequence.

**Problem:** ESMFold REST API rejects sequences > ~600 aa. TRPV3 is 791 aa.

**Decision:** Truncate TRPV3 to the N-terminal ankyrin repeat domain (aa 1–400). Biologically justified: the 5′ PMD damage maps to aa 1–8 (the first 24 nt of the CDS). The 3′ damage maps to aa 784–791, which are outside the truncated domain — those residues are excluded from the structural comparison and noted in the paper.

**Implementation:** `TRUNCATE = {"TRPV3": 400}` dict in `run_esmfold.py`. The table footnote clarifies: *"† TRPV3 folded on N-terminal ankyrin repeat domain (aa 1–400); full-length 791 aa exceeds ESMFold API limit."*

---

## P21 — ESMFold B-factor Column in 0–1 Scale (Not 0–100)

**Phase:** 4 — run_esmfold.py pLDDT extraction
**Where it surfaced:** First pLDDT extraction reported values of 0.67–0.80 instead of the expected 67–80 range.

**Problem:** Standard PDB convention stores pLDDT in the B-factor column in the 0–100 scale. ESMFold REST API returns B-factors in the 0–1 scale (a known quirk of this endpoint). Direct use of the raw value would produce non-standard pLDDT values that misrepresent confidence.

**Fix:** Multiply by 100: `bfactor = float(line[60:66].strip()) * 100.0`. Also switched to extracting only Cα atoms (one per residue, stable across all structures) rather than first-atom-per-residue fallback.

**Final values:** HBB 79.70 (ref), KCNK9 67.94 (ref), TRPV3 77.37 (ref, N-term domain only). All above the 70 pLDDT "confident fold" threshold except KCNK9 (67.94, borderline — expected for a multi-pass TM protein predicted from sequence).

---

## P22 — evaluate_scaling.py v1: Stochastic PMD Scaling Produced 0–12 Sites

**Phase:** 3 ext — Figure 3
**Where it surfaced:** First version of `evaluate_scaling.py` swept `DAMAGE_SCALE ∈ {1, 2, 5, 10, 20}` — applying stochastic damage at amplified rates and evaluating recovery at damaged positions.

**Problem:** The PMD profile only defines non-trivial probabilities for positions 1–25 from each read end. Within a 2000 bp window, that's at most 50 terminal nucleotides. Of those, ~25% are C or G (relevant for deamination), giving ~12 candidate positions per window. At authentic 1× rates, peak C→T is ~0.35% → expected damaged positions across 69 windows ≈ 0.3 — zero in practice. At 10× → ~3.5%, still only ~12 × 0.035 × 69 ≈ 29 sites total. Not enough for a powered comparison.

**Fix:** Switched evaluation strategy entirely. Instead of applying stochastic damage and evaluating at damaged positions, mask ALL terminal C/G positions deterministically and sweep the terminal zone width `T_END ∈ {3, 5, 10, 15, 20, 25}`. This gives 99–790 sites per T_END and well-powered statistics at every value.

**Why this is more principled:** DAM was trained to mask terminal C/G positions (not to damage-simulate and reconstruct). Evaluating the model on exactly its training objective (masked terminal C/G reconstruction) is cleaner than the stochastic simulation, which would add PMD-simulation variance on top of the masking experiment.

---

## P23 — MC1R Not Annotated in Asian Elephant Genome

**Phase:** 3 ext — validation expansion
**Where it surfaced:** `expand_validation.py` — NCBI Gene search for MC1R in Elephas maximus returned no results.

**Problem:** MC1R (melanocortin-1 receptor) is not annotated under that gene symbol in GCF_024166365.1 (EanMak 1.0). The Asian Elephant genome annotation uses gene symbols inconsistently for some coat-colour loci, similar to HBB (P4).

**Decision:** Skipped MC1R. The remaining four genes (TRPA1, UCP1, ADRB3, FASN) contributed 557 windows — sufficient for the validation expansion goal. A fallback description search could recover the locus but was not necessary given the total window count.

**Impact:** None on results. 557 extra windows raised total to 626, giving p < 0.001 at all six T_END values.

---

## P24 — Multi-Seed Benchmark Overturns the Single-Seed "dam_power Wins" Story

**Phase:** Resubmission (TCBB editorial reject — "insufficient depth", "needs more experimental comparison with existing methods")

**Where it surfaced:** I retrained 5 masking arms (mlm, dam linear, dam_smooth, dam_power, pmdmask) on 3 seeds (42, 1, 2), all on Colab T4, then evaluated terminal C/G recovery and authentic-damage background recovery per seed. Code `evaluation/evaluate_multiseed.py`, weights `training/multiseed_models/<arm>-s<seed>/`, results `evaluation/results_multiseed.json` + `table_multiseed.txt`.

**Problem:** The single-seed (seed 42) run told two stories I was about to build the resubmission on, and multi-seed broke one:
- Seed 42 said linear DAM ≈ MLM on terminal (+0.016, p=0.438) but dam_power beat MLM (+0.061, p=0.012), implying the squared-frequency mapping was what "rescued" DAM.
- Across 3 seeds this is false. Linear DAM, smoothed, and power give the SAME terminal gain within noise (mean over MLM: +0.040 / +0.043 / +0.047; MLM terminal 0.343; all three variants positive on all three seeds). The p=0.438-vs-0.012 gap existed only on seed 42. The "power map is special" claim was a single-seed fluctuation and is dropped.

**What survived (confirmed on all 3 seeds):**
- DAM beats MLM on terminal C/G recovery (every variant positive on every seed, ~+0.04). The paper's core claim holds once it does not rest on the one unlucky seed where linear DAM looked dead (that seed drove the TCBB-facing "weak result" worry).
- DAM pays no background cost (dam variants vs MLM on background all ns: +0.008 / +0.022 / −0.027).
- pmd-mask (hard threshold) collapses on background (−0.134 mean, negative all 3 seeds, p=0.004 on seed 42), confirming soft > hard damage-aware masking.

**The awkward fact:** pmd-mask beats DAM on terminal by a wide margin (+0.164 vs +0.04, all p < 0.001). I cannot claim DAM is the best terminal reconstructor.

**Decision:**
- Report LINEAR DAM as the method (simplest, and it works). Present smoothed/power as an ablation showing the result is insensitive to the mapping shape — insensitivity is a robustness point, not a weakness. Remove every "power / freq² is special" claim.
- Frame the contribution as a tradeoff, not a clean sweep: hard damage-thresholding (pmd-mask) over-specializes to the terminal zone (large terminal gain, background collapse); soft damage-aware masking (DAM) improves terminal over uniform MLM while preserving general infilling. DAM is the balanced reconstructor.

**Pending:** Per-seed terminal significance is consistent in sign but marginal individually (DAM significant on 1–2 of 3 seeds, each seed only 69 windows). Next is a pooled across-seed test (seed-averaged per-window one-sample test, windows as the replication unit, plus a MixedLM confirmatory) to give one defensible p-value for the ~+0.04 terminal gain. See P25.

---

## P25 — The DAM Collator's "First Base" Test Is a DNABERT-2 Tokenization Accident (D2 blocker + fix)

**Phase:** Depth study D2 (second backbone). See `Documentation/11_depth_study_plan.md`.

**Where it surfaced:** Porting DAM to GENA-LM (the first second-backbone). The dataset built cleanly (275/69, identical windows, GENA-LM tokenized), but the DAM collator masked only ~0.6% of tokens vs MLM's 15% — an unfair budget that would rig the comparison against DAM.

**Root cause:** `collator_dam.py` flags a token as damage-relevant by its FIRST character (`_build_base_lookup`). Diagnostic on the two tokenizers:
- DNABERT-2: 73% of tokens are C/G-first (A-first only 1% — its BPE merges almost never start a token with A), so first-base tracks C/G content well.
- GENA-LM: 4% of tokens are C/G-first (A-first 49%, T-first 47%), so first-base flags almost nothing.
- BUT 96% of tokens in BOTH tokenizers CONTAIN a C or G. So "first base" is a DNABERT-2 accident; "contains C/G at a known position" is tokenizer-robust. A reviewer could already attack the original paper as "DAM only works because of a DNABERT-2 BPE quirk."

**Fix (chosen — full offset-aware rebuild):** new `masking/collator_dam_offset.py` (`OffsetAwareDamageCollator`). Reconstructs each token's actual nucleotides and their ABSOLUTE window position from the token string (DNA BPE / k-mer tokens ARE their nucleotide content), then scores a token by the strongest damage among the C's it covers (by 5' distance, ct_5p) and G's it covers (by 3' distance, ga_3p), rescaled so damage-relevant tokens average 0.15. Tokenizer-agnostic (DNABERT-2 / GENA-LM / NT). This also upgrades the method from "accidental" to "nucleotide-level, principled," which directly rebuts the reviewers.

**Pay-off test 1 (collator level, PASSED):** masking rate by nucleotide position, both backbones — fair overall budget (DNABERT-2 0.144, GENA-LM 0.146) AND a terminal gradient (5' nt[0:5] ≈ 0.21–0.25 decaying to ~0.14 interior) that is nearly identical across the two tokenizers. GENA-LM's 0.6% failure is fixed; the DAM signature ports.

**Decision / next:** collator mechanism ports (necessary condition met). Before the full investment (multi-seed both backbones + re-verifying/retraining DNABERT-2 DAM with the offset-aware collator so all backbones use the same method), run Pay-off test 2: a short single-seed GENA-LM train (MLM vs DAM-offset) + terminal eval. Proceed to full rollout only if DAM shows a terminal edge there. The original first-base `collator_dam.py` is kept (reproduces the published DNABERT-2 numbers); the offset-aware one becomes the method going forward.

---

## P26 — GENA-LM Loaded With the WRONG Architecture (Test-2 first result was INVALID)

**Phase:** Depth study D2, Pay-off test 2 (GENA-LM).

**Where it surfaced:** First GENA-LM Test-2 eval returned DAM −0.01 vs MLM (a weak negative), with abnormally low recovery (~0.19 vs DNABERT-2's ~0.34). The HuggingFace load printed: checkpoint weights `pre_attention_ln` / `post_attention_ln` "were not used", and `attention.output.LayerNorm` / `output.LayerNorm` "newly initialized".

**Root cause:** GENA-LM (`AIRI-Institute/gena-lm-bert-base-t2t`) is a **pre-LayerNorm** BERT, and its `auto_map` maps only `AutoModel -> modeling_bert.BertForMaskedLM` — there is NO `AutoModelForMaskedLM` entry. So `AutoModelForMaskedLM.from_pretrained(..., trust_remote_code=True)` silently fell back to transformers' standard **post-LN** `BertForMaskedLM`, dropping GENA-LM's real LayerNorm weights and randomly initialising the wrong ones. This happened at BOTH training (Colab) and eval → the models were fine-tuned from a mangled architecture. The −0.01 is a loading bug, NOT "DAM fails on GENA-LM." DO NOT report that number.

**Fix (verified):** load GENA-LM via `AutoModel.from_pretrained(..., trust_remote_code=True)` (its custom class IS a masked LM with a 32000-vocab head). Clean load (no mismatched/newly-init weights), forward loss 6.3 vs the broken 32.4, trainable end-to-end. Implemented as a config-selectable loader in `train.py` (`model_class: AutoModel`, default `AutoModelForMaskedLM`); added to both GENA-LM configs; `evaluate_genalm_terminal.py` uses `AutoModel`. The invalid Colab models (mlm-genalm-s42, dam-offset-genalm-s42 in Drive + locally staged) are discarded.

**Lesson:** when porting to a new backbone, CHECK the load log for "newly initialized" / "not used" weights and sanity-check the initial loss (~ln(vocab)); a wrong model can still train and silently produce a plausible-looking (wrong) number. **Next:** retrain GENA-LM MLM vs DAM-offset with the corrected loader, re-run the terminal eval, THEN decide the full rollout.

---

## P27 — Pay-off Test 2 PASSED: Offset-Aware DAM Beats MLM on GENA-LM (full rebuild justified)

**Phase:** Depth study D2, Pay-off test 2 (GENA-LM), with the P26 loader fix.

**Setup:** GENA-LM (`gena-lm-bert-base-t2t`, pre-LN, loaded via `AutoModel`) fine-tuned twice from identical hyperparameters, seed 42, 20 epochs — the ONLY difference is the collator (uniform MLM vs `OffsetAwareDamageCollator`). Final eval_loss near-identical (MLM 5.795, DAM 5.786), i.e. equal overall masked-LM fit. Terminal recovery eval re-tokenizes each of the 626 eval windows with GENA-LM's BPE, masks the token covering each terminal C (5' side) / G (3' side), predicts, scores the base at the right sub-offset. `evaluation/evaluate_genalm_terminal.py` → `evaluation/results_genalm_terminal.json`.

**Result (DAM-offset − MLM, paired t-test + 10k-bootstrap CI):**
- terminal 3 nt: MLM 0.163 → DAM 0.264, Δ **+0.101**, p=1.6e-13, CI [+0.075, +0.127], n=486
- terminal 5 nt: MLM 0.172 → DAM 0.252, Δ **+0.080**, p=6.1e-17, CI [+0.062, +0.098], n=565
- terminal 10 nt: MLM 0.202 → DAM 0.241, Δ **+0.039**, p=8.2e-13, CI [+0.029, +0.050], n=619

**Reading:** on a different lab's model with a completely different (BPE) tokenizer, the offset-aware DAM improves terminal reconstruction over uniform MLM, and the edge is LARGEST at the most-damaged extreme (terminal 3 nt, +0.10) and decays inward (10 nt, +0.04) — the damage-awareness signature, not a flat offset. Equal overall eval_loss means the terminal gain is not bought by sacrificing general infilling. This is the architecture-independence evidence the TCBB reject demanded: DAM is not a DNABERT-2/BPE-quirk artifact (recall P25 — the OLD first-base collator masked only 0.6% on GENA-LM; the offset-aware rebuild both fixes that AND makes the method nucleotide-principled). The $0.30/hr full rebuild is justified.

**Caveats to carry into the full rollout (do not overclaim yet):**
- Single seed (42). The dam_power lesson (P24) says single-seed can mislead — the full GENA-LM arm must be multi-seed (42/1/2) before this goes in the paper.
- Terminal-only metric. Must still measure GENA-LM BACKGROUND / non-terminal NRR to confirm DAM doesn't collapse there (the pmd-mask failure mode, P24).
- GENA-LM numbers are backbone-specific, not comparable to DNABERT-2 absolute values.

**Decision / next (full rollout, gate cleared):** (1) multi-seed GENA-LM MLM vs DAM-offset (seeds 1, 2) + background NRR; (2) re-verify/retrain DNABERT-2 DAM with the offset-aware collator so every backbone shares one principled method; (3) Nucleotide Transformer as the third backbone; (4) D3 domains. See `Documentation/11_depth_study_plan.md`.

**Background check (same seed-42 models, `evaluation/evaluate_genalm_background.py`, central band = the interior where PMD damage is at baseline):** Δ(DAM−MLM) = +0.009 / +0.010 / +0.011 at a 3/5/10 nt central band (p = 0.13 / 0.024 / 0.0006). DAM does NOT sacrifice interior recovery — it is flat-to-slightly-positive on background. Contrast pmd-mask's −0.134 background collapse (P24). So on GENA-LM the full picture is: DAM improves C/G reconstruction EVERYWHERE, largest at the damaged terminal (+0.10) and small-but-non-negative in the interior (+0.01) — cleanly the "balanced reconstructor" claim, with zero background cost. (Mechanism: DAM concentrates its 15% budget on C/G-containing tokens globally, elevated further at the ends; MLM spreads budget over A/T tokens too. The eval scores C/G recovery only — the damage-repair-relevant metric; A/T recovery under DAM is not measured and is expected to be ~MLM-or-below, which is immaterial to deamination repair.) One caveat removed: the pmd-mask collapse mode does not occur. Remaining caveat: multi-seed (42/1/2) still owed before the paper.

**MULTI-SEED CONFIRMED (2026-11-06, seeds 42/1/2, `evaluation/evaluate_genalm_multiseed.py` → `results_genalm_multiseed.json`).** Pooled = seed-averaged per-window Δ, one-sample t + Wilcoxon + 10k-bootstrap CI (windows = replication unit, same pooling as the DNABERT-2 multi-seed benchmark). All six models verified `pre_layer_norm=True`, eval_loss 5.79–5.85, MLM≈DAM per seed.
- **TERMINAL** (DAM−MLM), pooled Δ / p_t / CI: 3 nt **+0.110**, 2.2e-17, [+0.086,+0.135]; 5 nt **+0.087**, 3.9e-21, [+0.070,+0.104]; 10 nt **+0.048**, 1.3e-18, [+0.038,+0.059]. Per-seed signs all positive and tight (e.g. 3 nt: +0.101/+0.110/+0.120).
- **BACKGROUND** (central band), pooled Δ / p_t: 3 nt **+0.015** (5.8e-4); 5 nt **+0.015** (7.3e-5); 10 nt **+0.014** (8.7e-9). All positive, all seeds same sign. With 3 seeds the interior effect is now significant everywhere (single-seed s42 had the 3 nt band at p=0.13) — DAM slightly IMPROVES background C/G recovery, ~7× smaller than the terminal gain. The damage-gradient signature (large at the ends, small in the interior) holds. No collapse.

**GENA-LM is CLOSED as a paper-ready second-backbone result:** terminal win + no background cost + multi-seed, on a different lab's model with a different tokenizer, using the nucleotide-principled offset-aware collator. This is the architecture-independence evidence the TCBB reject demanded. Remaining D2 work: re-verify/retrain DNABERT-2 DAM under the SAME offset-aware collator (so both backbones share one method), then Nucleotide Transformer as the third backbone.

## P28 — DNABERT-2 Re-Verified Under the Offset-Aware Collator: Original Result Was Not a First-Base Artifact (D2 closes the home backbone)

**Phase:** Depth study D2, "one method across backbones". Closes the item left open in P27: the paper's DNABERT-2 DAM used the OLD first-base collator (P25). To claim one principled method on every backbone, DNABERT-2 must be re-trained with the SAME `OffsetAwareDamageCollator` used on GENA-LM, and still beat MLM.

**Setup:** DNABERT-2 (`zhihan1996/DNABERT-2-117M`) fine-tuned with collator `dam_offset` (linear, scale_to 0.15, power 1.0), seeds 42/1/2, 20 epochs, Drive-free Colab. MLM arm reused from the existing `multiseed_models/mlm-baseline-s*` (collator-invariant — only the DAM arm changes). Models → `training/multiseed_models/dam-offset-s{42,1,2}/` (`pytorch_model.bin`, byte-verified 468345303). Training eval_loss 3.83 / 3.71 / 3.82. Eval `evaluation/evaluate_dnabert_offset_multiseed.py` → `evaluation/results_dnabert_offset_multiseed.json` (same 626-window pooled stats as GENA-LM: seed-averaged per-window Δ, one-sample t + Wilcoxon + 10k-bootstrap CI). DNABERT-2 DOES support `offset_mapping`, so the eval ports directly.

**Result (DAM-offset − MLM, pooled over seeds 42/1/2):**
- **TERMINAL** (damaged ends): 3 nt **+0.067** (p=4.9e-12, CI [+0.049,+0.086], n=486); 5 nt **+0.045** (p=9.1e-10, [+0.031,+0.059], n=565); 10 nt **+0.034** (p=1.9e-13, [+0.026,+0.044], n=619). Every seed positive (3 nt: +0.075/+0.061/+0.066).
- **BACKGROUND** (central band): 3 nt **+0.029** (p=7.6e-7); 5 nt **+0.026** (p=9.7e-8); 10 nt **+0.021** (p=8.2e-11). All positive, all seeds same sign. No collapse.

**Reading:** under the nucleotide-principled collator (not the first-base proxy), DNABERT-2 DAM still beats MLM at the terminal on every seed, with the correct damage gradient (edge largest at 3 nt, decaying inward) and NO background cost. So the original DNABERT-2 result (P25's first-base collator) was NOT an artifact of that proxy — it survives the principled method. Both backbones now share ONE collator and show the SAME pattern (terminal win + positive-small background = balanced reconstructor). D2 architecture independence is demonstrated on two backbones from different labs with different tokenizers.

**Honest nuance (carry into the Phase-4 rigor pass):** offset-aware DAM's terminal edge on DNABERT-2 (+0.067 @ 3 nt) is SMALLER than the first-base DAM's (+0.103 @ 3 nt, the Phase-0 `evaluate_scaling_multiseed` number), but it gains a clear background improvement the first-base version did not have (+0.029 vs ~0). This is the same soft-vs-hard tradeoff seen with pmd-mask (P24), now internal to DAM: the first-base proxy happened to concentrate masking harder at C/G-first tokens near the ends; the offset-aware version spreads over all C/G globally, so it is more balanced and slightly less terminal-peaked. Both beat MLM with overwhelming significance. For the unified cross-backbone table, STANDARDIZE on the offset-aware model set — it is the one method shared across DNABERT-2 + GENA-LM (+ NT to come), and it resolves the 38.6% vs 36.3% checkpoint mismatch by picking a single self-consistent set.

**Cross-backbone consistency (both under the identical offset-aware collator, pooled Δ):**
| | terminal 3 nt | terminal 10 nt | background 3 nt |
|---|---|---|---|
| DNABERT-2 | +0.067 | +0.034 | +0.029 |
| GENA-LM | +0.110 | +0.048 | +0.015 |

Same sign, same gradient, same balanced-reconstructor shape on two independent backbones. (Absolute magnitudes are not comparable across backbones — different tokenizers / vocab / pretraining — only the pattern is.)

**Decision / next:** D2 is now two backbones strong. Proceed to the third backbone (Nucleotide Transformer): the `OffsetAwareDamageCollator` ports unchanged (it reconstructs positions from vocab token strings via cumsum of token lengths — no `offset_mapping` needed at train time), and `build_dataset.py` is already backbone-parametrized (`VESTIGE_MODEL`/`VESTIGE_OUT`). Only NT-specific work = an eval that rebuilds offsets from NT's 6-mer token strings (NT's slow tokenizer likely lacks `offset_mapping`).

## P29 — Nucleotide Transformer Confirms DAM on a THIRD Backbone: D2 Architecture Independence Is Overwhelming

**Phase:** Depth study D2, third backbone. DNABERT-2 (BPE, MosaicBERT) and GENA-LM (BPE, pre-LN BERT) are both Transformer-encoder BERT variants with byte-pair vocabularies. NT is a genuinely different design — InstaDeep, ESM-style architecture with rotary embeddings, and a NON-OVERLAPPING 6-MER tokenizer (not BPE) — so it stresses the method on a vocabulary where every token is a fixed 6 nt.

**Setup:** `InstaDeepAI/nucleotide-transformer-v2-50m-multi-species`, fine-tuned 3 MLM + 3 DAM (seeds 42/1/2), 20 epochs, Drive-free Colab. NT has NO reusable MLM arm (unlike DNABERT-2, where the paper's `mlm-baseline-s*` was reused), so all 6 arms trained fresh. Loads cleanly via `AutoModelForMaskedLM` → `EsmForMaskedLM`, missing_keys=0 (NO `model_class` override — P26 trap checked and absent); base masked-LM loss on real DNA 0.036 vs random 8.32. Dataset `training/dataset_nt` (275/69, same windows as the other backbones). Models `training/multiseed_models/{mlm-nt,dam-offset-nt}-s{42,1,2}/`, training eval_loss MLM 5.31/5.24/5.25 ≈ DAM 5.29/5.26/5.25 (equal overall fit). Eval `evaluation/evaluate_nt_multiseed.py` → `results_nt_multiseed.json`. NT's slow tokenizer has no `offset_mapping`, so offsets are rebuilt by cumsum over per-token nucleotide-string lengths — EXACT here because all 626 windows are pure uppercase ACGT (no `<unk>` desync; a coverage guard asserts the reconstructed spans reach len(dna)). This is the SAME position model the `OffsetAwareDamageCollator` uses at train time, so eval and training agree by construction.

**Result (DAM-offset − MLM, pooled over seeds 42/1/2):**
- **TERMINAL**: 3 nt **+0.233** (p=1.1e-44, CI [+0.204,+0.263], n=486); 5 nt **+0.173** (p=1.5e-44, [+0.151,+0.196], n=565); 10 nt **+0.112** (p=3.2e-52, [+0.099,+0.126], n=619). Every seed positive and tight (3 nt: +0.218/+0.248/+0.234).
- **BACKGROUND**: 3 nt **+0.018** (p=6.5e-4); 5 nt **+0.035** (p=1.1e-13); 10 nt **+0.050** (p=1.7e-37). All positive, all seeds same sign. No collapse.

**Reading:** DAM beats MLM at the terminal on NT too, with overwhelming significance and the correct damage gradient (largest at 3 nt, decaying inward). NT's terminal edge (+0.23 @ 3 nt) is the LARGEST of the three backbones — plausibly because a terminal base sits inside a fixed 6-mer token, so uniform MLM almost never targets it specifically while DAM concentrates masking on exactly those damage-relevant tokens, giving a larger relative gain. Background is positive everywhere (no collapse); note NT's background Δ INCREASES with band width (0.018 → 0.050) rather than decreasing — the interior DAM advantage widens — but it never goes negative, so the balanced-reconstructor claim holds. Absolute magnitudes are NOT comparable across backbones (different tokenizers/vocab/pretraining); only the PATTERN is, and all three backbones show the identical pattern.

**Three-backbone cross-consistency (all under the identical offset-aware collator, pooled Δ):**
| Backbone (lab / tokenizer / arch) | terminal 3 nt | terminal 10 nt | background 3 nt |
|---|---|---|---|
| DNABERT-2 (Zhihan / BPE / MosaicBERT) | +0.067 | +0.034 | +0.029 |
| GENA-LM (AIRI / BPE / pre-LN BERT) | +0.110 | +0.048 | +0.015 |
| NT (InstaDeep / 6-mer / ESM-style) | +0.233 | +0.112 | +0.018 |

Three labs, three tokenizers, three architectures (including a non-BPE k-mer vocabulary), every seed same sign, same damage gradient, same no-collapse background. This is decisive architecture-independence evidence — DAM is a general principle, not a DNABERT-2/BPE/aDNA artifact.

**Decision / next:** D2 (architecture independence) is COMPLETE on three backbones — Phase 1 done. Remaining depth-study work is all DNABERT-2 (no new backbone, fast runs, MLM reused): Phase 2 D3 (synthetic corruption-shape sweep + FFPE = domain independence), Phase 3 pmd-mask tradeoff curve, Phase 4 rigor pass + manuscript. The unified cross-backbone table above is the Phase-4 headline; standardize all DNABERT-2 numbers on the offset-aware set.

## P30 — Run 2 (D3 Synthetic + pmd-mask Tradeoff): Corruption-Shape Independence CONFIRMED; the Power Knob Is INERT and P24's "pmd Collapse" Does NOT Reproduce

**Phase:** Depth study Phase 2 (D3 synthetic corruption-shape sweep) + Phase 3 (pmd-mask tradeoff curve). All DNABERT-2; MLM / dam-offset (power 1) / pmd-mask arms reused from the offset-aware multiseed set (P28). 12 new arms trained headless on a Kaggle GPU (6 synthetic + 6 tradeoff power 2/4), seeds 42/1/2, 20 epochs. (Infra for reproducibility: Kaggle's Python-3.13 triton is newer than DNABERT-2's bundled flash-attention kernels (`tl.dot(..., trans_b=True)` removed), so `train.py` now makes triton look absent via `sys.modules["triton"] = None` → DNABERT-2 takes its torch-attention fallback, the same path as Apple MPS. The kernel also harvests each arm's `pytorch_model.bin` and deletes its optimizer-state checkpoints per arm, or `/kaggle/working` fills at ~arm 10.)

### D3 synthetic corruption-shape sweep — STRONG WIN
Setup: `PositionProfileCollator` masks C/G by an arbitrary per-absolute-position weight w[p] (not distance-from-end). Two deliberately UN-terminal shapes: `central` (Gaussian bump mid-window, σ=L/12 — interior is the damaged zone, ends clean) and `periodic` (sin², 50 bp — no notion of end at all). DAM-matched arm (`syn-<profile>-s*`) vs reused uniform MLM (`mlm-baseline-s*`). Corruption zone = w[p] ≥ 0.5·max(w); score C/G recovery there. Eval `evaluation/evaluate_synthetic_multiseed.py` → `results_synthetic_multiseed.json`, same pooled stats as the backbone evals.
Result (DAM-matched − MLM, pooled seeds 42/1/2):
- **central** (zone n=392): per-seed +0.029 / +0.062 / +0.041, pooled **+0.0437**, p=5.1e-45, CI [+0.038, +0.049].
- **periodic** (zone n=1000): per-seed +0.023 / +0.024 / +0.027, pooled **+0.0248**, p=4.1e-105, CI [+0.023, +0.027].
Reading: DAM matched to an arbitrary, un-terminal corruption shape beats uniform MLM at recovering that shape's damaged positions — every seed, both shapes, CIs clear of zero. "Match training masking to the corruption distribution" is CORRUPTION-SHAPE-AGNOSTIC, not an ends-are-special artifact. This is the domain / corruption-shape independence axis to pair with three-backbone architecture independence (P29).

### Tradeoff curve — the power knob is INERT, and P24's pmd background-collapse does NOT reproduce
Setup: offset-aware DAM on the REAL aDNA profile at power 1 (= `dam-offset`, reused) / 2 / 4, vs reused MLM and pmd-mask. Terminal + background (mid-window) C/G recovery at w=3/5/10. Eval `evaluation/evaluate_tradeoff_multiseed.py` → `results_tradeoff_multiseed.json`.
Result (pooled seeds, recovery at w = 3 / 5 / 10):
| arm | t@3 | t@5 | t@10 | bg@3 | bg@5 | bg@10 |
|---|---|---|---|---|---|---|
| mlm | 0.206 | 0.243 | 0.265 | 0.372 | 0.342 | 0.319 |
| p1 (dam-offset) | 0.273 | 0.287 | 0.300 | 0.401 | 0.368 | 0.340 |
| p2 | 0.267 | 0.291 | 0.304 | 0.388 | 0.359 | 0.332 |
| p4 | 0.269 | 0.294 | 0.309 | 0.388 | 0.360 | 0.332 |
| pmd | 0.514 | 0.496 | 0.474 | 0.390 | 0.364 | 0.348 |

Two honest, UNEXPECTED findings:
1. **The power knob is inert.** p1 ≈ p2 ≈ p4 on BOTH axes — raising power 1→2→4 barely moves terminal (0.273→0.267→0.269 @3) or background. There is NO soft→hard continuum; the three DAM variants sit on one operating point. **Mechanism (verified, `scratchpad/diag_power.py`):** power DOES sharpen the per-token extreme-terminal mask-prob (mean at 0–2 nt: 0.22 → 0.37 → 0.50 for power 1/2/4, max → 1.0), BUT deamination-susceptible terminal tokens are only ~113 of ~16,000 tokens (<1%), so even power 4 puts just 3.7% of the 15% budget within 10 nt of a terminus (power 1 = 1.8%). DAM always spreads its budget across the WHOLE sequence, so >96% of the training signal is interior reconstruction regardless of power, and the models converge. pmd-mask is a DIFFERENT regime — terminal-only masking, ~100% of a smaller budget at the ends → terminal 0.514. The concentration knob cannot bridge soft-DAM to pmd-mask because the terminus is a negligible fraction of positions. Structural, not a bug.
2. **P24 REVISED: pmd-mask does NOT collapse on background under the offset-aware multiseed protocol.** pmd bg = 0.390 / 0.364 / 0.348 is ≈ or BETTER than DAM (0.401 / 0.368 / 0.340) at wider windows, while pmd terminal (0.514) dwarfs DAM (0.273). The earlier P24 "pmd beats terminal but COLLAPSES on bg → DAM is the balanced champion" was an artifact of the first-base eval and does NOT survive the rigorous protocol. Caveat: the background metric is weakly discriminating (all arms 0.37–0.40 vs terminal spanning 0.21–0.51), so "no collapse" partly reflects that interior C/G reconstruction is an easy, masking-strategy-insensitive task.
(DAM still beats MLM on both axes — p1 Δterm +0.067 / Δbg +0.029 @3 — so the DAM>MLM result is untouched; only the DAM-vs-pmd comparison changed.)

### Decision (framing, user 2026-11-08): pmd-mask = REFERENCE POINT, not rival
DAM's contribution rests on **DAM > MLM** — architecture independence (P28/P29, three backbones) + corruption-shape independence (D3 above) — NOT on beating pmd-mask. In the paper, pmd-mask is a domain-specific terminal-only reference that specializes on the terminal at the cost of generality (it is BUILT from the known aDNA damage profile and masks only the ends; it does not transfer to other corruption shapes — the D3 synthetic sweep is exactly the setting it cannot address). DROP every "pmd-mask collapses on background" claim (retires the P24 talking point). The tradeoff-curve experiment is NOT a paper figure as a continuum; if referenced at all, report it honestly as "soft-DAM and pmd-mask are distinct masking regimes a concentration knob cannot interpolate, because susceptible termini are <1% of positions." Remaining at the time = Phase 4 rigor pass + the optional FFPE leg + manuscript.

## P31 — D3 FFPE (real clinical deamination domain): STRONG WIN — DAM generalises beyond ancient DNA

**Phase:** Depth study Phase 2 (D3), the REAL-domain leg that completes "domain independence" alongside the synthetic shapes (P30). All DNABERT-2; MLM reused from `mlm-baseline-s*`; 3 FFPE-matched DAM arms trained headless on Kaggle (`angshuman12/vestige-ffpe-run`, dataset v4), seeds 42/1/2, same train.py (triton-absent fix) + per-arm checkpoint harvest.

**FFPE damage shape (grounded, not invented):** formalin-fixed clinical tissue DNA carries the SAME C→T / G→A cytosine deamination as aDNA, but its positional signature is NEAR-UNIFORM along the read, not aDNA's terminal "smiley" decay — formalin deaminates single- and double-stranded DNA alike, so there is no single-stranded-overhang gradient (FFPE error-profile preprint bioRxiv 2023.03.02.530819, "almost uniform distribution of C→T variant rate"; mechanism Do & Dobrovic 2015). Modeled in `evaluation/make_ffpe_profile.py` → `damage/synthetic/ffpe.npy` as a near-flat per-position weight (baseline + mild 5'/3' bump; mid 0.833 / ends 0.859, zone w≥0.5·max covers 100% of the read). Fed to the SAME `PositionProfileCollator` as the synthetic sweep. HONEST CAVEAT: this is literature-SHAPED (near-uniform), not derived from a specific FFPE BAM via mapDamage2 — the shape is the scientifically meaningful part (the collator rescales every profile to the 15% budget), but a real-BAM profile is the stronger version if a reviewer asks.

**Result (DAM-matched − uniform MLM, pooled seeds 42/1/2, `evaluate_ffpe_multiseed.py` → `results_ffpe_multiseed.json`):** zone = 2000 positions (whole read, confirming near-uniform). Per-seed Δ +0.0317 / +0.0931 / +0.0717 (all positive), **pooled Δ +0.0655**, p_t=3e-130, Wilcoxon=3e-93, CI [+0.0613, +0.0696], n=626.

**Reading:** DAM matched to the flat FFPE profile beats MLM at recovering C/G across the ENTIRE read — the LARGEST D3 effect (> central +0.0437 > periodic +0.0248). Mechanism: when damage is uniform, "match the mask to the corruption distribution" reduces to "mask the deamination-susceptible bases (C/G) everywhere", which beats uniform MLM (that wastes budget on A/T). So the principle holds even when the corruption has NO positional structure, in a REAL clinical sequencing domain — the strongest form of the "not just ancient DNA" generalisation the paper previously only asserted. Adds a third row to the D3 table (Table 3, `phase4_tables.md`): central / periodic (synthetic) + FFPE (real).

**Decision / next:** D3 is now COMPLETE on both legs (synthetic shapes + real FFPE domain). Phase 2 fully done. **All GPU work for the depth study is now COMPLETE.** Remaining = apply the manuscript change-list (§7 in `12_phase4_rigor.md`, now including the FFPE row) — user-led, no GPU. Models `training/multiseed_models/ffpe-s{42,1,2}/` (gitignored).
