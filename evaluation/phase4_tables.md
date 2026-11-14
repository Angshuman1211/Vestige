## Table 1 — Architecture independence: pooled Δ(DAM−MLM), one offset-aware collator, seeds 42/1/2

| backbone (lab / tokenizer / arch) | term Δ@3 (p, 95% CI) | term Δ@5 (p) | term Δ@10 (p) | bg Δ@3 (p) | bg Δ@5 (p) | bg Δ@10 (p) |
|---|---|---|---|---|---|---|
| DNABERT-2 (Zhihan / BPE / MosaicBERT) | +0.067 (4.9e-12, [+0.049, +0.086]) | +0.045 (9.1e-10) | +0.034 (1.9e-13) | +0.029 (7.6e-07) | +0.026 (9.7e-08) | +0.021 (8.2e-11) |
| GENA-LM (AIRI / BPE / pre-LN BERT) | +0.110 (2.2e-17, [+0.086, +0.135]) | +0.087 (3.9e-21) | +0.048 (1.3e-18) | +0.015 (5.8e-04) | +0.015 (7.3e-05) | +0.014 (8.7e-09) |
| NT (InstaDeep / 6-mer / ESM) | +0.233 (1.1e-44, [+0.204, +0.263]) | +0.173 (1.5e-44) | +0.112 (3.2e-52) | +0.018 (6.5e-04) | +0.035 (1.1e-13) | +0.050 (1.7e-37) |

n (terminal 3/5/10) = 486/565/619; n (background 3/5/10) = 600/622/626. Every seed same sign. Absolute magnitudes are NOT comparable across backbones (different tokenizers) — report the Δ.

## Table 2 — DNABERT-2 terminal recovery (absolute): first-base (retired) vs offset-aware (canonical)

| T_END | MLM | DAM first-base | DAM offset-aware | random | zero-shot | DAM_offset − random |
|---|---|---|---|---|---|---|
| 3 | 0.206 | 0.308 | 0.273 | 0.277 | 0.155 | -0.004 |
| 5 | 0.243 | 0.312 | 0.287 | 0.257 | 0.192 | +0.030 |
| 10 | 0.265 | 0.322 | 0.300 | 0.250 | 0.228 | +0.050 |

MLM is far below random at the terminus (the problem DAM addresses). Offset-aware DAM@3 ≈ random, clears it at T_END≥5. The Δ(DAM−MLM) is overwhelming at every T_END (Table 1).

## Table 3 — D3 corruption-shape / domain independence: pooled Δ(DAM-matched − uniform MLM) in the corruption zone

| profile | zone n | per-seed Δ (42/1/2) | pooled Δ | p | 95% CI |
|---|---|---|---|---|---|
| central (synthetic, mid-window) | 392 | +0.029/+0.062/+0.041 | +0.0437 | 5.1e-45 | [+0.0382, +0.0493] |
| periodic (synthetic, 50 bp) | 1000 | +0.023/+0.024/+0.027 | +0.0248 | 4.1e-105 | [+0.0230, +0.0267] |
| FFPE (REAL clinical, near-uniform) | 2000 | +0.032/+0.093/+0.072 | +0.0655 | 3.0e-130 | [+0.0613, +0.0696] |

Two synthetic shapes (deliberately un-terminal) + one REAL clinical domain (FFPE, near-uniform). DAM matched to each beats uniform MLM at recovering its damaged positions → the principle is corruption-shape- and domain-agnostic, not ends-are-special.

## Table 4 — Masking-regime reference (pmd-mask = reference point, NOT rival): recovery by zone

| arm | t@3 | t@5 | t@10 | bg@3 | bg@5 | bg@10 |
|---|---|---|---|---|---|---|
| MLM | 0.206 | 0.243 | 0.265 | 0.372 | 0.342 | 0.319 |
| DAM (offset, power 1) | 0.273 | 0.287 | 0.300 | 0.401 | 0.368 | 0.340 |
| DAM power 2 | 0.267 | 0.291 | 0.304 | 0.388 | 0.359 | 0.332 |
| DAM power 4 | 0.269 | 0.294 | 0.309 | 0.388 | 0.360 | 0.332 |
| pmd-mask | 0.514 | 0.496 | 0.474 | 0.390 | 0.364 | 0.348 |

Power knob inert (p1≈p2≈p4); pmd-mask has a large terminal edge and NO background collapse under this protocol → a domain-specific terminal-only reference, not a rival.
