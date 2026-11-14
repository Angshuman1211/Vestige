"""FFPE (formalin-fixed) deamination profile — the D3 REAL-domain leg.

FFPE tissue DNA carries the same C->T / G->A cytosine deamination as ancient DNA, but its
POSITIONAL signature is different. Published FFPE error-profile data report an almost UNIFORM
C->T rate along the read (formalin deaminates single- and double-stranded DNA alike, so there
is no single-stranded-overhang model), in contrast to aDNA's steep terminal "smiley" decay.

I model FFPE as a near-uniform per-position weight with only a mild 5'/3' enrichment (the
cytosine-deamination component shows slight terminal elevation). This is the deliberately
UN-terminal, REAL-domain counterpart to the aDNA profile: if DAM matched to it still beats
uniform MLM at recovering the deamination-susceptible (C/G) positions, then "match the masking
to the corruption distribution" holds for a real clinical sequencing domain, not just ancient
DNA, and for a corruption shape with essentially no positional structure.

Shape source: FFPE error-profile characterisation reporting a near-uniform C->T rate along the
read (bioRxiv 2023.03.02.530819); deamination mechanism in Do & Dobrovic 2015. The absolute
rate is irrelevant here — the PositionProfileCollator rescales every profile to the 15% MLM
budget, so only the SHAPE (near-flat) matters.

Run from project root:
    python evaluation/make_ffpe_profile.py
"""
from pathlib import Path

import numpy as np

L = 2000
OUT = Path("damage/synthetic")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = np.arange(L, dtype=np.float64)
    amp, decay = 0.2, 50.0                       # mild terminal bump on a flat baseline
    ffpe = 1.0 + amp * (np.exp(-p / decay) + np.exp(-(L - 1 - p) / decay))
    ffpe /= ffpe.max()
    np.save(OUT / "ffpe.npy", ffpe)

    mid = ffpe[L // 3: 2 * L // 3].mean()
    ends = np.concatenate([ffpe[: L // 6], ffpe[-L // 6:]]).mean()
    cover = (ffpe >= 0.5 * ffpe.max()).mean() * 100
    print(f"ffpe saved {OUT/'ffpe.npy'}  mid-third={mid:.3f}  ends={ends:.3f}  "
          f"min={ffpe.min():.3f}  max={ffpe.max():.3f}")
    print(f"near-uniform: zone (w>=0.5*max) covers {cover:.0f}% of positions "
          f"(vs a terminal profile which would cover only the ends)")


if __name__ == "__main__":
    main()
