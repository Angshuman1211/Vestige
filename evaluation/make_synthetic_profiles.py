"""
Synthetic corruption-shape profiles for the D3 sweep.

Each profile is a per-absolute-position weight w[p] over a 2000 bp window, fed to the
PositionProfileCollator. The two shapes are chosen to CONTRAST with ancient DNA's
terminal (U-shaped) damage so the sweep tests corruption-shape independence, not an
ends-are-special artifact:

  central   Gaussian bump centred mid-window (sigma = L/12). The exact opposite of
            terminal damage: the interior is the damaged zone, the ends are clean.
  periodic  sin^2 with a 50 bp period. Damage recurs at fixed intervals along the
            fragment, a shape with no notion of "end" or "middle" at all.

Only the SHAPE matters (the collator rescales each profile to the 15% MLM budget), so
profiles are stored unnormalised in [0, 1]. Run from project root:
    python evaluation/make_synthetic_profiles.py
"""
from pathlib import Path

import numpy as np

L = 2000
OUT = Path("damage/synthetic")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = np.arange(L, dtype=np.float64)

    central = np.exp(-((p - (L - 1) / 2) ** 2) / (2 * (L / 12) ** 2))
    central /= central.max()

    period = 50.0
    periodic = np.sin(np.pi * p / period) ** 2          # in [0,1], 50 bp period

    for name, w in [("central", central), ("periodic", periodic)]:
        f = OUT / f"{name}.npy"
        np.save(f, w)
        # crude shape summary: mean weight in the mid third vs the two end-sixths
        mid = w[L // 3: 2 * L // 3].mean()
        ends = np.concatenate([w[: L // 6], w[-L // 6:]]).mean()
        print(f"{name:<9} saved {f}  mid-third={mid:.3f}  ends={ends:.3f}  max={w.max():.3f}")


if __name__ == "__main__":
    main()
