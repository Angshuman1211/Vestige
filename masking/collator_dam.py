"""
DamageAwareDataCollator — VESTIGE Phase 2 core contribution.

Subclasses HuggingFace DataCollatorForLanguageModeling and overrides
torch_mask_tokens() to assign per-position, per-base masking probabilities
derived from empirical mapDamage2 PMD profiles.

Masking probability assignment:
    C at token position p from 5′  →  ct_5p[ min(p, max_pos) ]
    G at token position p from 3′  →  ga_3p[ min(seq_len-1-p, max_pos) ]
    A / T                          →  baseline_prob  (default 0.03)

By default (scale_to=0.15), the probability matrix is uniformly scaled so
E[masking rate] ≈ 0.15, preserving the relative damage gradient while
matching the MLM baseline training density — making the ablation fair.

Run unit test from project root:
    python masking/collator_dam.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average, edge-padded to preserve length (D1 smooth ablation)."""
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(xp, kernel, mode="valid")[: len(x)]


class DamageAwareDataCollator(DataCollatorForLanguageModeling):
    """
    Args:
        tokenizer:            HuggingFace tokenizer (DNABERT-2 AutoTokenizer).
        damage_profile_path:  Path to damage_profile.npy (output of parse_profiles.py).
        baseline_prob:        Masking probability for A/T bases (default 0.03).
        scale_to:             Scale probability matrix so E[masking] = this value
                              (default 0.15). Set to None to use raw damage frequencies.
    """

    def __init__(
        self,
        tokenizer,
        damage_profile_path: str | Path,
        baseline_prob: float = 0.03,
        scale_to: float | None = 0.15,
        smooth: int = 0,
        power: float = 1.0,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", scale_to if scale_to is not None else baseline_prob)
        super().__init__(tokenizer=tokenizer, **kwargs)

        profile = np.load(damage_profile_path, allow_pickle=True).item()
        # Indices are 0-based: ct_5p[0] = C→T rate at position 1 from 5′ end
        ct = np.asarray(profile["ct_5p"], dtype=np.float64)
        ga = np.asarray(profile["ga_3p"], dtype=np.float64)
        # D1 ablation knobs (defaults = identity, so standard DAM is unchanged):
        #   smooth > 1 : centred moving average over the profile (raw vs smoothed)
        #   power != 1 : prob proportional to freq**power before rescale (mapping shape)
        if smooth and smooth > 1:
            ct = _moving_avg(ct, smooth)
            ga = _moving_avg(ga, smooth)
        if power != 1.0:
            ct = np.power(ct, power)
            ga = np.power(ga, power)
        self.ct_5p = torch.tensor(ct, dtype=torch.float32)
        self.ga_3p = torch.tensor(ga, dtype=torch.float32)
        self.baseline_prob = baseline_prob
        self.scale_to = scale_to
        self.smooth = smooth
        self.power = power

        # Precompute vocab-size lookup: token_id → {0:other, 1:A, 2:C, 3:G, 4:T}
        self._base_lookup: torch.Tensor = self._build_base_lookup()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_base_lookup(self) -> torch.Tensor:
        """Map each token_id to its first DNA base character (0 if not a base token)."""
        vocab = self.tokenizer.get_vocab()
        lookup = torch.zeros(len(vocab), dtype=torch.uint8)
        base_map = {"A": 1, "C": 2, "G": 3, "T": 4}
        for token, idx in vocab.items():
            # Strip BPE prefix markers (## for BERT-style, ▁ for SentencePiece)
            clean = token.lstrip("#").lstrip("▁").upper()
            if clean and clean[0] in base_map:
                lookup[idx] = base_map[clean[0]]
        return lookup

    def _prob_matrix(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Build (batch_size, seq_len) float tensor of per-token masking probabilities.

        Position convention matches the misincorporation.txt convention:
          - Token at index 0 is position 1 from the 5′ end.
          - Token at index seq_len-1 is position 1 from the 3′ end.
        """
        batch_size, seq_len = inputs.shape
        device = inputs.device

        ct = self.ct_5p.to(device)   # (n_pos,)
        ga = self.ga_3p.to(device)   # (n_pos,)
        lookup = self._base_lookup.to(device)

        # Token positions clamped to profile length (70 positions)
        pos = torch.arange(seq_len, device=device)
        pos_5p = pos.clamp(max=len(ct) - 1)                        # 0-based dist from 5′
        pos_3p = (seq_len - 1 - pos).clamp(max=len(ga) - 1)       # 0-based dist from 3′

        base_ids = lookup[inputs]   # (batch, seq_len)

        is_C = base_ids == 2
        is_G = base_ids == 3

        prob = torch.full(
            (batch_size, seq_len), self.baseline_prob,
            dtype=torch.float32, device=device,
        )
        # Broadcast (seq_len,) damage vectors across the batch dimension
        ct_row = ct[pos_5p].unsqueeze(0).expand(batch_size, -1)   # (batch, seq_len)
        ga_row = ga[pos_3p].unsqueeze(0).expand(batch_size, -1)   # (batch, seq_len)
        prob = torch.where(is_C, ct_row, prob)
        prob = torch.where(is_G, ga_row, prob)

        if self.scale_to is not None:
            # Scale over C/G positions only — A/T are 0 and must not dilute the mean.
            # This ensures C/G average exactly scale_to, with the damage gradient preserved.
            cg_mask = is_C | is_G
            mean_p = prob[cg_mask].mean() if cg_mask.any() else prob.mean()
            if mean_p > 0:
                prob = (prob * (self.scale_to / mean_p)).clamp(max=1.0)

        return prob

    # ------------------------------------------------------------------
    # Override
    # ------------------------------------------------------------------

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()

        prob = self._prob_matrix(inputs)

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        row, already_has_special_tokens=True
                    )
                    for row in inputs.tolist()
                ],
                dtype=torch.bool,
                device=inputs.device,
            )
        prob.masked_fill_(special_tokens_mask.bool(), 0.0)

        masked_indices = torch.bernoulli(prob).bool()
        labels[~masked_indices] = -100   # loss computed only on masked tokens

        # 80 % → [MASK] token, 10 % → random token, 10 % → unchanged
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        idx_replace = (
            torch.bernoulli(torch.full(inputs.shape, 0.8, device=inputs.device)).bool()
            & masked_indices
        )
        inputs[idx_replace] = mask_id

        idx_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5, device=inputs.device)).bool()
            & masked_indices
            & ~idx_replace
        )
        inputs[idx_random] = torch.randint(
            len(self.tokenizer), inputs.shape, dtype=torch.long, device=inputs.device
        )[idx_random]

        return inputs, labels


# ----------------------------------------------------------------------
# Unit test
# Verify elevated masking density at terminal positions for all-C input.
# Run: python masking/collator_dam.py
# ----------------------------------------------------------------------

class _MockTokenizer:
    """Minimal tokenizer mock — 'C' lives at token_id 5."""

    mask_token = "[MASK]"
    _vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "[MASK]": 4,
              "C": 5, "G": 6, "A": 7, "T": 8}

    def get_vocab(self):
        return self._vocab

    def get_special_tokens_mask(self, ids, already_has_special_tokens=True):
        specials = {0, 1, 2, 3, 4}
        return [1 if t in specials else 0 for t in ids]

    def convert_tokens_to_ids(self, token):
        return self._vocab[token]

    def __len__(self):
        return len(self._vocab)


if __name__ == "__main__":
    import tempfile, os
    from scipy.stats import chi2_contingency

    torch.manual_seed(42)

    # Synthetic exponentially decaying damage profile
    n_pos = 70
    positions = np.arange(n_pos)
    ct_5p = 0.05 * np.exp(-0.3 * positions) + 0.001
    ga_3p = ct_5p.copy()
    profile = {"ct_5p": ct_5p, "ga_3p": ga_3p, "positions": positions + 1}

    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp = f.name
    np.save(tmp, profile, allow_pickle=True)

    collator = DamageAwareDataCollator(
        tokenizer=_MockTokenizer(),
        damage_profile_path=tmp,
        scale_to=None,   # raw probs — gradient must be visible for the test to be meaningful
    )
    os.unlink(tmp)

    # All-C input: 1 sequence, 60 tokens, token_id=5 ('C')
    SEQ_LEN  = 60
    N_TRIALS = 10_000
    inputs   = torch.full((1, SEQ_LEN), 5, dtype=torch.long)

    mask_counts = torch.zeros(SEQ_LEN)
    for _ in range(N_TRIALS):
        inp = inputs.clone()
        _, labels = collator.torch_mask_tokens(inp)
        mask_counts += (labels[0] != -100).float()

    # Groups for chi-squared test
    # Terminal: first 5 positions from 5' end  (highest expected damage)
    # Central:  positions 20–49               (expected background level)
    TERM_SLICE = slice(0, 5)
    CENT_SLICE = slice(20, 50)

    n_term = TERM_SLICE.stop - TERM_SLICE.start   # 5
    n_cent = CENT_SLICE.stop - CENT_SLICE.start   # 30

    term_masked   = int(mask_counts[TERM_SLICE].sum().item())
    term_unmasked = N_TRIALS * n_term - term_masked
    cent_masked   = int(mask_counts[CENT_SLICE].sum().item())
    cent_unmasked = N_TRIALS * n_cent - cent_masked

    # 2×2 contingency table: [terminal, central] × [masked, not-masked]
    table = [[term_masked, term_unmasked],
             [cent_masked, cent_unmasked]]
    chi2, p, dof, expected = chi2_contingency(table)

    term_rate = mask_counts[TERM_SLICE].mean().item() / N_TRIALS
    cent_rate = mask_counts[CENT_SLICE].mean().item() / N_TRIALS

    print("Chi-squared test — masking distribution, all-C input, raw damage probs:")
    print(f"  Terminal masking rate (pos  1– 5): {term_rate:.5f}")
    print(f"  Central  masking rate (pos 21–50): {cent_rate:.5f}")
    print(f"  Ratio terminal / central:          {term_rate / cent_rate:.1f}×")
    print(f"  χ²({dof}) = {chi2:.1f},  p = {p:.2e}")

    assert p < 0.001, f"FAIL: p = {p:.2e} — not significant at α = 0.001"
    assert term_rate > cent_rate, "FAIL: terminal rate should exceed central rate"
    print("PASS: masking is significantly concentrated at terminal positions (p < 0.001) ✓")
