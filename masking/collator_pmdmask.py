"""
PmdMaskCollator — pmd-mask-style deterministic damage-site masking baseline.

Prior art: pmd-mask (Lefeuvre 2023) consumes mapDamage output and hard-masks the
PMD-affected sites for variant calling. As a training-masking baseline comparable
to DAM, it is framed as the hard-thresholding counterpart of DAM's soft weighting:

    DAM       — mask each C/G position with probability proportional to its
                empirical damage frequency (soft, stochastic).
    pmd-mask  — deterministically mask the MOST-damaged C/G positions (the sites
                pmd-mask would flag), greedy to the same budget (hard, top-k).

Both target the same deamination-concentrated region at a matched 15% C/G budget,
so the head-to-head isolates one variable: proportional weighting vs hard
thresholding. Identical 80/10/10 corruption to MLM/DAM.

Usage (matches DAM):
    from masking.collator_pmdmask import PmdMaskCollator
    collator = PmdMaskCollator(tokenizer=tokenizer,
                               damage_profile_path="damage/damage_profile.npy")

Run unit test from project root:
    python masking/collator_pmdmask.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


class PmdMaskCollator(DataCollatorForLanguageModeling):
    """
    Args:
        tokenizer:            HuggingFace tokenizer (DNABERT-2 AutoTokenizer).
        damage_profile_path:  Path to damage_profile.npy (same file DAM uses).
        mlm_probability:      Fraction of C/G positions to mask (default 0.15),
                              matching DAM's scaled C/G masking density.
    """

    def __init__(
        self,
        tokenizer,
        damage_profile_path: str | Path,
        mlm_probability: float = 0.15,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", mlm_probability)
        super().__init__(tokenizer=tokenizer, **kwargs)

        profile = np.load(damage_profile_path, allow_pickle=True).item()
        self.ct_5p = torch.tensor(profile["ct_5p"], dtype=torch.float32)
        self.ga_3p = torch.tensor(profile["ga_3p"], dtype=torch.float32)
        self._base_lookup = self._build_base_lookup()

    def _build_base_lookup(self) -> torch.Tensor:
        vocab = self.tokenizer.get_vocab()
        lookup = torch.zeros(len(vocab), dtype=torch.uint8)
        base_map = {"A": 1, "C": 2, "G": 3, "T": 4}
        for token, idx in vocab.items():
            clean = token.lstrip("#").lstrip("▁").upper()
            if clean and clean[0] in base_map:
                lookup[idx] = base_map[clean[0]]
        return lookup

    def _damage_score(self, inputs: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len) per-token damage frequency; 0 for A/T/non-base."""
        batch_size, seq_len = inputs.shape
        device = inputs.device
        ct = self.ct_5p.to(device)
        ga = self.ga_3p.to(device)
        lookup = self._base_lookup.to(device)

        pos = torch.arange(seq_len, device=device)
        pos_5p = pos.clamp(max=len(ct) - 1)
        pos_3p = (seq_len - 1 - pos).clamp(max=len(ga) - 1)

        base_ids = lookup[inputs]
        is_C = base_ids == 2
        is_G = base_ids == 3

        score = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        ct_row = ct[pos_5p].unsqueeze(0).expand(batch_size, -1)
        ga_row = ga[pos_3p].unsqueeze(0).expand(batch_size, -1)
        score = torch.where(is_C, ct_row, score)
        score = torch.where(is_G, ga_row, score)
        return score, (is_C | is_G)

    def torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        bsz, seq_len = inputs.shape
        device = inputs.device

        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [
                    self.tokenizer.get_special_tokens_mask(
                        row, already_has_special_tokens=True
                    )
                    for row in inputs.tolist()
                ],
                dtype=torch.bool,
                device=device,
            )

        score, cg_mask = self._damage_score(inputs)
        score = score.masked_fill(special_tokens_mask.bool(), -1.0)
        score = score.masked_fill(~cg_mask, -1.0)  # only C/G eligible

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for b in range(bsz):
            n_cg = int(cg_mask[b].sum())
            budget = int(round(self.mlm_probability * n_cg))
            if budget <= 0:
                continue
            # greedy top-k most-damaged C/G positions (deterministic given sequence)
            top = torch.topk(score[b], k=min(budget, seq_len)).indices
            top = top[score[b][top] > 0]  # keep only genuinely damaged (>0) sites
            masked_indices[b, top] = True

        labels[~masked_indices] = -100

        # identical 80/10/10 to MLM/DAM
        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        idx_replace = (
            torch.bernoulli(torch.full(inputs.shape, 0.8, device=device)).bool()
            & masked_indices
        )
        inputs[idx_replace] = mask_id
        idx_random = (
            torch.bernoulli(torch.full(inputs.shape, 0.5, device=device)).bool()
            & masked_indices
            & ~idx_replace
        )
        inputs[idx_random] = torch.randint(
            len(self.tokenizer), inputs.shape, dtype=torch.long, device=device
        )[idx_random]

        return inputs, labels


# ----------------------------------------------------------------------
# Unit test — all-C input: masking must land on the 5' terminal (highest ct),
# at ~15% of C/G positions. Run: python masking/collator_pmdmask.py
# ----------------------------------------------------------------------

class _MockTokenizer:
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

    torch.manual_seed(42)
    n_pos = 70
    positions = np.arange(n_pos)
    ct_5p = 0.05 * np.exp(-0.3 * positions) + 0.001   # decaying toward 5' end
    profile = {"ct_5p": ct_5p, "ga_3p": ct_5p.copy(), "positions": positions + 1}
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        tmp = f.name
    np.save(tmp, profile, allow_pickle=True)

    collator = PmdMaskCollator(tokenizer=_MockTokenizer(),
                               damage_profile_path=tmp, mlm_probability=0.15)
    os.unlink(tmp)

    SEQ_LEN = 60
    inputs = torch.full((1, SEQ_LEN), 5, dtype=torch.long)  # all 'C'
    _, labels = collator.torch_mask_tokens(inputs.clone())
    sel = (labels[0] != -100)
    sel_positions = torch.nonzero(sel).flatten().tolist()

    rate = sel.float().mean().item()
    mean_pos = float(np.mean(sel_positions)) if sel_positions else -1
    print("PmdMaskCollator — selection test (all-C input, 60 tokens):")
    print(f"  Masking rate:                 {rate:.4f}  (target 0.15, all C so C/G=all)")
    print(f"  Selected positions:           {sel_positions}")
    print(f"  Mean selected position:       {mean_pos:.1f}  (5' terminal = low index)")

    assert abs(rate - 0.15) < 0.04, f"FAIL: budget off — {rate:.4f}"
    assert mean_pos < SEQ_LEN / 4, f"FAIL: not 5'-concentrated — mean pos {mean_pos:.1f}"
    print("PASS: masks the most-damaged (5' terminal) C positions at ~15% budget ✓")
