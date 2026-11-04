"""
OffsetAwareDamageCollator — tokenizer-agnostic DAM (D2 enabler).

The original collator_dam.py decides a token is damage-relevant from its FIRST
character. That is a DNABERT-2 accident (its BPE merges almost never start a token
with A, so first-base tracks C/G content ~73% of the time). On other tokenizers
(GENA-LM: 96% of tokens start with A/T) that proxy collapses and DAM masks almost
nothing.

This collator instead reconstructs each token's actual nucleotides and their
ABSOLUTE position in the window from the token string itself (DNA BPE / k-mer
tokens ARE their nucleotide content), then assigns a per-token masking probability
from the damage profile: a token's score is the strongest damage signal among the
C nucleotides it covers (rate by 5' distance, ct_5p) and the G nucleotides it
covers (rate by 3' distance, ga_3p). Rescaled so damage-relevant tokens average
scale_to, matching the MLM budget. Ports to DNABERT-2, GENA-LM, and NT unchanged.

Run the unit test from project root:
    python masking/collator_dam_offset.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    pad = window // 2
    xp = np.pad(x, pad, mode="edge")
    return np.convolve(xp, np.ones(window) / window, mode="valid")[: len(x)]


class OffsetAwareDamageCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        damage_profile_path: str | Path,
        scale_to: float | None = 0.15,
        power: float = 1.0,
        smooth: int = 0,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", scale_to if scale_to is not None else 0.15)
        super().__init__(tokenizer=tokenizer, **kwargs)

        profile = np.load(damage_profile_path, allow_pickle=True).item()
        ct = np.asarray(profile["ct_5p"], dtype=np.float64)
        ga = np.asarray(profile["ga_3p"], dtype=np.float64)
        if smooth and smooth > 1:
            ct, ga = _moving_avg(ct, smooth), _moving_avg(ga, smooth)
        if power != 1.0:
            ct, ga = np.power(ct, power), np.power(ga, power)
        self.ct = torch.tensor(ct, dtype=torch.float32)
        self.ga = torch.tensor(ga, dtype=torch.float32)
        self.scale_to = scale_to
        self.power = power
        self.smooth = smooth
        self._build_token_tables()

    def _build_token_tables(self):
        """Per token-id: nucleotide length + intra-token offsets of its C's and G's.
        Special / non-DNA tokens get length 0 (they do not advance position and are
        never damage-relevant)."""
        vocab = self.tokenizer.get_vocab()
        V = max(vocab.values()) + 1
        lengths = torch.zeros(V, dtype=torch.long)
        c_lists: list[list[int]] = [[] for _ in range(V)]
        g_lists: list[list[int]] = [[] for _ in range(V)]
        for token, idx in vocab.items():
            clean = token.lstrip("#").lstrip("▁").upper()
            if clean and all(ch in "ACGT" for ch in clean):
                lengths[idx] = len(clean)
                c_lists[idx] = [i for i, ch in enumerate(clean) if ch == "C"]
                g_lists[idx] = [i for i, ch in enumerate(clean) if ch == "G"]
        maxC = max((len(x) for x in c_lists), default=1) or 1
        maxG = max((len(x) for x in g_lists), default=1) or 1
        c_off = torch.full((V, maxC), -1, dtype=torch.long)
        g_off = torch.full((V, maxG), -1, dtype=torch.long)
        for idx in range(V):
            for j, o in enumerate(c_lists[idx]):
                c_off[idx, j] = o
            for j, o in enumerate(g_lists[idx]):
                g_off[idx, j] = o
        self.tok_len = lengths
        self.c_off = c_off
        self.g_off = g_off

    def _prob_matrix(self, inputs: torch.Tensor) -> torch.Tensor:
        B, L = inputs.shape
        dev = inputs.device
        tok_len = self.tok_len.to(dev)
        c_off, g_off = self.c_off.to(dev), self.g_off.to(dev)
        ct, ga = self.ct.to(dev), self.ga.to(dev)
        nct, nga = len(ct), len(ga)

        lengths = tok_len[inputs]                              # (B,L)
        starts = torch.cumsum(lengths, dim=1) - lengths        # (B,L) 0-based nt start
        seqlen = lengths.sum(dim=1, keepdim=True)              # (B,1)

        c = c_off[inputs]                                      # (B,L,maxC)
        g = g_off[inputs]
        c_valid, g_valid = c >= 0, g >= 0

        abs_c = starts.unsqueeze(-1) + c.clamp(min=0)          # 5' distance = abs position
        d5 = abs_c.clamp(max=nct - 1)
        c_dmg = torch.where(c_valid, ct[d5], torch.zeros_like(ct[d5]))
        cmax = c_dmg.amax(dim=-1)                              # (B,L)

        abs_g = starts.unsqueeze(-1) + g.clamp(min=0)
        dist3 = (seqlen.unsqueeze(-1) - 1 - abs_g).clamp(min=0, max=nga - 1)
        g_dmg = torch.where(g_valid, ga[dist3], torch.zeros_like(ga[dist3]))
        gmax = g_dmg.amax(dim=-1)

        score = torch.maximum(cmax, gmax)                      # (B,L)

        if self.scale_to is not None:
            relevant = c_valid.any(-1) | g_valid.any(-1)       # tokens with any C or G
            if relevant.any():
                mean_s = score[relevant].mean()
                if mean_s > 0:
                    score = (score * (self.scale_to / mean_s)).clamp(max=1.0)
        return score

    def torch_mask_tokens(
        self, inputs: torch.Tensor,
        special_tokens_mask: Optional[torch.Tensor] = None, **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        prob = self._prob_matrix(inputs)
        if special_tokens_mask is None:
            special_tokens_mask = torch.tensor(
                [self.tokenizer.get_special_tokens_mask(row, already_has_special_tokens=True)
                 for row in inputs.tolist()],
                dtype=torch.bool, device=inputs.device)
        prob.masked_fill_(special_tokens_mask.bool(), 0.0)

        masked_indices = torch.bernoulli(prob).bool()
        labels[~masked_indices] = -100

        mask_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        idx_replace = torch.bernoulli(
            torch.full(inputs.shape, 0.8, device=inputs.device)).bool() & masked_indices
        inputs[idx_replace] = mask_id
        idx_random = (torch.bernoulli(torch.full(inputs.shape, 0.5, device=inputs.device)).bool()
                      & masked_indices & ~idx_replace)
        inputs[idx_random] = torch.randint(
            len(self.tokenizer), inputs.shape, dtype=torch.long, device=inputs.device)[idx_random]
        return inputs, labels
