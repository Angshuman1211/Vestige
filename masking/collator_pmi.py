"""
PmiMaskingCollator — PMI-Masking baseline (Levine et al., 2021).

Content-adaptive masking: instead of masking uniformly (MLM), by damage (DAM),
or by contiguity (SpanBERT), PMI-Masking preferentially masks tokens that form
strong collocations — high pointwise mutual information (PMI) adjacent pairs —
forcing the model to recover correlated spans rather than locally-predictable
tokens. This is the "mask by content/difficulty" comparator the TCBB editor asked
for, and it is distinct from the other baselines (selection by corpus statistics,
not position or randomness).

Interface matches MLM/DAM/Span/PmdMask: subclass DataCollatorForLanguageModeling,
override torch_mask_tokens, keep the identical 80/10/10 corruption so the only
variable is WHICH tokens are selected.

STATUS: skeleton. Boilerplate (base lookup, 80/10/10, budget) is done.
The two methods that ARE the PMI idea are left as TODO (your build):
    _compute_pmi(self, token_sequences)  — precompute the PMI table from the corpus
    _select(self, inputs_row)            — pick the top-PMI tokens to the budget

Run unit test from project root once the TODOs are filled:
    python masking/collator_pmi.py
"""
from __future__ import annotations

from collections import Counter
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


class PmiMaskingCollator(DataCollatorForLanguageModeling):
    """
    Args:
        tokenizer:        HuggingFace tokenizer (DNABERT-2 AutoTokenizer).
        token_sequences:  Iterable of token-id lists (the training corpus) used to
                          precompute PMI. In train.py pass the tokenised train split.
        mlm_probability:  Fraction of tokens to select (default 0.15), matched budget.
    """

    def __init__(
        self,
        tokenizer,
        token_sequences,
        mlm_probability: float = 0.15,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", mlm_probability)
        super().__init__(tokenizer=tokenizer, **kwargs)
        # token_pmi[token_id] = that token's best adjacent-pair PMI (filled by _compute_pmi)
        self.token_pmi: dict[int, float] = {}
        self._compute_pmi(token_sequences)

    # ------------------------------------------------------------------
    # TODO (your build) 1/2 — precompute the PMI table from the corpus
    # ------------------------------------------------------------------
    def _compute_pmi(self, token_sequences) -> None:
        """
        Fill self.token_pmi: token_id -> scalar PMI score.

        PMI of an adjacent pair (a, b):
            PMI(a,b) = log2( P(a,b) / (P(a) * P(b)) )
        where P(a) = count(a)/N_tokens and P(a,b) = count(a,b)/N_bigrams.

        Suggested recipe:
          1. One pass over token_sequences: Counter of unigrams, Counter of adjacent
             bigrams, totals N_tokens and N_bigrams.
          2. For each bigram, compute PMI (optionally keep only pairs with
             count(a,b) >= min_count, e.g. 5, to avoid noisy high-PMI singletons).
          3. token_pmi[t] = max PMI over every bigram t participates in (left or
             right neighbour). Tokens never in a kept bigram get the min PMI.
        Store nothing else you do not need — token_pmi is all _select reads.
        """
        raise NotImplementedError("PMI precompute — your build")

    # ------------------------------------------------------------------
    # TODO (your build) 2/2 — select the top-PMI tokens to the budget
    # ------------------------------------------------------------------
    def _select(self, inputs_row: torch.Tensor, special_row: torch.Tensor) -> torch.Tensor:
        """
        Return a bool mask (seq_len,) selecting ~mlm_probability tokens, preferring
        the highest-PMI tokens (strong collocations).

        Suggested recipe:
          1. budget = round(mlm_probability * (#non-special tokens)).
          2. score each position by self.token_pmi.get(token_id, min_pmi);
             set special tokens to -inf so they are never picked.
          3. take the top-`budget` positions by score (torch.topk). Optional: grow
             each picked token to its high-PMI neighbour so whole collocations mask
             together (closer to the paper); the plain top-k version is a fine start.
        """
        raise NotImplementedError("PMI-weighted selection — your build")

    # ------------------------------------------------------------------
    # DONE — identical 80/10/10 to MLM/DAM/Span/PmdMask; only _select differs
    # ------------------------------------------------------------------
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

        masked_indices = torch.zeros_like(inputs, dtype=torch.bool)
        for b in range(bsz):
            masked_indices[b] = self._select(
                inputs[b], special_tokens_mask[b]
            ).to(device)

        labels[~masked_indices] = -100

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
# Unit test (fill in once the TODOs are done). Suggested checks:
#   - budget ~15% met on a random token sequence
#   - high-PMI tokens selected more often than low-PMI tokens on a corpus where
#     one bigram is planted far above chance
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("PmiMaskingCollator skeleton — implement _compute_pmi and _select, then add a test.")
