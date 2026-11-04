"""
SpanMaskingCollator — SpanBERT-style contiguous-span masking baseline.

Selects contiguous spans (geometric span length, SpanBERT default p=0.2,
capped at max_span) until ~mlm_probability of tokens are selected, then applies
the identical 80/10/10 corruption used by MLMCollator and DamageAwareDataCollator.
The only variable versus those baselines is WHICH tokens are selected
(contiguous spans vs uniform random vs damage-weighted), not the corruption
scheme or the masking budget — isolating the selection strategy for a fair
comparison on the terminal-zone NRR evaluation.

Usage (matches DAM/MLM):
    from masking.collator_span import SpanMaskingCollator
    collator = SpanMaskingCollator(tokenizer=tokenizer, mlm_probability=0.15)

Run unit test from project root:
    python masking/collator_span.py
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


class SpanMaskingCollator(DataCollatorForLanguageModeling):
    """
    Args:
        tokenizer:        HuggingFace tokenizer (DNABERT-2 AutoTokenizer).
        mlm_probability:  Fraction of tokens to select per sequence (default 0.15),
                          matching the MLM and DAM masking budget.
        geometric_p:      Geometric parameter for span lengths (SpanBERT uses 0.2).
        max_span:         Maximum span length (SpanBERT caps at 10).
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability: float = 0.15,
        geometric_p: float = 0.2,
        max_span: int = 10,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", mlm_probability)
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.geometric_p = geometric_p
        self.max_span = max_span

    # ------------------------------------------------------------------
    # Span selection — the only piece that differs from the other collators
    # ------------------------------------------------------------------

    def _select_spans(self, seq_len: int, special_row: torch.Tensor) -> torch.Tensor:
        """Bool mask (seq_len,) selecting ~mlm_probability tokens as contiguous spans."""
        budget = int(round(self.mlm_probability * seq_len))
        sel = torch.zeros(seq_len, dtype=torch.bool)
        valid = ~special_row.bool()
        if budget == 0 or not valid.any():
            return sel

        guard = 0
        while int(sel.sum()) < budget and guard < 20 * seq_len:
            guard += 1
            span_len = min(int(np.random.geometric(self.geometric_p)), self.max_span)
            start = int(np.random.randint(0, seq_len))
            for i in range(start, min(start + span_len, seq_len)):
                if valid[i] and not sel[i]:
                    sel[i] = True
                    if int(sel.sum()) >= budget:
                        break
        return sel

    # ------------------------------------------------------------------
    # Override — identical 80/10/10 to MLM/DAM, only the selection differs
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
            masked_indices[b] = self._select_spans(
                seq_len, special_tokens_mask[b].detach().cpu()
            ).to(device)

        labels[~masked_indices] = -100  # loss only on selected tokens

        # 80 % → [MASK], 10 % → random token, 10 % → unchanged (identical to MLM/DAM)
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
# Unit test — verify budget (~15%) is met and selections are contiguous spans.
# Uses only torch/numpy (no scipy). Run: python masking/collator_span.py
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
    torch.manual_seed(42)
    np.random.seed(42)

    SEQ_LEN, N_TRIALS, P = 60, 4000, 0.15
    collator = SpanMaskingCollator(tokenizer=_MockTokenizer(), mlm_probability=P)
    inputs = torch.full((1, SEQ_LEN), 5, dtype=torch.long)  # all 'C', no special tokens

    total_masked = 0
    run_lengths = []  # lengths of contiguous selected runs, to confirm span structure
    for _ in range(N_TRIALS):
        sel = collator._select_spans(SEQ_LEN, torch.zeros(SEQ_LEN, dtype=torch.bool))
        total_masked += int(sel.sum())
        # measure contiguous run lengths
        run = 0
        for v in sel.tolist():
            if v:
                run += 1
            elif run:
                run_lengths.append(run); run = 0
        if run:
            run_lengths.append(run)

    mean_rate = total_masked / (N_TRIALS * SEQ_LEN)
    mean_run = float(np.mean(run_lengths))
    frac_multi = float(np.mean([r >= 2 for r in run_lengths]))

    print("SpanMaskingCollator — selection test (all-C input, 60 tokens):")
    print(f"  Mean masking rate:            {mean_rate:.4f}  (target {P})")
    print(f"  Mean contiguous run length:   {mean_run:.2f} tokens")
    print(f"  Fraction of runs length >= 2: {frac_multi:.2f}  (spans, not singletons)")

    assert abs(mean_rate - P) < 0.03, f"FAIL: budget off — {mean_rate:.4f} vs {P}"
    assert mean_run > 1.3, f"FAIL: selections not span-like — mean run {mean_run:.2f}"
    print("PASS: budget met and selections form contiguous spans ✓")
