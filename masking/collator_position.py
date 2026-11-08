"""
PositionProfileCollator — arbitrary-shape corruption masking (D3 synthetic sweep).

The OffsetAwareDamageCollator encodes the ANCIENT-DNA corruption shape: it masks a
C by its 5' distance (ct_5p) and a G by its 3' distance (ga_3p), i.e. damage is a
function of distance-from-terminus. That is the right inductive bias for aDNA, but
it builds the terminal shape in by construction, so it cannot test a DIFFERENT
corruption shape.

This collator generalises the idea to ANY position-dependent corruption: it reads a
single per-absolute-position weight profile w[p] and masks each token by the
strongest w among the deamination-susceptible (C or G) nucleotides it covers,
rescaled so damage-relevant tokens average scale_to (the MLM budget). Feed it a
central / periodic / arbitrary w and the DAM-matched arm learns to reconstruct THAT
shape. If a DAM arm matched to w beats uniform MLM at recovering w's high-weight
positions, the claim "match training masking to the corruption distribution" is
corruption-shape-agnostic, not an ends-are-special artifact.

The token->nucleotide reconstruction (string-based, tokenizer-agnostic) is identical
to OffsetAwareDamageCollator, so this ports across backbones unchanged too.

Run the unit test from project root:
    python masking/collator_position.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


class PositionProfileCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        tokenizer,
        profile_path: str | Path,
        scale_to: float | None = 0.15,
        **kwargs,
    ):
        kwargs.setdefault("mlm", True)
        kwargs.setdefault("mlm_probability", scale_to if scale_to is not None else 0.15)
        super().__init__(tokenizer=tokenizer, **kwargs)

        w = np.load(profile_path, allow_pickle=True)
        w = np.asarray(w, dtype=np.float64).ravel()      # per-absolute-position weight
        self.w = torch.tensor(w, dtype=torch.float32)
        self.scale_to = scale_to
        self._build_token_tables()

    def _build_token_tables(self):
        """Per token-id: nucleotide length + intra-token offsets of its C's and G's.
        Special / non-DNA tokens get length 0. Identical to OffsetAwareDamageCollator."""
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
        w = self.w.to(dev)
        nw = len(w)

        lengths = tok_len[inputs]                              # (B,L)
        starts = torch.cumsum(lengths, dim=1) - lengths        # (B,L) 0-based nt start

        c = c_off[inputs]                                      # (B,L,maxC)
        g = g_off[inputs]
        c_valid, g_valid = c >= 0, g >= 0

        abs_c = (starts.unsqueeze(-1) + c.clamp(min=0)).clamp(max=nw - 1)
        c_w = torch.where(c_valid, w[abs_c], torch.zeros_like(w[abs_c]))
        cmax = c_w.amax(dim=-1)                                # (B,L)

        abs_g = (starts.unsqueeze(-1) + g.clamp(min=0)).clamp(max=nw - 1)
        g_w = torch.where(g_valid, w[abs_g], torch.zeros_like(w[abs_g]))
        gmax = g_w.amax(dim=-1)

        score = torch.maximum(cmax, gmax)                      # (B,L)

        if self.scale_to is not None:
            relevant = c_valid.any(-1) | g_valid.any(-1)
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


if __name__ == "__main__":
    # Smoke test: a central-bump profile should concentrate masking on mid-window C/G.
    import numpy as np
    from transformers import AutoTokenizer
    L = 2000
    pos = np.arange(L)
    central = np.exp(-((pos - L / 2) ** 2) / (2 * (L / 10) ** 2))
    tmp = Path("/tmp/_central_profile.npy"); np.save(tmp, central)
    tok = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)
    coll = PositionProfileCollator(tok, tmp, scale_to=0.15)
    seq = "ACGT" * 500
    enc = tok(seq, return_tensors="pt")
    prob = coll._prob_matrix(enc["input_ids"])
    offs = tok(seq, return_offsets_mapping=True)["offset_mapping"]
    mids = [i for i, (s, e) in enumerate(offs) if s != e and abs((s + e) / 2 - L / 2) < 100]
    ends = [i for i, (s, e) in enumerate(offs) if s != e and (e < 100 or s > L - 100)]
    print(f"mean mask prob  mid-window: {prob[0, mids].mean():.3f}   ends: {prob[0, ends].mean():.3f}")
    print("central profile concentrates on mid-window:",
          prob[0, mids].mean() > 3 * prob[0, ends].mean())
