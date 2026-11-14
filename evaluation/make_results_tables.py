"""Regenerate the Phase-4 authoritative result tables from the canonical result JSONs.

Single reproducible source for every manuscript number, so nothing is hand-typed into the
paper. Reads only the frozen offset-aware multiseed outputs (+ scaling_results.json for the
random/zero-shot references and the first-base reconciliation). Writes
evaluation/phase4_tables.md and prints the same.

Run from project root:
    python evaluation/make_results_tables.py
"""
import json
from pathlib import Path

EV = Path("evaluation")


def load(name):
    return json.loads((EV / name).read_text())


def p(x):                      # compact p-value, 1-decimal mantissa
    return f"{x:.1e}"


def main():
    L = []
    def line(s=""):
        L.append(s)

    # ---- Table 1: three-backbone architecture independence (deltas) ----
    backbones = [
        ("DNABERT-2 (Zhihan / BPE / MosaicBERT)", "results_dnabert_offset_multiseed.json"),
        ("GENA-LM (AIRI / BPE / pre-LN BERT)",    "results_genalm_multiseed.json"),
        ("NT (InstaDeep / 6-mer / ESM)",          "results_nt_multiseed.json"),
    ]
    line("## Table 1 — Architecture independence: pooled Δ(DAM−MLM), one offset-aware collator, seeds 42/1/2")
    line()
    line("| backbone (lab / tokenizer / arch) | term Δ@3 (p, 95% CI) | term Δ@5 (p) | term Δ@10 (p) | bg Δ@3 (p) | bg Δ@5 (p) | bg Δ@10 (p) |")
    line("|---|---|---|---|---|---|---|")

    def cell(block, w, ci=False):
        e = block[str(w)]
        s = f"{e['delta']:+.3f} ({p(e['p_t'])}"
        if ci:
            s += f", [{e['ci95'][0]:+.3f}, {e['ci95'][1]:+.3f}]"
        return s + ")"

    for name, f in backbones:
        d = load(f)
        t, b = d["terminal"], d["background"]
        line(f"| {name} | {cell(t, 3, ci=True)} | {cell(t, 5)} | {cell(t, 10)} "
             f"| {cell(b, 3)} | {cell(b, 5)} | {cell(b, 10)} |")
    # n per column is shared across backbones (same eval windows)
    dn = load(backbones[0][1])
    ns_t = "/".join(str(dn["terminal"][w]["n"]) for w in ("3", "5", "10"))
    ns_b = "/".join(str(dn["background"][w]["n"]) for w in ("3", "5", "10"))
    line()
    line(f"n (terminal 3/5/10) = {ns_t}; n (background 3/5/10) = {ns_b}. Every seed same sign. "
         "Absolute magnitudes are NOT comparable across backbones (different tokenizers) — report the Δ.")
    line()

    # ---- Table 2: DNABERT-2 absolute terminal reconciliation ----
    sc = load("scaling_results.json")             # first-base + random/zero-shot refs
    tr = load("results_tradeoff_multiseed.json")  # offset-aware mlm + dam(p1), same models
    line("## Table 2 — DNABERT-2 terminal recovery (absolute): first-base (retired) vs offset-aware (canonical)")
    line()
    line("| T_END | MLM | DAM first-base | DAM offset-aware | random | zero-shot | DAM_offset − random |")
    line("|---|---|---|---|---|---|---|")
    for w in ("3", "5", "10"):
        e = sc[w]
        mlm_off = tr["mlm"]["terminal"][w]
        dam_off = tr["p1"]["terminal"][w]
        rnd = e["rnd_mean"]
        line(f"| {w} | {mlm_off:.3f} | {e['dam_mean']:.3f} | {dam_off:.3f} "
             f"| {rnd:.3f} | {e['zs_mean']:.3f} | {dam_off - rnd:+.3f} |")
    line()
    line("MLM is far below random at the terminus (the problem DAM addresses). Offset-aware DAM@3 ≈ random, "
         "clears it at T_END≥5. The Δ(DAM−MLM) is overwhelming at every T_END (Table 1).")
    line()

    # ---- Table 3: D3 corruption-shape / domain independence ----
    sy = load("results_synthetic_multiseed.json")
    ff = load("results_ffpe_multiseed.json")
    rows = [("central (synthetic, mid-window)", sy["central"]),
            ("periodic (synthetic, 50 bp)",     sy["periodic"]),
            ("FFPE (REAL clinical, near-uniform)", ff["ffpe"])]
    line("## Table 3 — D3 corruption-shape / domain independence: pooled Δ(DAM-matched − uniform MLM) in the corruption zone")
    line()
    line("| profile | zone n | per-seed Δ (42/1/2) | pooled Δ | p | 95% CI |")
    line("|---|---|---|---|---|---|")
    for name, e in rows:
        ps = "/".join(f"{x:+.3f}" for x in e["per_seed_delta"])
        line(f"| {name} | {e['zone_positions']} | {ps} | {e['delta']:+.4f} "
             f"| {p(e['p_t'])} | [{e['ci95'][0]:+.4f}, {e['ci95'][1]:+.4f}] |")
    line()
    line("Two synthetic shapes (deliberately un-terminal) + one REAL clinical domain (FFPE, near-uniform). "
         "DAM matched to each beats uniform MLM at recovering its damaged positions → the principle is "
         "corruption-shape- and domain-agnostic, not ends-are-special.")
    line()

    # ---- Table 4: masking-regime reference (pmd-mask) ----
    line("## Table 4 — Masking-regime reference (pmd-mask = reference point, NOT rival): recovery by zone")
    line()
    line("| arm | t@3 | t@5 | t@10 | bg@3 | bg@5 | bg@10 |")
    line("|---|---|---|---|---|---|---|")
    for key, disp in [("mlm", "MLM"), ("p1", "DAM (offset, power 1)"), ("p2", "DAM power 2"),
                      ("p4", "DAM power 4"), ("pmd", "pmd-mask")]:
        if key not in tr:
            continue
        t, b = tr[key]["terminal"], tr[key]["background"]
        line(f"| {disp} | {t['3']:.3f} | {t['5']:.3f} | {t['10']:.3f} "
             f"| {b['3']:.3f} | {b['5']:.3f} | {b['10']:.3f} |")
    line()
    line("Power knob inert (p1≈p2≈p4); pmd-mask has a large terminal edge and NO background collapse "
         "under this protocol → a domain-specific terminal-only reference, not a rival.")

    out = "\n".join(L) + "\n"
    (EV / "phase4_tables.md").write_text(out)
    print(out)
    print(f"[written] {EV / 'phase4_tables.md'}")


if __name__ == "__main__":
    main()
