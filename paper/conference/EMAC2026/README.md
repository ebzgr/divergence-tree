# EMAC 2026 competitive paper (LaTeX)

Blind submission build for the European Marketing Academy (EMAC) Fall Conference 2026 competitive papers track. See the official PDF guidelines on the conference site for deadlines and track names.

## Files

| File | Purpose |
|------|---------|
| [`EMAC2026.tex`](EMAC2026.tex) | Original blind submission draft (A4, 2.5 cm margins, 1.5 spacing, Times 12/14). **Title matches** [`../../main.tex`](../../main.tex) (`Unblackbox: Understanding Tradeoffs in Personalized Marketing Interventions`). |
| [`EMAC2026_2.tex`](EMAC2026_2.tex) | Rewritten blind submission draft derived from [`../../main.tex`](../../main.tex) — restores joint splitting criterion equation, model selection, and a regulatory use-cases section, while staying inside the 10-page cap. |
| [`abstract_emac.tex`](abstract_emac.tex) | Defines `\EMACAbstractText` (must stay **≤ 1000 characters** including spaces). Shared by both `EMAC2026.tex` and `EMAC2026_2.tex`. |
| [`abstract_plain.txt`](abstract_plain.txt) | Plain-text copy of the abstract for quick `wc -m` checks. |

### Why the EMAC abstract differs from `main.tex`

EMAC’s competitive-paper rules cap the **page-1 abstract at 1{,}000 characters** (including spaces and punctuation). The abstract in `main.tex` is roughly **1{,}900 characters**, so it **cannot** be pasted verbatim. The text in `abstract_emac.tex` is a **shortened version of the same three ideas** (tradeoffs and who-vs-why gap; descriptive joint segmentation; simulations and implications). If you tighten or lengthen it, always re-check the character limit.

## Compile (from this directory)

```bash
pdflatex EMAC2026
bibtex EMAC2026
pdflatex EMAC2026
pdflatex EMAC2026
```

For the rewritten version, swap `EMAC2026` with `EMAC2026_2` in each step.

Bibliography uses [`../../ref.bib`](../../ref.bib). Tables and figures use paths under `../../tables/` and `../../figures/figurefiles/`.

## Abstract character limit

```bash
wc -m abstract_plain.txt
```

Must print **1000 or less** before submission.

## Blind review checklist

- No author names, affiliations, acknowledgments, or grants in the PDF.
- Set PDF metadata safely: `hyperref` uses empty `pdfauthor`; keep `pdftitle` non-identifying or generic.
- Before upload, verify file properties (Word/LibreOffice users see EMAC PDF; for LaTeX, re-check with your PDF viewer or `exiftool` if you touch metadata manually).

## Figures

Factor plots are copied from the v4 analyzer output when you refresh results:

`outputs/simulations/Comprehensive_simulation_v4/aggregated/v4_lambda_twostep_comparison/analysis/plots/`

into `../../figures/figurefiles/` (see repo `ai-doc/06_paper_workflow.md`).

## Track name

The title page uses **Marketing Analytics and Research Method** as the intended track (italic). Replace with the **exact** label from the EMAC 2026 website if it differs.

## Page limit

Competitive papers: **10 pages maximum** (title page, abstract, body, references, tables, figures). After compiling, check:

```bash
pdfinfo EMAC2026.pdf | grep Pages
```

This build is sized to approach the **10-page** cap while staying compliant; after edits, check `pdfinfo` (recent builds land near **9 pages** with two tables and three figures).
