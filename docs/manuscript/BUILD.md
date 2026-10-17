# Manuscript builds

`sentinel_gl_submission.tex` is the complete eight-page double-column submission draft. Its 22 cited records are in `references_typeset.bib`; all five vector figures are in `../figures/2023/`. An alternative ICT-CEEL layout and three Word drafts are included. Their presence does not assert venue acceptance or final submission compliance.

With a TeX installation providing the referenced packages, run from `docs/manuscript`:

```sh
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error sentinel_gl_submission.tex
bibtex sentinel_gl_submission
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error sentinel_gl_submission.tex
pdflatex -no-shell-escape -interaction=nonstopmode -halt-on-error sentinel_gl_submission.tex
```

The ICT-CEEL source has its own formatted reference list and can be built directly with two `pdflatex` passes. Compiled PDFs and Word documents are copied unchanged from the locally prepared artifacts. No institution or venue endorsement is implied.

`tools/prepare_2023_manuscript.py` renders figures from recorded analysis plus actual ROI arrays and attention tensors. It does not acquire data or train a model. Those bulk inputs are not bundled, so running the renderer requires staging them first. Numerical tables and final figures are included for inspection without regeneration.
