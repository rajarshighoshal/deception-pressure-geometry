# TMLR manuscript

`main.tex` is the anonymous TMLR submission source. It reads the five public figures directly from
`docs/figures/`; figures are not duplicated under `paper/`.

Build from this directory with:

```bash
latexmk -pdf main.tex
```

`make` uses `latexmk` when available and otherwise runs `pdflatex`, `bibtex`, and the required
final `pdflatex` passes directly. The official unmodified `tmlr.sty` and `tmlr.bst` are bundled
beside `main.tex`, so the source package does not depend on private files or local paths.

The submitted PDF must remain anonymous. The public repository or author identity should not be
linked from the double-blind submission; prepare a separate de-anonymized preprint only when
desired.

## Public source closure

This repository tracks the manuscript source, bibliography, TMLR style files, and the five source
figures under `docs/figures/`. The rendered PDF, LaTeX intermediates, and submission archives are
build products and are intentionally not tracked. The verified source currently renders to 19
pages; the main argument through the conclusion ends on page 12, with reproducibility, broader
impact, and references beginning on page 13.
