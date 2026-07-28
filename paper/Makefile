.PHONY: all clean

LATEXMK := $(shell command -v latexmk 2>/dev/null)
PDFLATEX ?= pdflatex
BIBTEX ?= bibtex

all:
	@if [ -n "$(LATEXMK)" ]; then \
		latexmk -pdf main.tex; \
	else \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex && \
		$(BIBTEX) main && \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex && \
		$(PDFLATEX) -interaction=nonstopmode -halt-on-error main.tex; \
	fi

clean:
	@if [ -n "$(LATEXMK)" ]; then \
		latexmk -C main.tex; \
	else \
		$(RM) main.aux main.bbl main.blg main.log main.out main.pdf; \
	fi
