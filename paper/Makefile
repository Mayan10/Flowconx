# Build the paper. Tables and figures come from results/ via
# `make paper` at the repository root -- never edit paper/tables/*.tex.
#
#   make        build main.pdf
#   make clean  remove build artifacts
#
# Two engines are supported, because neither is universally installed:
#
#   tectonic  self-contained, downloads the acmart class and fonts on first
#             run, and drives bibtex itself. This is what the paper is
#             actually built with; `brew install tectonic` is the whole setup.
#   pdflatex  the classic route, for a machine that already has a full TeX
#             distribution including acmart.
#
# tectonic is preferred when both are present. Override with `make ENGINE=pdflatex`.

MAIN := main

ENGINE := $(shell command -v tectonic >/dev/null 2>&1 && echo tectonic || echo pdflatex)

.PHONY: all clean assets engine-check

all: $(MAIN).pdf

# Regenerate every table and figure from results/ before building, so the PDF
# can never be newer than the numbers it reports.
assets:
	$(MAKE) -C .. paper

engine-check:
	@command -v $(ENGINE) >/dev/null 2>&1 || { \
	  echo "error: neither tectonic nor pdflatex is on PATH."; \
	  echo "       install one:  brew install tectonic"; \
	  exit 1; }

$(MAIN).pdf: assets engine-check $(MAIN).tex sections/*.tex references.bib
ifeq ($(ENGINE),tectonic)
	tectonic -X compile $(MAIN).tex --outdir . --keep-logs
else
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex
endif

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf $(MAIN).xdv
