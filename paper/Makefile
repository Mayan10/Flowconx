# Build the paper. Tables and figures come from results/ via
# `make paper` at the repository root -- never edit paper/tables/*.tex.
#
#   make        build main.pdf
#   make clean  remove build artifacts
#
# Requires a TeX distribution with the acmart class.

MAIN := main

.PHONY: all clean assets

all: $(MAIN).pdf

# Regenerate every table and figure from results/ before building, so the PDF
# can never be newer than the numbers it reports.
assets:
	$(MAKE) -C .. paper

$(MAIN).pdf: assets $(MAIN).tex sections/*.tex references.bib
	pdflatex -interaction=nonstopmode $(MAIN).tex
	bibtex $(MAIN) || true
	pdflatex -interaction=nonstopmode $(MAIN).tex
	pdflatex -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log $(MAIN).out $(MAIN).pdf
