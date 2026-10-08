#!/bin/sh
# Build the IEEE paper and its supplementary appendices. Each reads the other's labels (xr), which latexmk does not
# track, so the second pass forces a rerun (-g).
set -e
cd "$(dirname "$0")"
python3 ../scripts/ieee_bib.py   # the bibliographies with IEEE's abbreviated venue names
latexmk -pdf -interaction=nonstopmode ieee-supplement.tex >/dev/null 2>&1 || true
latexmk -pdf -interaction=nonstopmode ieee-paper.tex >/dev/null 2>&1 || true
latexmk -g -pdf -interaction=nonstopmode ieee-supplement.tex >/dev/null 2>&1 || true
latexmk -g -pdf -interaction=nonstopmode ieee-paper.tex >/dev/null 2>&1 || true
for f in ieee-paper ieee-supplement; do
  echo "$f: $(grep -c '^! ' $f.log) errors, $(grep -c '^Overfull' $f.log) overfull, $(grep -c 'Reference.*undefined' $f.log) undefined refs, $(grep -c 'multiply defined' $f.log) multiply defined (the other document's citation labels, which xr also imports; each document's own win), $(pdfinfo $f.pdf | grep Pages)"
done
