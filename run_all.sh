#!/bin/bash
# Rebuild every number, table and chart from the four raw xlsx files in data/
set -e
cd "$(dirname "$0")"
python3 -W ignore 01_clean.py
python3 -W ignore 02_inference.py
python3 -W ignore 03_models.py
python3 -W ignore 04_portfolio.py
python3 -W ignore 05_figs.py
python3 -W ignore 06_rule_check.py
# The workflow diagram needs the Mermaid CLI; skip it if that isn't installed.
if command -v mmdc >/dev/null 2>&1; then
  mmdc -i figs/workflow.mmd -o figs/00_workflow.png -w 2400 -b '#fcfcfb' -s 2
else
  echo "mmdc not found: keeping the existing figs/00_workflow.png"
fi
