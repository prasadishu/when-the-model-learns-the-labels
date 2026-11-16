#!/usr/bin/env bash
set -euo pipefail
export PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/01_staging_and_cohort.py
python scripts/02_benchmark.py
python scripts/03_audit_and_secondary.py
python scripts/04_deep_learning.py
python scripts/05_make_figures.py
