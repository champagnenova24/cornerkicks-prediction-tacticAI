"""Create `results/`, `figures/` and `models/`. Imported for the side effect.

The reorganised layout, which `check_paths.py` enforces statically:

    results/   every .csv a script writes
    figures/   every .png
    models/    every .pt
    (root)     .parquet — derived data, deliberately left in place

All three directories are committed with content, so they exist in any
checkout; this module only covers the case where one was deleted, or a new
result type is written before the directory is created. Missing them raises
`FileNotFoundError` at write time, which is loud rather than silent — but it
happens *after* the work, which for the 45-minute diffusion runs is expensive.

Kept out of `paths.py` on purpose: importing that module resolves the
StatsBomb checkout and calls `SystemExit` if it is absent, which the
torch-only scripts (`diffusion_corners.py` and everything downstream of it)
do not otherwise need.

Imported directly by the scripts that define their own `SP` and write results.
The scripts that instead do `from diffusion_corners import SP` already take
their path root from that module, so they inherit the directories from it too.
"""
import os

SP = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SP, 'results')
FIGURES = os.path.join(SP, 'figures')
MODELS = os.path.join(SP, 'models')

for _d in (RESULTS, FIGURES, MODELS):
    os.makedirs(_d, exist_ok=True)
