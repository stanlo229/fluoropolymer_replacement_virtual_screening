# Archive

Superseded code, kept for provenance. Nothing in the working pipeline imports
from here, and paths inside these files are not maintained (they still refer
to the pre-2026-09-30 layout: `_hsp/`, flat `results/`, `analysis_2b/`).

| File | Why it was retired |
|---|---|
| `si_correction.py` | Per-Si-atom offset on top of a Si→C substitution, fitted to 2 compounds; drove most Si monomers negative. `run_pipeline.py` now just copies the S-P values into the `*_corr` columns. Si is handled by the added Si groups of the v3 formula. |
| `validate_hsp.py` | Expected HSP columns in the free `HSPiPDataSet.xls`, which has only names and CAS numbers, so it never ran. Replaced by `analysis/build_reference.py` + the CV scripts on the licensed HSPiP workbook. |
| `make_hansen_grids.py` | Produced the unconstrained top-50 max-δD/δP/δH grids. The top-50 outputs were dropped on 2026-09-30; `rank_top_monomers.py` produces the constrained top-25 lists. |
| `rerun_all_v2.sh` | Ran the v2 recalculation on the login node. Replaced by `v2/submit_v2_rerun.sh` (itself archived with v2). |
| `sp_fixes.py`, `eval_fixes.py`, `fix_eval_reference.csv` | Validation of the four second-order identification fixes from the 2012 worked examples. The fixes are now in `hsp_calculator.py` (verified identical). |
| `assemble_final.py`, `assemble.log`, `check.log` | An unfinished assembly of refractivity-δD + ENS + si_gc. Superseded by `v2/finalize_hsp.py` and then by `analysis/apply_gc_formula.py` (v3). |

## `v2/` — HSPiP-trained ExtraTrees model (superseded by v3 on 2026-10-01)
v2 predicted HSP with a tree ensemble on fingerprints, descriptors and S-P values
(`predict_monomers.py`, `finalize_hsp.py`, `model_cv.py`, `expected_error.py`,
`predict_smiles.py`), with a separate ridge fit for Si (`si_gc.py`), a size-intensive
refractivity δD cross-check (`dispersion*.py`, `branched_check.py`) and a determinism
test (`check_determinism.py`). It was deterministic but not a formula, and in
scaffold-grouped CV it was less accurate than the v3 group-contribution formula
(median Ra error 3.96 vs 2.83 on the curated set). Its outputs remain in `results/v2/`.
The job scripts (`submit_v2_rerun.sh`, `submit_hsp6.sh`, `submit_check_determinism.sh`)
are here too; their paths are not maintained.
