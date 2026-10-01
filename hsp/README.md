# hsp/ — Hansen solubility parameters for the norbornene monomer library

Predicts δD, δP and δH for ~79k norbornene ester/amide monomers built from catalogue
alcohols and amines, and ranks them by Hansen distance (Ra) to PTFE, water,
diiodomethane and n-hexadecane.

## Methods (three versions, all kept for comparison)

| Version | HSP source | Where | Status |
|---|---|---|---|
| `legacy_sp` | Published Stefanis–Panayiotou 2012 group contribution (`hsp_calculator.py`) | `results/legacy_sp/` | Baseline. Goes negative for polycyclics (17% of monomers) and drifts with size. |
| `v2` | ExtraTrees model trained on HSPiP (code in `archive/v2/`) | `results/v2/` | Superseded. Deterministic but not a formula. Si not scored. |
| **`v3` (adopted)** | **Refitted group-contribution formula**, size-intensive form δk = √(Σ Nᵢ eₖ,ᵢ / Σ Nᵢ vᵢ), using the S-P groups plus added Si / new-atom groups, with values fitted to HSPiP (`analysis/gc_refit.py`, `gc_benchmark.py`, `apply_gc_formula.py`) | `results/v3/` | **Current method.** Most accurate in CV (median Ra error 2.83 on the curated set vs 3.34 for published S-P). |

> **Si monomers are ranked separately, in `results/v3/si/`, because their values may not be
> reliable.** The main rankings in `results/v3/` contain no Si monomers. `results/v3/si/README.md`
> sets out why the Si values may and may not be trusted; the full analysis is §10 of REPORT.md
> (`analysis/si_reliability.py`). Every Si row carries an `hsp_note`, and the Si grids are titled
> "Si MONOMERS ONLY".

Accuracy, the literature comparison and every caveat are in
`results/benchmark/reference/REPORT.md` (gitignored, because it quotes HSPiP values) and
`docs/literature_ml_hsp.md`. The legacy method is described in `METHODS.md`.

## Layout

```
hsp/
  hsp_calculator.py, group_tables.py   S-P calculator + published tables (with the 2012 worked-example fixes)
  generate_monomers.py, run_pipeline.py  monomer library + legacy S-P pipeline
  rank_top_monomers.py                   constrained top-25 rankings (C1-C5 filters)
  solvent_incompatibility.py             probe-liquid HSP, Ra / RED helpers (imported)
  visualize_hsp.py                       Hansen-space plots
  analysis/                              reference building, CV benchmarks, v3 formula
  jobs/                                  SLURM scripts (submit from hsp/)
  archive/                               retired code, with reasons (archive/README.md)
  docs/                                  literature review
  viz/                                   interactive 3D explorer (template + build script)
  reference/                             licensed HSPiP workbook + papers   (gitignored)
  results/legacy_sp/  v2/  v3/           outputs per method version (v3 = non-Si rankings)
  results/v3/si/                         Si-monomer rankings, kept separate (may not be reliable)
  results/benchmark/                     CV predictions, fitted group tables, report   (gitignored)
  results/manual/2026-09-30/             the six requested monomers (hsp6*.csv)
  logs/                                  SLURM logs
```

## Running

Everything heavier than a quick check runs as a SLURM job. Submit from `hsp/`:

```bash
cd hsp
sbatch jobs/submit_hsp.sh                                # legacy: library + S-P + rankings
sbatch jobs/submit_v3_formula.sh                         # v3: CV + fit formula, apply, rank (non-Si + Si), Si analysis
sbatch --export=ALL,FROM=apply jobs/submit_v3_formula.sh # v3 from a later stage (bench|apply|down|si)
sbatch jobs/submit_gc_benchmark.sh                       # formula vs XGBoost vs ExtraTrees CV only
python analysis/predict_smiles.py "SMILES" ...           # v3 HSP for any SMILES (seconds; fitted tables needed)
```

The cluster has only GPU partitions; these CPU jobs use `gpubase_l40s_b1` with no `--gres`.
`analysis/build_reference.py` must be run once after adding a new HSPiP workbook to `reference/`.

## Interactive explorer

`results/v3/hsp_explorer.html` is a single self-contained file (about 5 MB, works offline) with a
3D plot of all 79,314 monomers in δD/δP/δH space. Hover a point to see its structure and HSP values;
click to pin it. You can colour by Ra to each probe liquid or by any component, highlight any top-25
list, search SMILES fragments, and toggle the Si monomers (amber diamonds, flagged as possibly
unreliable). To share it, send the file; it opens in any recent browser. A hosted copy lives on
claude.ai. Rebuild after a new v3 run with `python viz/build_explorer.py` (from `hsp/`).

## Licensed data

`reference/` holds the HSPiP workbook and paywalled papers. `results/benchmark/` holds files that
contain HSPiP values. Both are gitignored. The `results/v2` and `v3` tables contain predictions
only. The v2 tables also name each monomer's nearest HSPiP compound, but carry none of its values.

Large per-monomer tables that the jobs regenerate are not tracked in git: all of v2's, and the
v3 duplicates (`monomers_hsp_corrected.csv`, `hsp_Ra_ranked.csv`, `monomers_hsp_solvent_Ra.csv`).
`results/v3/monomers_hsp_formula.csv` is the tracked full v3 table.
