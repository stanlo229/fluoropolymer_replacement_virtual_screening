#!/bin/bash
#SBATCH --job-name=hsp_v3_formula
#SBATCH --account=aip-aspuru
#SBATCH --partition=gpubase_l40s_b1
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/hsp_v3_formula_%j.out
# v3: refitted group-contribution formula (size-intensive Fedors form).
#   bench  analysis/gc_benchmark.py     CV + fit group tables
#   apply  analysis/apply_gc_formula.py library + six monomers
#   down   rankings/plots in results/v3 (non-Si) and results/v3/si (Si only)
#   si     analysis/si_reliability.py   evidence for/against the Si values
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
[ -f hsp_calculator.py ] || { echo "submit from hsp/: sbatch jobs/submit_v3_formula.sh"; exit 1; }
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
FROM=${FROM:-bench}; run=0
f() { grep --line-buffered -v "single bond\|BondStereo" || true; }
for s in bench apply down si; do
  [ "$s" = "$FROM" ] && run=1; [ $run = 1 ] || { echo "skip $s"; continue; }
  echo "== $s ($(date +%T)) =="
  case $s in
    bench) python -u analysis/gc_benchmark.py 2>&1 | f > results/benchmark/reference/gc_benchmark_log.txt ;;
    apply) python -u analysis/apply_gc_formula.py 2>&1 | f | tee results/v3/apply_log.txt ;;
    down)  AAE=$(python -c "
import pandas as pd; s=pd.read_csv('results/benchmark/reference/gc_benchmark_summary.csv')
b=s[(s.tier==1)&(s.subset=='all')&s.model.str.startswith('Fedors')].sort_values('med_Ra_err').iloc[0]
print(f'{b.MAE_D},{b.MAE_P},{b.MAE_H}')" 2>/dev/null | tail -1)
           echo "method AAE $AAE"
           VER=v3 METHOD_AAE=$AAE jobs/run_downstream.sh 2>&1 | f > results/v3/downstream_log.txt
           # Si monomers ranked on their own (values may not be reliable)
           VER=v3/si METHOD_AAE=$AAE \
             TITLE_NOTE="Si MONOMERS ONLY: HSP may not be reliable (see results/v3/si/README.md)" \
             jobs/run_downstream.sh 2>&1 | f > results/v3/si/downstream_log.txt
           # dendron monomers ranked on their own as well
           if [ -s results/v3/dendrons/monomers_hsp_corrected.csv ] && \
              [ "$(wc -l < results/v3/dendrons/monomers_hsp_corrected.csv)" -gt 1 ]; then
             VER=v3/dendrons METHOD_AAE=$AAE TITLE_NOTE="DENDRON MONOMERS (focal-point attached)" \
               jobs/run_downstream.sh 2>&1 | f > results/v3/dendrons/downstream_log.txt
           fi ;;
    si)    python -u analysis/si_reliability.py 2>&1 | f > results/benchmark/reference/si_reliability_log.txt ;;
  esac
done
echo "ALL DONE ($(date +%T))"
