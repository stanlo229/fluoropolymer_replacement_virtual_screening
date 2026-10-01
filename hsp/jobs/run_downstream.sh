#!/bin/bash
# Rankings and Hansen-space plots for one result version:
#     VER=v3 METHOD_AAE=D,P,H jobs/run_downstream.sh
# reads results/$VER/monomers_hsp_corrected.csv and writes into results/$VER/.
# Called from the submit_* job scripts; runs on the node, not the login node.
#
# METHOD_AAE = held-out CV MAE (D,P,H) of the HSP source on the curated HSPiP
# master set, used to judge whether a ranking is resolved. Default: the S-P
# paper's in-sample Table A.4 values.
#
# The unconstrained top-50 grids (solvent_incompatibility.py CLI,
# make_hansen_grids.py) are no longer produced; the constrained top25_ranked_*
# lists from rank_top_monomers.py replace them.
set -euo pipefail
cd "$(dirname "$0")/.."
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate
export VER=${VER:-v3}
IN=results/$VER/monomers_hsp_corrected.csv
OUT=results/$VER
CAT=../dataset/catalogues.csv
AAE_ARG=()
[ -n "${METHOD_AAE:-}" ] && AAE_ARG=(--method_aae "$METHOD_AAE")

echo "== Hansen-space plots ($VER) =="
python - <<'EOF'
import os
import pandas as pd
from visualize_hsp import visualize
v = os.environ["VER"]
df = pd.read_csv(f"results/{v}/monomers_hsp_corrected.csv", low_memory=False).dropna(subset=["delta_D_corr"])
print(visualize(df, out_dir=f"results/{v}"))
EOF

echo "== Constrained top-25 rankings ($VER) =="
python rank_top_monomers.py --input "$IN" --out_dir "$OUT" --catalogues "$CAT" \
    --top_n 25 --ncols 5 "${AAE_ARG[@]}"
