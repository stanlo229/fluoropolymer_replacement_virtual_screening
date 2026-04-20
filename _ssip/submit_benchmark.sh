#!/bin/bash
#SBATCH --job-name=ssip_benchmark
#SBATCH --account=aip-aspuru
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=./logs/benchmark.out
#SBATCH --error=./logs/benchmark.err

# ============================================================
# SSIP benchmark: PFAS + aliphatic carbon chain molecules
#
# Purpose: validate that the ML-MEPS SSIP pipeline produces
# physically sensible α/β values before running the full
# norbornene virtual screen.
#
# Molecule series:
#   1. Alkanes (n-hexane, octane, hexadecane)     → expect α≈0, β≈0
#   2. Aliphatic donors/acceptors (MeOH, BuOH,   → known Hunter α/β
#      OctOH, AcOH, HexAcid)
#   3. Perfluoroalkanes (C6F14, C8F18)            → expect α≈0, β≈0
#   4. PFAS acids & sulfonics (PFBA, PFOA,        → COOH donor preserved;
#      PFBS, PFOS, GenX)                             β suppressed by CF2
#
# Usage:
#   sbatch submit_benchmark.sh                    # fresh run
#   sbatch submit_benchmark.sh --resume           # resume interrupted run
#   WORKDIR=runs/benchmark_02 sbatch submit_benchmark.sh
# ============================================================

# ---- Configurable paths ------------------------------------------------
REPO_DIR="/home/stanlo/scratch/Repos/fluoropolymer_replacement_virtual_screening"
CODE_DIR="${REPO_DIR}/_ssip"
WORKDIR="${WORKDIR:-${REPO_DIR}/runs/benchmark_01}"
DEVICE="cuda"

EXTRA_ARGS="$@"

# ---- Environment -------------------------------------------------------
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

# ---- Setup -------------------------------------------------------------
mkdir -p "${WORKDIR}" "${REPO_DIR}/_ssip/logs"

echo "============================================================"
echo "SSIP benchmark started: $(date)"
echo "Node:    $(hostname)"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Workdir: ${WORKDIR}"
echo "============================================================"

# ---- Write benchmark library.csv into workdir --------------------------
# Bypasses generate_library.py (catalogue-based) so we can run any molecule
# directly. Downstream scripts only need 'smiles' and 'mol_id'.
python - <<PYEOF
import csv, os

WORKDIR = os.environ.get("WORKDIR", "${WORKDIR}")

molecules = [
    # --- Series 1: Alkanes (expect α≈0, β≈0) ---
    ("CCCCCC",             "bench_hexane",       "alkane"),
    ("CCCCCCCC",           "bench_octane",        "alkane"),
    ("CCCCCCCCCCCCCCCC",   "bench_hexadecane",    "alkane"),

    # --- Series 2: Aliphatic donors/acceptors (calibration-overlap compounds) ---
    ("CO",                 "bench_methanol",      "alcohol"),
    ("CCCCO",              "bench_butanol",       "alcohol"),
    ("CCCCCCCCO",          "bench_octanol",       "alcohol"),
    ("CC(=O)O",            "bench_acetic_acid",   "carboxylic_acid"),
    ("CCCCCC(=O)O",        "bench_hexanoic_acid", "carboxylic_acid"),

    # --- Series 3: Perfluoroalkanes (expect α≈0, β≈0) ---
    # C6F14
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",                             "bench_pfhexane",  "perfluoroalkane"),
    # C8F18
    ("FC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",              "bench_pfoctane",  "perfluoroalkane"),

    # --- Series 4: PFAS acids & sulfonics ---
    # PFBA  C4: CF3(CF2)2COOH
    ("OC(=O)C(F)(F)C(F)(F)C(F)(F)F",                                              "bench_pfba",  "perfluoro_acid"),
    # PFOA  C8: CF3(CF2)6COOH
    ("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",                "bench_pfoa",  "perfluoro_acid"),
    # PFBS  C4: CF3(CF2)3SO3H
    ("OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",                                  "bench_pfbs",  "perfluoro_sulfonic"),
    # PFOS  C8: CF3(CF2)7SO3H
    ("OS(=O)(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",    "bench_pfos",  "perfluoro_sulfonic"),
    # GenX (HFPO-DA): CF3CF2OCF2COOH
    ("OC(=O)C(F)(F)OC(F)(F)C(F)(F)F",                                             "bench_genx",  "perfluoro_ether_acid"),
]

out_path = os.path.join(WORKDIR, "library.csv")
fieldnames = ["smiles", "mol_id", "compound_type",
              "scaffold_isomer", "sidechain_id", "sidechain_smiles",
              "source", "link", "molecular_weight_sidechain", "price_per_gram"]

with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for smiles, mol_id, ctype in molecules:
        writer.writerow({
            "smiles": smiles,
            "mol_id": mol_id,
            "compound_type": ctype,
            "scaffold_isomer": "",
            "sidechain_id": "",
            "sidechain_smiles": "",
            "source": "benchmark",
            "link": "",
            "molecular_weight_sidechain": "",
            "price_per_gram": "",
        })

print(f"Wrote {len(molecules)} benchmark molecules to {out_path}")
PYEOF

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to write benchmark library.csv" >&2
    exit 1
fi

# ---- Run pipeline (skip generate; run geom→charges→surfaces→calibrate→footprint) -
python "${CODE_DIR}/run_pipeline.py" \
    --workdir  "${WORKDIR}" \
    --device   "${DEVICE}"  \
    --skip     generate     \
    --resume                \
    ${EXTRA_ARGS}

PIPELINE_STATUS=$?

echo "============================================================"
echo "Pipeline finished: $(date)  (exit ${PIPELINE_STATUS})"

# ---- Analyze results ---------------------------------------------------
if [ -f "${WORKDIR}/ssip_results.csv" ]; then
    echo ""
    echo "--- Benchmark analysis ---"
    python "${CODE_DIR}/analyze_benchmark.py" \
        --results "${WORKDIR}/ssip_results.csv"
    ANALYZE_STATUS=$?
else
    echo "WARNING: ssip_results.csv not found — pipeline may have failed"
    ANALYZE_STATUS=1
fi

echo "============================================================"
echo "Done: $(date)"
echo "Results dir: ${WORKDIR}"
echo "============================================================"

exit $(( PIPELINE_STATUS != 0 ? PIPELINE_STATUS : ANALYZE_STATUS ))
