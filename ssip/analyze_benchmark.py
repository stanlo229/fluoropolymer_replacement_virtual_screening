"""
analyze_benchmark.py
--------------------
Compare SSIP pipeline predictions against expected Hunter-scale α/β values
for the PFAS + aliphatic benchmark set.

Usage:
    python analyze_benchmark.py --results runs/benchmark_01/ssip_results.csv
"""

import argparse
import csv
import sys

# ── Expected values (Hunter α/β scale) ───────────────────────────────────────
# Sources:
#   Methanol, ethanol, water: Calero et al. PCCP 2013 (calibration set)
#   Alcohols, acids: Driver & Hunter PCCP 2020 Table S1
#   PFAS: qualitative expectations from perfluorination H-bond literature
#   NaN = no strong expectation (direction-only test)
NaN = float("nan")

EXPECTED = {
    # mol_id             : (alpha_exp, beta_exp, note)
    "bench_hexane"       : (0.0,  0.0,  "alkane baseline"),
    "bench_octane"       : (0.0,  0.0,  "alkane baseline"),
    "bench_hexadecane"   : (0.0,  0.0,  "alkane baseline"),
    "bench_methanol"     : (2.12, 2.19, "Hunter calibration reference"),
    "bench_butanol"      : (2.12, 2.40, "aliphatic alcohol"),
    "bench_octanol"      : (2.12, 2.40, "aliphatic alcohol — chain-length insensitive"),
    "bench_acetic_acid"  : (2.00, 1.80, "carboxylic acid donor + carbonyl acceptor"),
    "bench_hexanoic_acid": (2.00, 1.70, "carboxylic acid — chain-length insensitive"),
    "bench_pfhexane"     : (0.0,  0.0,  "perfluoroalkane — C-F is not H-bond acceptor"),
    "bench_pfoctane"     : (0.0,  0.0,  "perfluoroalkane — C-F is not H-bond acceptor"),
    "bench_pfba"         : (2.00, NaN,  "COOH donor preserved; β < acetic acid"),
    "bench_pfoa"         : (2.00, NaN,  "COOH donor preserved; β < hexanoic acid"),
    "bench_pfbs"         : (NaN,  0.0,  "sulfonic acid — stronger donor than PFBA; β≈0"),
    "bench_pfos"         : (NaN,  0.0,  "sulfonic acid — stronger donor than PFBA; β≈0"),
    "bench_genx"         : (2.00, NaN,  "COOH donor; β slightly > PFOA (ether O)"),
}

# ── Pass/fail criteria ────────────────────────────────────────────────────────
TOL_CALIBRATION = 0.30   # methanol must match within ±0.30
TOL_GENERAL     = 0.50   # other absolute comparisons
TOL_ZERO        = 0.15   # zero-baseline threshold


def _fmt(v):
    return f"{v:6.3f}" if v == v else "   NaN"  # NaN check


def _pass_fail(pred, exp, tol, direction=None):
    """Return (symbol, reason) given prediction, expectation, tolerance.
    direction: '<' means pred must be < exp (not checked against tol)
    """
    import math
    if math.isnan(exp):
        return "  —  ", ""
    if direction == "<":
        ok = pred < exp
        return ("PASS" if ok else "FAIL"), (f"must be < {exp:.2f}, got {pred:.3f}")
    if direction == ">":
        ok = pred > exp
        return ("PASS" if ok else "FAIL"), (f"must be > {exp:.2f}, got {pred:.3f}")
    ok = abs(pred - exp) <= tol
    return ("PASS" if ok else "FAIL"), (f"|{pred:.3f} - {exp:.2f}| = {abs(pred-exp):.3f}, tol={tol}")


def load_results(path):
    results = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mol_id = row["mol_id"]
            results[mol_id] = {
                "alpha_max": float(row.get("alpha_max", 0) or 0),
                "beta_max":  float(row.get("beta_max",  0) or 0),
                "N_ssip":    int(float(row.get("N_ssip", 0) or 0)),
                "AvdW":      float(row.get("AvdW", 0) or 0),
            }
    return results


def run_checks(results):
    """Run pairwise relational checks between molecules."""
    checks = []

    def _get(mol_id, key):
        return results.get(mol_id, {}).get(key, float("nan"))

    # COOH donor parity: PFBA α ≈ acetic acid α (±0.50)
    pfba_a  = _get("bench_pfba",       "alpha_max")
    acoh_a  = _get("bench_acetic_acid","alpha_max")
    if pfba_a == pfba_a and acoh_a == acoh_a:
        diff = abs(pfba_a - acoh_a)
        ok = diff <= 0.50
        checks.append(("COOH donor parity", "bench_pfba α ≈ bench_acetic_acid α",
                        "PASS" if ok else "FAIL",
                        f"|{pfba_a:.3f} - {acoh_a:.3f}| = {diff:.3f}, tol=0.50"))

    # CF2 β suppression: β(PFBA) < β(acetic acid)
    pfba_b  = _get("bench_pfba",        "beta_max")
    acoh_b  = _get("bench_acetic_acid", "beta_max")
    if pfba_b == pfba_b and acoh_b == acoh_b:
        ok = pfba_b < acoh_b
        checks.append(("CF2 β suppression (C4)", "bench_pfba β < bench_acetic_acid β",
                        "PASS" if ok else "FAIL",
                        f"{pfba_b:.3f} < {acoh_b:.3f}"))

    # CF2 β suppression: β(PFOA) < β(hexanoic acid)
    pfoa_b  = _get("bench_pfoa",         "beta_max")
    hexac_b = _get("bench_hexanoic_acid","beta_max")
    if pfoa_b == pfoa_b and hexac_b == hexac_b:
        ok = pfoa_b < hexac_b
        checks.append(("CF2 β suppression (C8)", "bench_pfoa β < bench_hexanoic_acid β",
                        "PASS" if ok else "FAIL",
                        f"{pfoa_b:.3f} < {hexac_b:.3f}"))

    # Sulfonic > carboxylic α
    pfbs_a = _get("bench_pfbs", "alpha_max")
    pfba_a = _get("bench_pfba", "alpha_max")
    if pfbs_a == pfbs_a and pfba_a == pfba_a:
        ok = pfbs_a > pfba_a
        checks.append(("Sulfonic > carboxylic α", "bench_pfbs α > bench_pfba α",
                        "PASS" if ok else "FAIL",
                        f"{pfbs_a:.3f} > {pfba_a:.3f}"))

    # GenX ether β > PFOA β
    genx_b = _get("bench_genx", "beta_max")
    pfoa_b = _get("bench_pfoa", "beta_max")
    if genx_b == genx_b and pfoa_b == pfoa_b:
        ok = genx_b > pfoa_b
        checks.append(("GenX ether β > PFOA β", "bench_genx β > bench_pfoa β",
                        "PASS" if ok else "FAIL",
                        f"{genx_b:.3f} > {pfoa_b:.3f}"))

    # Chain-length invariance for alkanes (α and β)
    for key in ("alpha_max", "beta_max"):
        vals = [_get(m, key) for m in ("bench_hexane", "bench_octane", "bench_hexadecane")]
        if all(v == v for v in vals):
            spread = max(vals) - min(vals)
            ok = spread < 0.10
            checks.append((f"Alkane chain-length ({key})",
                            "max−min across hexane/octane/hexadecane < 0.10",
                            "PASS" if ok else "FAIL",
                            f"spread = {spread:.3f}"))

    # Chain-length invariance for perfluoroalkanes
    for key in ("alpha_max", "beta_max"):
        vals = [_get(m, key) for m in ("bench_pfhexane", "bench_pfoctane")]
        if all(v == v for v in vals):
            spread = abs(vals[0] - vals[1])
            ok = spread < 0.10
            checks.append((f"PFalkane chain-length ({key})",
                            "|pfhexane − pfoctane| < 0.10",
                            "PASS" if ok else "FAIL",
                            f"spread = {spread:.3f}"))

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="ssip_results.csv",
                        help="Path to ssip_results.csv from run_pipeline.py")
    args = parser.parse_args()

    try:
        results = load_results(args.results)
    except FileNotFoundError:
        print(f"ERROR: {args.results} not found", file=sys.stderr)
        sys.exit(1)

    # ── Per-molecule table ────────────────────────────────────────────────────
    W = 20
    print()
    print("=" * 90)
    print("SSIP BENCHMARK — Per-molecule predictions vs. expected (Hunter scale)")
    print("=" * 90)
    hdr = (f"{'mol_id':<24} {'α_pred':>7} {'α_exp':>7} {'α_ok':>5}  "
           f"{'β_pred':>7} {'β_exp':>7} {'β_ok':>5}  {'note'}")
    print(hdr)
    print("-" * 90)

    per_mol_failures = 0
    for mol_id, (a_exp, b_exp, note) in EXPECTED.items():
        r = results.get(mol_id)
        if r is None:
            print(f"  {mol_id:<22}  MISSING — not in results")
            per_mol_failures += 1
            continue

        a_pred = r["alpha_max"]
        b_pred = r["beta_max"]

        # Choose tolerance
        tol_a = TOL_CALIBRATION if "methanol" in mol_id else TOL_ZERO if a_exp == 0.0 else TOL_GENERAL
        tol_b = TOL_CALIBRATION if "methanol" in mol_id else TOL_ZERO if b_exp == 0.0 else TOL_GENERAL

        a_sym, _ = _pass_fail(a_pred, a_exp, tol_a)
        b_sym, _ = _pass_fail(b_pred, b_exp, tol_b)

        import math
        a_fail = (a_sym == "FAIL")
        b_fail = (b_sym == "FAIL")
        if a_fail or b_fail:
            per_mol_failures += 1

        print(f"  {mol_id:<22}  {_fmt(a_pred)} {_fmt(a_exp)} {a_sym:>5}   "
              f"{_fmt(b_pred)} {_fmt(b_exp)} {b_sym:>5}   {note}")

    print("-" * 90)

    # ── Relational / pairwise checks ─────────────────────────────────────────
    print()
    print("=" * 90)
    print("SSIP BENCHMARK — Relational checks")
    print("=" * 90)
    rel_hdr = f"  {'Test':<30} {'Condition':<45} {'Result':>5}  Detail"
    print(rel_hdr)
    print("-" * 90)

    checks = run_checks(results)
    rel_failures = sum(1 for _, _, sym, _ in checks if sym == "FAIL")
    for test, cond, sym, detail in checks:
        print(f"  {test:<30} {cond:<45} {sym:>5}  {detail}")

    print("-" * 90)

    # ── Summary ──────────────────────────────────────────────────────────────
    total_failures = per_mol_failures + rel_failures
    print()
    print(f"  Per-molecule failures : {per_mol_failures}")
    print(f"  Relational failures   : {rel_failures}")
    print(f"  Total failures        : {total_failures}")
    print()
    if total_failures == 0:
        print("  OVERALL: PASS — pipeline produces physically sensible SSIP values")
    else:
        print("  OVERALL: FAIL — check failures above; consider GFN2-xTB MEPS fallback")
    print("=" * 90)
    print()

    sys.exit(0 if total_failures == 0 else 1)


if __name__ == "__main__":
    main()
