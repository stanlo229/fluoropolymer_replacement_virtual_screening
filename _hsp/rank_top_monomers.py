"""
rank_top_monomers.py

Produce three independent top-N rankings of norbornene monomers under a single
set of hard constraints:

  1. Most PTFE-like              -> lowest  Ra from PTFE
  2. Most water-repellent        -> highest Ra from Water
  3. Most diiodomethane-repellent-> highest Ra from Diiodomethane

Constraints applied, in order:

  C1  Catalogue filter (same rules as dataset/filter_catalogues.py, applied to
      the sidechain): drop chiral centres, phenols (OH on aromatic C) and
      anilines (N on aromatic C).

  C2  Physical HSP: any negative delta_D/P/H is a group-contribution failure
      -> drop. Si monomers are NOT exempted. Clamping their negative
      components to 0.0 was tested and rejected: PTFE sits at
      (12.7, 0, 0), so a clamped monomer lands exactly on two of its three
      axes and Ra_PTFE collapses to 2*|delta_D - 12.7|. Clamped rows were
      1.8% of the library but took 25/25 PTFE and 10/10 water slots — they
      won by construction, not by chemistry. Roughly 12 Si monomers survive
      this rule on their own merits.

  C3  Physical plausibility envelope: delta_D in [12, 24], delta_P <= 30,
      delta_H <= 45, the range spanned by real liquids in Hansen (2007)
      Appendix A. Without it the repellency rankings are headed by
      divergences — delta_H = 72.8 (water itself is 42.3), delta_P = 63.8,
      delta_D = 33.9. Excluded rows are written to
      excluded_outside_envelope.csv rather than dropped silently.

  C3b NOTE on amides: Table A.1 has no secondary-amide group. Two invented
      entries ("CONH", "CON") previously supplied one; they are removed, and
      such amides now decompose into the published catch-all groups
      ">C=O (except as above)" + "NH/N (except as above)", matching how the
      paper's own 2-pyrrolidone example is handled. Every amide monomer's HSP
      therefore CHANGED at this revision. n_unavail_{d,p,hb} record how many
      matched groups carry a "***" (no published value) contribution.

  C4  Relaxed group coverage: every atom of an element the Stefanis-Panayiotou
      tables actually cover (C,H,N,O,S,F,Cl,Br,I) must be matched. Atoms of
      elements with no table entry (Si,Se,B,P,Ge,Te,As,...) are exempt and
      recorded in `incomplete_elements`, so Si and Se monomers survive instead
      of being silently deleted by a strict n_unmatched == 0 rule.

  C5  Deduplication: stereo-stripped canonical SMILES, then HSP triplet
      (group contribution is blind to stereochemistry and to some structural
      isomerism, so those rows are not independent predictions).

Outputs, for each of the three rankings:
    top<N>_ranked_<name>.csv
    top<N>_ranked_<name>_grid.png
plus:
    monomers_filtered_ranked.csv   (all survivors, every Ra/RED column)
    filter_report.json             (attrition at each constraint)

Usage:
    python rank_top_monomers.py [--input PATH] [--out_dir PATH] [--top_n 25]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from solvent_incompatibility import (
    REFERENCES, R0_VALUES, compute_ra_columns, make_grid, _solvent_key,
)

RDLogger.DisableLog("rdApp.*")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# C1 — catalogue filter SMARTS (mirrors dataset/filter_catalogues.py)
# ---------------------------------------------------------------------------
_PHENOL = Chem.MolFromSmarts("[OX2H][c]")
_AROMATIC_AMINE = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(NS(=O));!$([N+]);!$(N=*)][c]"
)

# ---------------------------------------------------------------------------
# C4 — elements with at least one first-order group in group_tables.py.
# Anything outside this set contributes nothing to the Stefanis-Panayiotou sum
# and is therefore exempt from the coverage requirement (but flagged).
# ---------------------------------------------------------------------------
COVERED_ELEMENTS = {"C", "H", "N", "O", "S", "F", "Cl", "Br", "I"}

# ---------------------------------------------------------------------------
# Residual basic nitrogen: an sp3 N still protonatable after the linkage has
# formed (the sidechain amine that became the amide is no longer basic, but a
# second ring N — N-methylpiperidine, benzylpiperazine — is untouched).
# Aliphatic amines have pKaH ~9-10, so at neutral pH these are largely cationic.
# Hansen theory has no acid-base term and models them as neutral organics, so
# any Ra involving water is unreliable for these rows.
# ---------------------------------------------------------------------------
_BASIC_N = Chem.MolFromSmarts(
    "[NX3;!$(NC=O);!$(NS(=O));!$(N=*);!$([N+]);!$(Nc);!$(N#*)]"
)

# ---------------------------------------------------------------------------
# C3 — physical plausibility envelope, from the range spanned by real liquids
# in Hansen (2007) Appendix A. No organic liquid sits outside these bounds;
# values beyond them are group-contribution divergences.
# ---------------------------------------------------------------------------
ENVELOPE = {
    "delta_D": (12.0, 24.0),
    "delta_P": (0.0, 30.0),
    "delta_H": (0.0, 45.0),
}

# ---------------------------------------------------------------------------
# The method's own average absolute error on its fit set, Table A.4 (2012),
# second-order approximation, over 347-350 experimental compounds.
# Used to report whether a ranking is actually resolved: if the top-N spread is
# smaller than the AAE, the ordering is noise, not chemistry.
# ---------------------------------------------------------------------------
METHOD_AAE = {"delta_D_corr": 0.33, "delta_P_corr": 0.82, "delta_H_corr": 0.77}

# ---------------------------------------------------------------------------
# Dominance margin: how far one Hansen component exceeds the other two
# combined, in MPa^0.5 so the score reads directly.
#
#     m_D = delta_D - (delta_P + delta_H)     etc.
#
# This answers "max X, minimal everything else" in its literal sense: it
# rewards a large component AND small others. The alternative convention, Teas
# fractional parameters f = delta_X / sum(delta), scores purity irrespective of
# magnitude; the two agree closely for polar and H-bonding but pick very
# different dispersion sets, because delta_D has a floor near 12 for any
# organic while delta_P and delta_H can approach zero.
#
# The margin is a linear combination of all three measured components, so its
# uncertainty propagates as the root-sum-square of their individual AAEs:
#     sqrt(0.33^2 + 0.82^2 + 0.77^2) = 1.172 MPa^0.5
# ---------------------------------------------------------------------------
MARGIN_COLS = {
    "margin_dispersion": ("delta_D_corr", "delta_P_corr", "delta_H_corr"),
    "margin_polar":      ("delta_P_corr", "delta_D_corr", "delta_H_corr"),
    "margin_hbonding":   ("delta_H_corr", "delta_D_corr", "delta_P_corr"),
}
MARGIN_AAE = 1.172


# ---------------------------------------------------------------------------
# Cached per-SMILES property helpers
# ---------------------------------------------------------------------------
def _sidechain_flags(smiles: str) -> tuple[bool, bool, bool, bool]:
    """(parsed_ok, has_chiral, has_phenol, has_aniline) for a sidechain SMILES."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return (False, False, False, False)
    return (
        True,
        bool(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
        mol.HasSubstructMatch(_PHENOL),
        mol.HasSubstructMatch(_AROMATIC_AMINE),
    )


def _uncovered_elements(smiles: str) -> tuple[int, str]:
    """(n_atoms_of_uncovered_elements, comma-joined sorted element symbols)."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return (0, "")
    syms = [a.GetSymbol() for a in mol.GetAtoms()
            if a.GetSymbol() not in COVERED_ELEMENTS]
    return (len(syms), ",".join(sorted(set(syms))))


def _n_basic_nitrogens(smiles: str) -> int:
    mol = Chem.MolFromSmiles(str(smiles))
    return len(mol.GetSubstructMatches(_BASIC_N)) if mol else 0


def _has_isotope(smiles: str) -> bool:
    """True if any atom carries an isotope label (e.g. 13C-labelled reagents)."""
    mol = Chem.MolFromSmiles(str(smiles))
    return bool(mol) and any(a.GetIsotope() for a in mol.GetAtoms())


def _canon_no_stereo(smiles: str) -> str:
    mol = Chem.MolFromSmiles(str(smiles))
    return Chem.MolToSmiles(mol, isomericSmiles=False) if mol else str(smiles)


def _map_unique(series: pd.Series, fn):
    """Apply fn once per unique value, then broadcast back (SMILES parsing is slow)."""
    uniq = pd.Index(series.astype(str).unique())
    lookup = {v: fn(v) for v in uniq}
    return series.astype(str).map(lookup)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="results/monomers_hsp_corrected.csv")
    parser.add_argument("--out_dir", default="results")
    parser.add_argument("--catalogues", default="../dataset/catalogues.csv")
    parser.add_argument("--top_n", type=int, default=25)
    parser.add_argument("--ncols", type=int, default=5)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_col, p_col, h_col = "delta_D_corr", "delta_P_corr", "delta_H_corr"
    report: dict = {"constraints": []}

    def _step(name: str, df_before: pd.DataFrame, df_after: pd.DataFrame, detail: str = ""):
        rec = {
            "constraint": name,
            "before": len(df_before),
            "after": len(df_after),
            "removed": len(df_before) - len(df_after),
            "detail": detail,
        }
        report["constraints"].append(rec)
        log.info("%-34s %7d -> %7d  (-%d) %s",
                 name, rec["before"], rec["after"], rec["removed"], detail)

    # ---- load -------------------------------------------------------------
    df = pd.read_csv(args.input)
    n_loaded = len(df)
    log.info("Loaded %d monomers from %s", n_loaded, args.input)

    for c in (d_col, p_col, h_col):
        if c not in df.columns:
            raise SystemExit(f"Missing column {c} in {args.input} — run si_correction first.")

    df0 = df
    df = df.dropna(subset=[d_col, p_col, h_col]).copy()
    _step("C0 HSP present", df0, df)

    # ---- C1 catalogue filter ---------------------------------------------
    log.info("Evaluating C1 catalogue filter on unique sidechains…")
    flags = _map_unique(df["sidechain_smiles"], _sidechain_flags)
    df["_ok"] = [f[0] for f in flags]
    df["_chiral"] = [f[1] for f in flags]
    df["_phenol"] = [f[2] for f in flags]
    df["_aniline"] = [f[3] for f in flags]

    n_chiral = int(df["_chiral"].sum())
    n_phenol = int(df["_phenol"].sum())
    n_aniline = int(df["_aniline"].sum())

    before = df
    df = df[df["_ok"] & ~df["_chiral"] & ~df["_phenol"] & ~df["_aniline"]].copy()
    df = df.drop(columns=["_ok", "_chiral", "_phenol", "_aniline"])
    _step("C1 chiral/phenol/aniline", before, df,
          f"chiral={n_chiral} phenol={n_phenol} aniline={n_aniline} (overlapping)")

    # ---- C2 drop unphysical (no clamping, Si included) -------------------
    # A negative Hansen component has no physical meaning. Si monomers are NOT
    # exempted: the 3-reference empirical correction drives most of them
    # negative, and clamping those to 0.0 was tested and rejected — it places
    # them exactly on PTFE's delta_P = delta_H = 0 axes, so they win the
    # PTFE ranking by construction rather than by chemistry.
    si_mask = df.get("has_si", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    before = df
    neg_mask = (df[[d_col, p_col, h_col]] < 0).any(axis=1)
    n_si_dropped = int((neg_mask & si_mask).sum())
    df = df[~neg_mask].copy()
    _step("C2 unphysical delta < 0", before, df,
          f"of which Si monomers={n_si_dropped}")

    # ---- C3 physical plausibility envelope -------------------------------
    # Bounds spanned by real liquids in Hansen (2007) Appendix A. Values outside
    # these are group-contribution divergences, not chemistry: the library
    # otherwise reaches delta_H = 72.8 (water itself is 42.3) and delta_D = 33.9.
    before = df
    env = (
        df[d_col].between(ENVELOPE["delta_D"][0], ENVELOPE["delta_D"][1])
        & (df[p_col] <= ENVELOPE["delta_P"][1])
        & (df[h_col] <= ENVELOPE["delta_H"][1])
    )
    df_excluded = df[~env].copy()
    df = df[env].copy()
    _step("C3 physical envelope", before, df,
          f"dD {ENVELOPE['delta_D']}, dP<={ENVELOPE['delta_P'][1]}, "
          f"dH<={ENVELOPE['delta_H'][1]}")

    # ---- C4 relaxed coverage ---------------------------------------------
    log.info("Evaluating C4 relaxed coverage on unique monomers…")
    unc = _map_unique(df["monomer_smiles"], _uncovered_elements)
    df["n_uncovered_atoms"] = [u[0] for u in unc]
    df["incomplete_elements"] = [u[1] for u in unc]

    before = df
    n_strict = int((df["n_unmatched_atoms"] == 0).sum())
    df = df[df["n_unmatched_atoms"] <= df["n_uncovered_atoms"]].copy()
    _step("C4 relaxed group coverage", before, df,
          f"strict n_unmatched==0 would have kept {n_strict}")

    # ---- C5 dedup ---------------------------------------------------------
    # The dedup key (isomericSmiles=False) strips isotope labels as well as
    # stereochemistry, so a 13C-labelled reagent and its unlabelled twin collapse
    # to one row. Their HSP is bit-identical (group contribution ignores mass),
    # so prefer the unlabelled member as the representative — same numbers, but
    # an ordinary purchasable compound rather than an isotope-labelled one.
    before = df
    df["_iso"] = _map_unique(df["monomer_smiles"], _has_isotope)
    df = df.sort_values("_iso", kind="stable")
    n_iso = int(df["_iso"].sum())

    df["_canon"] = _map_unique(df["monomer_smiles"], _canon_no_stereo)
    df = df.drop_duplicates(subset=["_canon"]).drop(columns=["_canon"]).copy()
    after_stereo = len(df)

    df["_hsp_key"] = (
        df[d_col].round(4).astype(str) + "|"
        + df[p_col].round(4).astype(str) + "|"
        + df[h_col].round(4).astype(str)
    )
    df = df.drop_duplicates(subset=["_hsp_key"]).drop(columns=["_hsp_key"]).copy()
    n_iso_kept = int(df["_iso"].sum())
    df = df.drop(columns=["_iso"])
    _step("C5 dedup (stereo + HSP triplet)", before, df,
          f"stereo-dedup intermediate={after_stereo}; "
          f"isotope-labelled {n_iso} -> {n_iso_kept} after preferring unlabelled twins")

    if df.empty:
        raise SystemExit("No monomers survived the constraints.")

    # ---- purchase links ---------------------------------------------------
    cat_path = Path(args.catalogues)
    if cat_path.exists():
        cat = (pd.read_csv(cat_path, usecols=["canonical_smiles", "link", "source"])
                 .drop_duplicates(subset=["canonical_smiles"]))
        cat = cat.rename(columns={"source": "cat_source"})
        df = df.merge(cat, left_on="sidechain_smiles",
                      right_on="canonical_smiles", how="left")
        df = df.drop(columns=["canonical_smiles"])
        log.info("Purchase links joined for %d / %d monomers",
                 int(df["link"].notna().sum()), len(df))
    else:
        df["link"] = pd.NA
        log.warning("Catalogues not found at %s — link column empty", cat_path)

    # ---- Ra / RED ---------------------------------------------------------
    df, ra_cols, red_cols = compute_ra_columns(df, d_col, p_col, h_col)

    # ---- dominance margins -------------------------------------------------
    for mcol, (main, o1, o2) in MARGIN_COLS.items():
        df[mcol] = df[main] - (df[o1] + df[o2])
        METHOD_AAE[mcol] = MARGIN_AAE

    # ---- residual basic nitrogen (acid-base blind spot) --------------------
    df["n_basic_N"] = _map_unique(df["monomer_smiles"], _n_basic_nitrogens)
    n_basic_rows = int((df["n_basic_N"] > 0).sum())
    log.info("Residual basic N present in %d / %d monomers (%.1f%%)",
             n_basic_rows, len(df), 100 * n_basic_rows / len(df))

    # ---- audit badge ------------------------------------------------------
    def _badge(row) -> str:
        parts = []
        inc = str(row.get("incomplete_elements", "") or "")
        if inc:
            parts.append(f"no groups: {inc}")
        nb = int(row.get("n_basic_N", 0) or 0)
        if nb:
            parts.append(f"basic N x{nb}")
        return "  ".join(parts)

    df["qc_flag"] = df.apply(_badge, axis=1)

    # ---- rows removed by the envelope, kept for audit ---------------------
    if len(df_excluded):
        exc_path = out_dir / "excluded_outside_envelope.csv"
        df_excluded.sort_values(d_col, ascending=False).to_csv(exc_path, index=False)
        log.info("Envelope-excluded rows (%d) -> %s", len(df_excluded), exc_path)
        report["excluded_outside_envelope"] = str(exc_path)

    # ---- full survivor CSV ------------------------------------------------
    full_csv = out_dir / "monomers_filtered_ranked.csv"
    df.sort_values("Ra_PTFE").to_csv(full_csv, index=False)
    log.info("Full filtered set (%d monomers) -> %s", len(df), full_csv)

    # ---- three independent rankings --------------------------------------
    rankings = [
        ("ptfe", "Ra_PTFE", True,
         "Most PTFE-like  (lowest Ra from PTFE)"),
        ("water", "Ra_Water", False,
         "Most water-repellent  (highest Ra from Water)"),
        ("diiodomethane", "Ra_Diiodomethane", False,
         "Most diiodomethane-repellent  (highest Ra from Diiodomethane)"),
        # Individual Hansen components. Ranked over the SAME C1-C5 survivors as
        # everything else, unlike make_hansen_grids.py which ranks the raw
        # unfiltered table. Note these maximise a quantity the C3 envelope caps
        # from above, so check the resolution line in filter_report.json.
        ("max_dispersion", d_col, False,
         "Max dispersion  (highest delta_D)"),
        ("max_polar", p_col, False,
         "Max polarity  (highest delta_P)"),
        ("max_hbonding", h_col, False,
         "Max H-bonding  (highest delta_H)"),
        # Dominance margins: max one component, minimise the other two.
        ("dominant_dispersion", "margin_dispersion", False,
         "Dispersion-dominant  (max delta_D - [delta_P + delta_H])"),
        ("dominant_polar", "margin_polar", False,
         "Polarity-dominant  (max delta_P - [delta_D + delta_H])"),
        ("dominant_hbonding", "margin_hbonding", False,
         "H-bonding-dominant  (max delta_H - [delta_D + delta_P])"),
    ]

    report["rankings"] = {}
    csv_cols = (
        ["sidechain_smiles", "monomer_smiles", "linkage", "source", "link",
         "molecular_weight", d_col, p_col, h_col]
        + ra_cols + red_cols
        + list(MARGIN_COLS)
        + ["n_basic_N", "incomplete_elements", "n_unmatched_atoms",
           "n_unavail_d", "n_unavail_p", "n_unavail_hb",
           "n_uncovered_atoms", "has_si", "n_si_atoms"]
    )

    for safe_name, rank_col, ascending, label in rankings:
        df_top = (df.sort_values(rank_col, ascending=ascending)
                    .head(args.top_n).reset_index(drop=True))

        # Is this ranking resolved, or is it a tie inside the method's error?
        spread = float(df_top[rank_col].max() - df_top[rank_col].min())
        aae = METHOD_AAE.get(rank_col)
        resolution = None
        if aae is not None:
            best = float(df[rank_col].max() if not ascending else df[rank_col].min())
            n_tied = int((df[rank_col] >= best - aae).sum() if not ascending
                         else (df[rank_col] <= best + aae).sum())
            resolved = spread > aae
            resolution = {
                "top_n_spread": round(spread, 4),
                "method_AAE": aae,
                "resolved": resolved,
                "n_within_AAE_of_best": n_tied,
            }
            if not resolved:
                log.warning(
                    "%s: top-%d spans %.3f MPa^0.5 but the method's own AAE is "
                    "%.2f — %d monomers lie within one AAE of the best. This "
                    "ordering is inside the noise floor.",
                    safe_name, args.top_n, spread, aae, n_tied,
                )

        cols = [c for c in csv_cols if c in df_top.columns]
        csv_out = out_dir / f"top{args.top_n}_ranked_{safe_name}.csv"
        df_top[cols].to_csv(csv_out, index=False)

        is_component = rank_col in METHOD_AAE or rank_col in MARGIN_COLS
        sub = f"{len(df):,} monomers after constraints"
        if resolution is not None:
            sub += (
                f"  |  top-{args.top_n} spread {resolution['top_n_spread']:.3f} vs "
                f"method AAE {resolution['method_AAE']:.2f} MPa½"
                + ("" if resolution["resolved"] else
                   f"  —  NOT RESOLVED: {resolution['n_within_AAE_of_best']:,} "
                   f"monomers tie within one AAE")
            )
        title = (
            f"Top {args.top_n} Norbornene Monomers — {label}\n"
            f"{'' if is_component else '★ = ranking criterion  |  '}"
            f"all Ra in MPa½  |  {sub}"
        )
        grid_out = out_dir / f"top{args.top_n}_ranked_{safe_name}_grid.png"
        make_grid(df_top, ra_cols, d_col, p_col, h_col, grid_out,
                  title=title, rank_col=rank_col, ncols=args.ncols,
                  red_cols=red_cols, flag_col="qc_flag",
                  rank_label=("δD" if rank_col == d_col else
                              "δP" if rank_col == p_col else
                              "δH" if rank_col == h_col else
                              "δD margin" if rank_col == "margin_dispersion" else
                              "δP margin" if rank_col == "margin_polar" else
                              "δH margin" if rank_col == "margin_hbonding" else "Ra"))

        report["rankings"][safe_name] = {
            "n_basic_N_rows": int((df_top["n_basic_N"] > 0).sum()),
            "rank_col": rank_col,
            "ascending": ascending,
            f"{rank_col}_range": [float(df_top[rank_col].min()),
                                  float(df_top[rank_col].max())],
            "n_si": int(df_top["has_si"].sum()),
            "n_with_uncovered_elements": int((df_top["incomplete_elements"] != "").sum()),
            "csv": str(csv_out),
            "grid": str(grid_out),
            "resolution": resolution,
        }

        log.info("%-14s top%d: %s %.2f–%.2f  -> %s",
                 safe_name, args.top_n, rank_col,
                 df_top[rank_col].min(), df_top[rank_col].max(), grid_out.name)

    # ---- supplementary: water ranking with acid-base blind spot removed ---
    # Hansen has no acid-base term, so a monomer carrying a residual basic N is
    # modelled as a neutral organic when in water it is largely a cation. Those
    # rows dominate the plain water ranking, so emit a companion ranking with
    # them removed. Neither list is "the" answer: the plain one is what HSP
    # says, this one is what HSP can actually be trusted to say.
    df_nb = df[df["n_basic_N"] == 0]
    if len(df_nb) >= args.top_n:
        df_top = (df_nb.sort_values("Ra_Water", ascending=False)
                       .head(args.top_n).reset_index(drop=True))
        cols = [c for c in csv_cols if c in df_top.columns]
        csv_out = out_dir / f"top{args.top_n}_ranked_water_no_basic_N.csv"
        df_top[cols].to_csv(csv_out, index=False)

        title = (
            f"Top {args.top_n} Norbornene Monomers — Most water-repellent, "
            f"residual basic N excluded\n"
            f"★ = ranking criterion  |  all Ra in MPa½  |  "
            f"{len(df_nb):,} of {len(df):,} monomers carry no protonatable N  |  "
            f"Hansen theory has no acid-base term"
        )
        grid_out = out_dir / f"top{args.top_n}_ranked_water_no_basic_N_grid.png"
        make_grid(df_top, ra_cols, d_col, p_col, h_col, grid_out,
                  title=title, rank_col="Ra_Water", ncols=args.ncols,
                  red_cols=red_cols, flag_col="qc_flag")
        report["rankings"]["water_no_basic_N"] = {
            "rank_col": "Ra_Water",
            "ascending": False,
            "pool": len(df_nb),
            "Ra_Water_range": [float(df_top.Ra_Water.min()), float(df_top.Ra_Water.max())],
            "csv": str(csv_out), "grid": str(grid_out),
        }
        log.info("water(no basic N) top%d: Ra_Water %.2f–%.2f  -> %s",
                 args.top_n, df_top.Ra_Water.min(), df_top.Ra_Water.max(), grid_out.name)

    # ---- report -----------------------------------------------------------
    report["n_loaded"] = n_loaded
    report["n_final"] = len(df)
    report["covered_elements"] = sorted(COVERED_ELEMENTS)
    report["envelope"] = ENVELOPE
    report["n_si_surviving"] = int(df["has_si"].sum())
    report["n_with_residual_basic_N"] = n_basic_rows
    report["references"] = {k: list(v) for k, v in REFERENCES.items()}
    report["R0"] = R0_VALUES
    rep_path = out_dir / "filter_report.json"
    rep_path.write_text(json.dumps(report, indent=2))
    log.info("Filter report -> %s", rep_path)


if __name__ == "__main__":
    main()
