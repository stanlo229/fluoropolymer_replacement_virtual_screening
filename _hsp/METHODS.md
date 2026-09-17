# HSP screening pipeline — how the numbers are produced

Every step from the supplier catalogue to the ranked figures, with the source
for each constant and the known failure modes. Written 2026-09-17, after the
group tables were verified line by line against the primary source.

**Primary source.** Stefanis, E.; Panayiotou, C. *A new expanded solubility
parameter approach.* Int. J. Pharm. **426** (2012) 29–43, **Appendix A**
(doi:10.1016/j.ijpharm.2012.01.001). This supersedes the group tables in
Stefanis & Panayiotou, Int. J. Thermophys. **29** (2008) 568–585.

---

## 1. Monomer generation — `generate_monomers.py`

Input `dataset/catalogues.csv` (89,622 commercial compounds scraped from seven
suppliers). Each sidechain is attached **twice** to *exo*-norbornene-2,3-dicarboxylic
acid, `OC(=O)[C@H]1[C@@H](C(=O)O)[C@@H]2C=C[C@@H]1C2`, giving a symmetric
difunctional monomer:

| sidechain type | reaction | product |
|---|---|---|
| monoalcohol / diol | esterification | symmetric **diester** |
| monoamine | amidation | symmetric **diamide** |

Mechanically: locate the scaffold `[CX3](=O)[OX2H1]`, delete the hydroxyl O,
bond the carboxyl C to the reagent's O or N, sanitise, repeat on the second
acid. A compound whose declared `compound_type` and actual SMARTS match
disagree is written to `skipped.csv` instead.

Output: **79,314 monomers**.

---

## 2. HSP prediction — `hsp_calculator.py` + `group_tables.py`

### 2.1 Group counting

**First-order groups** (Table A.1, 76 groups) are matched by SMARTS in a fixed
priority order, specific before generic, with **no atom reuse** — each heavy
atom belongs to exactly one group. Atoms left over are counted in
`n_unmatched_atoms`, which is the honest signal that the decomposition is
incomplete.

**Second-order groups** (Table A.2, 37 groups) are identified structurally
after first-order counting. They do not consume atoms; they are corrections.
`W = 1` if any second-order group is present, else `W = 0`.

### 2.2 The equations (Eqs. A.2–A.4)

```
delta_d  = ( sum_i Ni*Ci_d  + W * sum_j Mj*Dj_d  + 959.11 ) ** 0.4126
delta_p  =   sum_i Ni*Ci_p  + W * sum_j Mj*Dj_p  + 7.6134
delta_hb =   sum_i Ni*Ci_hb + W * sum_j Mj*Dj_hb + 7.7003
```

All three constants and the exponent were verified against the paper.

### 2.3 Low-value fallback (Eqs. A.5–A.6)

The paper states Eqs. A.3/A.4 are **valid only above 3 MPa^0.5**. So if a
computed `delta_p` or `delta_hb` falls below 3, it is recomputed from scratch
using the separate low-value contribution tables (A.5 first-order, A.6
second-order) and different constants:

```
delta_p  = sum + W*sum + 2.6560      (when the standard value came out < 3)
delta_hb = sum + W*sum + 1.3720
```

### 2.4 Two deliberate extensions beyond the paper

1. **Negative `delta_d` bracket.** If the Eq. A.2 bracket goes negative, a real
   0.4126 power does not exist. The code computes `|bracket|**0.4126` and
   reapplies the sign. This is how a negative `delta_d` can appear; such rows
   are filtered later (§4, C2).
2. **`***` contributions treated as zero.** Many table cells are "not
   available". Skipping them in the sum silently produces a number that looks
   valid but is missing a term the authors declined to supply.
   `n_unavail_d/p/hb` count these per molecule so such rows can be identified.

   This is not a corner case here. The norbornene C=C always sits next to a
   ring CH, so **every** monomer fires the second-order group `>C{H or C}-C=`,
   whose `delta_p` is `***` in Table A.2. **100 % of the screened library is
   missing a `delta_p` term**, and 42.7 % / 21.2 % are missing a `delta_d` /
   `delta_hb` term (mostly the amide `NH (except as above)`, whose `delta_d`
   is also `***`). Ranking positions are still meaningful because the omission
   is common to every monomer built on this scaffold, but the absolute
   `delta_p` values carry a systematic offset of unknown size.

### 2.5 Silicon — why it is excluded

**Table A.1 contains no silicon group.** The method covers **C, H, O, N, S, F,
Cl, Br, I** and nothing else. Si, Se, B, P, Ge, Te, As have no contribution.

`hsp_calculator.py` substitutes Si→C before matching and `si_correction.py`
then applies a per-Si-atom empirical offset. Both halves fail:

- the substitution turns silyl groups into quaternary carbons, which alone
  leaves 95/307 Si monomers with a negative component (vs 8.4 % for non-Si);
- the offset is fitted to **2** unique compounds, both siloxanes, while only 4
  of 307 Si sidechains are siloxanes (122 silyl ethers, 160 with no Si–O bond);
- its `delta_H` coefficient is an artifact of the substitution — trimethylsilanol
  becomes *tert*-butanol, predicted `delta_H` 14.0 against 4.8 measured, giving
  −9.2 per Si;
- `mean_abs_residual = 2.76 MPa^0.5`: it does not reproduce its own references.

`calibrate_si_correction()` therefore reports `usable: false`, and Si monomers
are removed downstream by the negative-delta filter. Fixing this requires an
HSP source **outside** this method (HSPiP Y-MB, or measured values), not a
larger offset.

---

## 3. Verification performed (2026-09-17)

`group_tables.py` was diffed entry by entry against Appendix A.

**Result: 76/76 first-order and 37/37 second-order groups match exactly; all
seven constants match.** Second-order *identification* was checked against the
paper's own 20 worked examples (§A.4) — 10 tested, 10 reproduce.

Ten defects were found and fixed:

| # | Defect | Fix |
|---|---|---|
| 1 | `CONH` (secondary amide) with **invented** `dp=4.1000, dhb=3.5000` | removed |
| 2 | `CON` (tertiary amide) with **invented** `dp=3.8000, dhb=2.0000` | removed |
| 3–5 | `Cl-(C=C)`, `CF`, `CH2=C=C<` missing from Table A.1 | added |
| 6 | `CcyclicHm=Ncyclic-CcyclicHn=CcyclicHp` missing from Table A.2 | added |
| 7 | `>C<` in Table A.5 — not in A.5; held `>C=C<`'s value | removed |
| 8 | `CH2N` A.5 `dp` = 0.6477 (that is `-CH<`'s value) | → **0.7055** |
| 9 | `ACCOO` A.6 `dp` = 0.4912 — no published value | → `***` |
| 10 | `AC(ACHm)2AC(ACHn)2` A.6 `dp` = 0.0130 | → **0.0669** |

### The amide consequence

Defects 1–2 mattered most: **every diamide monomer in the library was scored
with invented numbers.** Table A.1 has `CONH2` (primary) and `CON(CH3)2` but no
secondary amide.

The published route is visible in the paper's own tables: Table A.2 carries
`NcyclicHm–Ccyclic=O` with **2-pyrrolidone** — a cyclic secondary amide — as
its example. That second-order correction only makes sense if the lactam N–H
was taken by the first-order catch-all `NH (except as above)`. So secondary
amides now decompose as

```
>C=O (except as above)   +   NH (except as above)
  (-127.16, 0.7691, 1.7033)     (***, -0.0746, 2.0646)
```

Amine groups (`CH2NH`, `CH3NH`, `CH2N`, …) were given `!$(NC=O)` exclusions so
they can no longer claim an amide nitrogen — an amide N is not an amine.
**All amide HSP values changed at this revision**, and the effect is large: the
invented `dp = 4.1000` had been propping up amide polarity, so the share of
amide monomers with a negative component rose from a few percent to **26.2 %**
(46,954 amides) against **4.5 %** for esters (32,360). The C2 filter removed
4,414 monomers where it previously removed 2,435, and the final screened
library fell from 7,913 to **7,052**.

### Gaps carried from the paper itself

Table A.5 lists `ACCH<`, `CHNH` and `CCl2F`, but Table A.1 has no first-order
entry for them, so they are unusable — 41 of 44 A.5 rows are live. Adding them
would require inventing the A.1 values, which is the defect just removed.

---

## 4. Screening constraints — `rank_top_monomers.py`

Applied in order; every stage is recorded in `results/filter_report.json`.
Attrition for the current run:

```
                                79,314 monomers generated
C1  chiral / phenol / aniline   79,314 -> 24,191   (-55,123)
C2  negative delta               24,191 -> 19,777   (-4,414, of which 201 Si)
C3  physical envelope            19,777 -> 14,630   (-5,147)
C4  relaxed group coverage       14,630 ->  9,201   (-5,429)
C5  dedup (stereo + triplet)      9,201 ->  7,052   (-2,149)
```


| | Constraint | Rationale |
|---|---|---|
| **C1** | drop chiral centres, phenols (`[OX2H][c]`), anilines | synthetic tractability; mirrors `dataset/filter_catalogues.py`, which existed but had never been run |
| **C2** | drop any negative `delta_d/p/hb` | a negative Hansen component is meaningless. Si is **not** exempt — clamping to 0.0 was tested and rejected: PTFE sits at (12.7, **0, 0**), so a clamped monomer lands on two of its three axes and wins by construction (clamped rows were 1.8 % of the library but took 25/25 PTFE slots) |
| **C3** | `delta_d ∈ [12,24]`, `delta_p ≤ 30`, `delta_h ≤ 45` | range spanned by real liquids in Hansen (2007) App. A. Without it the repellency rankings are headed by divergences — `delta_h` = 72.8 where water itself is 42.3. Excluded rows go to `excluded_outside_envelope.csv`, never dropped silently |
| **C4** | `n_unmatched_atoms ≤ n_atoms_of_uncovered_elements` | relaxed coverage: every atom of a **covered** element must be matched, while Si/Se/B/P/… atoms are exempt and named in `incomplete_elements`. A strict `== 0` rule would silently delete every Se compound |
| **C5** | dedup on stereo-stripped SMILES, then on the (d,p,h) triplet | group contribution is blind to stereochemistry and to some structural isomerism, so those rows are not independent predictions. Ties prefer the **non-isotope-labelled** twin — a ¹³C reagent and its unlabelled form give bit-identical HSP, so the purchasable one is shown |

---

## 5. Ranking and scoring

Hansen distance to a reference, with the conventional factor of 4 on the
dispersion term:

```
Ra  = sqrt( 4*(dD - dD_ref)^2 + (dP - dP_ref)^2 + (dH - dH_ref)^2 )
RED = Ra / R0        (RED < 1 => predicted compatible/soluble)
```

| Reference | dD | dP | dH | R0 |
|---|---|---|---|---|
| PTFE | 12.7 | 0.0 | 0.0 | 7.0 |
| Water | 15.5 | 16.0 | 42.3 | 17.8 |
| Diiodomethane | 17.8 | 3.9 | 5.5 | 7.0 (approx.) |
| n-Hexadecane | 16.3 | 0.0 | 0.0 | 5.0 (approx.) |

Three **independent** top-25 rankings are produced — not one blended score:

1. **Most PTFE-like** — lowest `Ra_PTFE`
2. **Most water-repellent** — highest `Ra_Water`
3. **Most diiodomethane-repellent** — highest `Ra_Diiodomethane`

plus a supplementary **water ranking excluding residual basic N** (§6).
Current results (3,621 ester / 3,431 amide survivors):

| Ranking | Ra range | ester/amide | with basic N |
|---|---|---|---|
| Most PTFE-like | 4.34 – 5.58 | 15 / 10 | 1 |
| Most water-repellent | 45.15 – 46.20 | 15 / 10 | 11 |
| Most diiodomethane-repellent | 31.59 – 43.81 | 12 / 13 | 5 |
| Water, basic N excluded | 44.75 – 46.10 | 7 / 18 | 0 |

### These three objectives genuinely conflict

PTFE (12.7, 0, 0) and diiodomethane (17.8, 3.9, 5.5) are only **Ra = 12.2**
apart, so anything near PTFE is automatically near diiodomethane. Measured on
the screened library, `Spearman(Ra_PTFE, Ra_Diiodo) ≈ +0.94`: pushing away from
diiodomethane *is* pushing away from PTFE. The three top-25 sets share **no
members**. Objective 3 is the opposite of objectives 1 and 2, not a weaker
version of them — for a fluoropolymer replacement, rankings 1 and 2 are the
useful ones and ranking 3 is a diagnostic.

---

## 6. Known limitations

- **No acid–base term.** Hansen is a three-term cohesive-energy model; it
  cannot see that a piperidine or piperazine protonates and hydrates in water.
  Such monomers are 10.8 % of the library but ~44 % of the plain water top-25.
  `n_basic_N` counts residual protonatable nitrogens, a `basic N ×k` badge
  marks them in every figure, and `top25_ranked_water_no_basic_N` re-runs the
  water ranking over monomers with none. Use that one for candidate selection.
- **Monomer, not polymer.** HSP is computed for the difunctional monomer. A
  ROMP polymer of it will differ.
- **`***` contributions counted as zero** (§2.4). Every monomer is missing a
  `delta_p` term; check `n_unavail_*` per row.
- **Norbornene ring approximation.** The bicyclic scaffold is scored with the
  `Ring of 5 carbons` second-order group; `approx_ring_correction` flags this
  on every row.
- **Method accuracy ceiling.** Even on the paper's own fit set (347–350
  compounds) the average absolute errors are `dD` 0.33, `dP` 0.82, `dH` 0.77
  MPa^0.5, with R² of 0.936 / 0.855 / 0.940. Differences between neighbouring
  ranked monomers are far smaller than that — treat the ranking as a coarse
  screen, not a decision procedure.

---

## 7. Reproducing

```bash
module load StdEnv/2023 python/3.11 rdkit/2024.03
source ~/projects/aip-aspuru/stanlo/.virtualenvs/ocsr/bin/activate

cd _hsp
python run_pipeline.py --catalogues ../dataset/catalogues.csv --out_dir ./results/
python solvent_incompatibility.py --input ./results/monomers_hsp_corrected.csv \
       --out_dir ./results/ --catalogues ../dataset/catalogues.csv --top_n 50
python rank_top_monomers.py --top_n 25 --out_dir results
```

| Output | Contents |
|---|---|
| `monomers.csv` / `skipped.csv` | generated monomers; rejected inputs with reasons |
| `monomers_hsp.csv` | raw group-contribution HSP |
| `monomers_hsp_corrected.csv` | after the (unusable) Si correction |
| `monomers_filtered_ranked.csv` | survivors of C1–C5 with all Ra/RED columns |
| `excluded_outside_envelope.csv` | rows C3 removed, for audit |
| `filter_report.json` | attrition at every constraint |
| `top25_ranked_*.csv` / `*_grid.png` | the four rankings and their figures |

Note `.gitignore` excludes `*.json`, so `filter_report.json` and
`si_correction_calibration.json` are regenerated locally rather than tracked.
