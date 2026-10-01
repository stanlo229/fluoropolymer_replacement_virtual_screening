# Si monomers (ranked separately: values may not be reliable)

These 307 Si-containing monomers are scored with the v3 group-contribution formula and ranked
on their own here (`top25_ranked_*`, 199 pass the ranking constraints). The main rankings in
`results/v3/` contain no Si monomers. Do not mix the two lists for decisions without measurement.

**Why the values may be reliable**
- On held-out small Si compounds the formula is as accurate as for non-Si compounds: median Ra
  error 2.1 on the 7 curated Si compounds (non-Si: 2.8), and 1.9 on 44 further Si compounds.
  Published Stefanis–Panayiotou gets 9.8 on the same 7.
- Siloxanes are reproduced well (δD within about 0.4–0.9, δP and δH near zero).
- Almost all monomers use well-supported Si groups: the Si atom type (47 compounds), Si–CH₃ (30) and
  Si–O–C (16). Only 17 of 307 depend on a group seen in fewer than 10 compounds.
- The size-intensive formula weights each group by its share of the molecule. Si is about 5% of a
  monomer's heavy atoms, so most of the value comes from the well-supported organic part.
- The values are physically plausible: δD 14.2–20.0, between Si reference compounds and non-Si
  monomers, with nothing negative.

**Why the values may not be reliable**
- Only 7 curated Si data points exist, and 5 are siloxanes; only 4 monomers are siloxanes. The
  other Si references are HSPiP's own estimates, not measurements.
- The dominant monomer chemistry is barely covered. Silyl ethers (144 monomers) have one curated
  reference, which is the worst-predicted (error 4.8). TBS ethers (104 monomers) appear in no
  reference at all. Si–N (15 monomers) rests on one compound, and Si–Si on none.
- The references are small: median 10 heavy atoms, against 41 for these monomers. Only 93 of 307
  are inside the reference size range.
- Independent methods disagree by more than the error bar: the earlier Si model differs from v3 by a
  median Ra of 3.1.
- The best Si monomer beats the best non-Si monomer in Ra(PTFE) by only 1.1, less than the
  expected error, so the Si lead in PTFE-likeness is not resolved.

**To make them reliable:** measure HSP for 3–5 Si monomers (a TBS ether, a TMS ether, a siloxane and
a C-only silane), or obtain independent Y-MB values from the HSPiP software.

Files: `si_monomers.csv` (all 307 with v3 values and the `hsp_note`), `top25_ranked_*.csv/png`
(Si-only rankings), `monomers_filtered_ranked.csv` (the 199 that pass the ranking constraints).
