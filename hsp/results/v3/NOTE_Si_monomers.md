# Si monomers are NOT reliable

The 307 Si-containing monomers are scored and ranked in v3, but their HSP values should not be
trusted. Filter them out (`has_si == True`) for any decision.

Why: the Si groups of the refitted formula (Si atom type, Si-O-Si, Si-O-C, Si-CH3, Si-OH,
Si-X, Si-N) were fitted on 67 HSPiP Si compounds, only 7 of them in the curated master set. All
of these are small: median 9 heavy atoms, against about 41 for the Si monomers. Cross-validation
on those 7 gave a median Ra error of 2.1, but that does not test the monomer size range.

In the tables: `hsp_source = gc_formula_si`, `expected_Ra_err = NaN`, and `hsp_note` carries this
warning. The ranking grids show a "Si: HSP unreliable" badge. `si_monomers.csv` lists them all.
