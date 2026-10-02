# Fluoropolymer replacement virtual screening

Virtual screening of fluorine-free norbornene monomers (for ROMP polymers) as PTFE replacements,
built from purchasable catalogue alcohols and amines.

| Folder | Contents |
|---|---|
| `dataset/` | Catalogue scraping via PubChem (Sigma-Aldrich, Ambeed, Combi-Blocks, TCI, Thermo Fisher, Oakwood, Matrix): `scrape_catalogues.py`, `catalogues.csv`. Scope: monoalcohols, diols, monoamines and dendrons with one focal OH/amine (`dendrons.py`); no fluorine or metals; since 2026-10 no MW limit and chiral compounds allowed (the original library used MW < 500 and no chiral centres). |
| `hsp/` | Hansen solubility parameters of the monomer library and the PTFE / water / diiodomethane / hexadecane rankings. See `hsp/README.md`. |
| `ssip/` | Surface site interaction point (SSIP) pipeline: geometry, charges, MEP surfaces, SSIP footprints, calibration. See `ssip/virtual_screen_plan.md`. |
| `runs/` | SSIP benchmark run outputs |

Heavy calculations run as SLURM jobs (`hsp/jobs/`, `ssip/submit_*.sh`). Licensed HSPiP data is
kept in gitignored folders and must never be committed.
