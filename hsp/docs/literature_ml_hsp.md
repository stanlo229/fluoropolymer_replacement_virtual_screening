# Literature: ML and refitted group contribution for HSP

Compiled 2026-09-30. Units are MPa½, errors are δD / δP / δH. "Second-hand" means
the number was taken from another paper's citation because the original was paywalled.

| Paper | Method | Training data | Validation | Reported error (δD / δP / δH) | Si / large molecules | Code / data |
|---|---|---|---|---|---|---|
| Sanchez-Lengeling et al., *Adv. Theory Simul.* 2019, 2, 1800069, [doi](https://doi.org/10.1002/adts.201800069) (gpHSP) | Gaussian process on Mordred / COSMOtherm / QM descriptors | 193 experimental solvents + 31 measured polymers | CV | MAE 0.68 / 1.93 / 1.57 (second-hand, quoted by Mathieu and Pang) | none | [the-matter-lab/gpHSP](https://github.com/the-matter-lab/gpHSP), MIT; ships `data/HSPiP.csv` |
| Mathieu, *ACS Omega* 2018, 3, 17049, [doi](https://doi.org/10.1021/acsomega.8b02601) | δD from Vm + refractivity; δP, δH = √(ΣE / Vm) from fragment energies; 3 / 13 / 9 parameters | 174 experimental (Hansen handbook) | LOO + external test of 769 handbook estimates | LOO 0.68 / 2.00 / 1.55; external 0.75 / 2.08 / 1.67. **S-P refitted on the same 174, LOO: 0.49 / 1.75 / 1.95** | Si excluded | Script + data in the SI |
| Járvás et al., *Fluid Phase Equilib.* 2011, 309, 8, [doi](https://doi.org/10.1016/j.fluid.2011.06.030) | ANN on 5 COSMO σ-moments | not verified | 17-compound test set | AAD 1.37 / 1.85 / 2.58 (second-hand) | unknown | no |
| Terrell, *Chem. Eng. Sci.* 2022, 248B, 117184, [doi](https://doi.org/10.1016/j.ces.2021.117184) | LASSO / Ridge "adaptable" group contribution | experimental small molecules, applied to 185 biomass oligomers | not verified (paywall) | not read | large oligomers, no experimental check | no repo |
| Przybyłek et al., *J. Chem.* 2019, 9858371, [arXiv](https://arxiv.org/abs/1901.03408) | MARSplines on PaDEL descriptors | 130 experimental | LOO / LMO | CV MAE 0.43 / 1.25 / 1.15 (small set, optimistic) | no | no |
| Al-Sakkari et al., *Digit. Chem. Eng.* 2025, 14, 100207, [doi](https://doi.org/10.1016/j.dche.2024.100207) | kNN / DT / RF / XGB / SVR / ANN / LSTM + decision fusion on RDKit + Morgan | ~12,000 from HSPiP + literature (mostly Y-MB estimates) | random 85/15 | test RMSE ≈ 0.38 / 1.03 / 0.9 (agreement with Y-MB, not experiment) | ≥1 Si compound, no analysis | paper and SI open, no code |
| Pang, Pine, Sulemana, *Digital Discovery* 2024, [doi](https://doi.org/10.1039/D3DD00119A) | fine-tuned ChemBERTa; Mol2Vec / Morgan + XGBoost / FFNN | 1,183 experimental (Abbott set) | random 6-fold | best MAE 0.59 / 2.01 / 1.79; Morgan + XGBoost 0.65 / 2.23 / 2.13 | notes group contribution is weaker for large multifunctional molecules | [jiayunpang/hsp_embedding](https://github.com/jiayunpang/hsp_embedding) |
| Cvetković et al., *Chemom. Intell. Lab. Syst.* 2024, 251, 105168, [doi](https://doi.org/10.1016/j.chemolab.2024.105168) | GA-MLR, XGBoost (Mordred), GNNs (AttentiveFP / GAT) | 1,192 (Abbott set) | random 80/20, nested CV + 93 independent compounds | test RMSE (best per component) 0.72 / 2.10 / 2.22 | no | [darjacvetkovic/HSP-predictions](https://github.com/darjacvetkovic/HSP-predictions) |
| Wojeicchowski et al., *I&EC Res.* 2022, 61, 15631, [doi](https://doi.org/10.1021/acs.iecr.2c01592) | linear regression on COSMO-RS σ-moments | 195 literature (133 / 62) | random split, outliers removed before scoring | test MAE 0.98 / 1.74 / 1.44 | no | data in the SI |
| Hassan & Kazemi, *Sci. Rep.* 2025, [doi](https://doi.org/10.1038/s41598-025-12758-1) | 14 ML models on physical properties | 1,799 DIPPR polymer points | random 70/15/15 | **total Hildebrand δ only**, not HSP components | no | on request |
| Faasen et al., *JCIS* 2020, 575, 326, [doi](https://doi.org/10.1016/j.jcis.2020.04.070) | MD-derived HSP of siloxane surfactants | – | – | no comparison with experiment | the only Si-focused HSP work found | CC-BY |

Not readable (paywall): the full text of gpHSP, Járvás 2011, Terrell 2022, Pantelidou et al. *Key Eng. Mater.* 2025 (XGBoost / CatBoost on HSPiP), a *J. Mol. Liq.* 2024 cocrystal paper (ANN / XGBoost on group counts + σ-moments), and Hukkerikar 2012 (GC+).

## Synthesis
1. **Achievable error.** On the ~1.2k experimental Hansen / Abbott set with random splits, the best models reach MAE of about 0.6–0.7 / 1.9–2.1 / 1.6–1.8. Training and scoring on the ~10k HSPiP set gives much lower numbers, but that measures agreement with Y-MB, not with experiment. **No paper uses a scaffold, cluster or size-based split**, so extrapolation error has never been reported.
2. **ML does not clearly beat refitted group contribution.** On the same reference data, Mathieu's S-P refit is better than gpHSP on δD and δP, and ChemBERTa is no better than either. Gains over a Morgan + XGBoost baseline are small.
3. **Refitted group values.** Mathieu 2018 is the only verified refit, on 174 compounds. He argues that linear-in-δ forms (S-P, Marrero–Gani) "are inconsistent with the definition of the HSP components as size-intensive quantities", and uses √(ΣE / Vm) himself. No paper refits a group-contribution scheme on the full HSPiP ~10k set.
4. **Large molecules and Si.** There is essentially no direct evidence. No ML paper reports Si coverage or per-size errors, Mathieu excludes Si, and Pirika notes that Y-MB had problems with very large molecules.

## How this project compares (scaffold-grouped 5-fold CV, `analysis/gc_benchmark.py`)
Median Ra error on the curated master set (n = 961): refitted size-intensive formula 2.83, XGBoost on fingerprints 3.05, published S-P 3.34, XGBoost on group counts 3.40, S-P equations refitted 3.64, ExtraTrees 3.96. MAE for the formula is 0.58 / 1.92 / 1.98, in line with the literature's random-split figures despite the stricter scaffold split.
