"""Known dendrons (must be detected, with the right focal group) and look-alikes (must not)."""
from rdkit import Chem
import dendrons as dd

POS = {  # name: (smiles, family, min generation, focal kind)
    "Frechet G1-OH": ("OCc1cc(OCc2ccccc2)cc(OCc2ccccc2)c1", "frechet", 1, "OH"),
    "Frechet G2-OH": ("OCc1cc(OCc2cc(OCc3ccccc3)cc(OCc3ccccc3)c2)cc(OCc2cc(OCc3ccccc3)cc(OCc3ccccc3)c2)c1", "frechet", 2, "OH"),
    "Percec G1 tris(dodecyloxy)benzyl-OH": ("OCc1cc(OCCCCCCCCCCCC)c(OCCCCCCCCCCCC)c(OCCCCCCCCCCCC)c1", "percec", 1, "OH"),
    "Percec 3,5-bis(octyloxy)benzyl-OH": ("OCc1cc(OCCCCCCCC)cc(OCCCCCCCC)c1", "percec", 1, "OH"),
    "bis-MPA G1 acetonide, hydroxyethyl focal": ("OCCOC(=O)C1(C)COC(C)(C)OC1", "bis_mpa", 1, "OH"),
    "bis-MPA G1 OH-terminated, hydroxyethyl focal": ("OCCOC(=O)C(C)(CO)CO", "bis_mpa", 1, "OH"),
    "PAMAM-type G1 amine focal": ("NCCN(CCC(=O)NCCN)CCC(=O)NCCN", "pamam", 1, "NH"),
    "Newkome/Behera amine": ("NC(CCC(=O)OC(C)(C)C)(CCC(=O)OC(C)(C)C)CCC(=O)OC(C)(C)C", "newkome", 1, "NH"),
    "polyglycerol G1 1,3-dibenzyloxy-2-propanol": ("OC(COCc1ccccc1)COCc1ccccc1", "glycerol", 1, "OH"),
    "polyglycerol G1 acetonide (Haag)": ("OC(COCC1COC(C)(C)O1)COCC1COC(C)(C)O1", "glycerol", 1, "OH"),
}
NEG = {
    "benzyl alcohol": "OCc1ccccc1", "3,5-dimethoxybenzyl alcohol": "OCc1cc(OC)cc(OC)c1",
    "1,3-dimethoxy-2-propanol": "OC(COC)COC", "triethanolamine": "OCCN(CCO)CCO",
    "pentaerythritol tetraacetate-ol": "OCC(COC(C)=O)(COC(C)=O)COC(C)=O", "4-hexyloxybenzyl alcohol": "OCc1ccc(OCCCCCC)cc1",
    "methyl bis-MPA (no unique focal)": "COC(=O)C(C)(CO)CO", "2-ethylhexanol": "CCCCC(CC)CO",
    "1,3-diethoxy-2-propanol": "CCOCC(COCC)O", "1,3-diphenoxy-2-propanol": "OC(COc1ccccc1)COc1ccccc1",
    "triglycerol (linear)": "OCC(O)COCC(O)COCC(O)CO", "triglycerol diacrylate (linear)": "C=CC(=O)OCC(O)COCC(O)COCC(O)COC(=O)C=C",
}
ok = True
for name, (smi, fam, gmin, focal) in POS.items():
    i = dd.analyse(Chem.MolFromSmiles(smi))
    f_ok = (i.focal_alcohol is not None) if focal == "OH" else (i.focal_amine is not None)
    good = i.is_dendron and i.family == fam and i.generation >= gmin and f_ok
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  {name:46s} family={i.family:14s} G={i.generation} units={i.n_branch_units} focalOH={i.focal_alcohol} focalNH={i.focal_amine}")
for name, smi in NEG.items():
    i = dd.analyse(Chem.MolFromSmiles(smi))
    usable = i.is_dendron and (i.focal_alcohol is not None or i.focal_amine is not None)
    good = not usable
    ok &= good
    print(f"{'PASS' if good else 'FAIL'}  NOT a usable dendron: {name:33s} -> is_dendron={i.is_dendron} {i.family} focal={i.focal_alcohol, i.focal_amine}")
print("ALL PASS" if ok else "SOME FAILED")
