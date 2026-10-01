"""dD on branched aliphatic esters/ethers/hydrocarbons: refractivity model vs S-P vs ENS (CV predictions)."""
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
import dispersion as dp
vmod,p=dp.main()
cv=pd.read_csv(dp.HSP/'results/benchmark/reference/model_cv_predictions.csv',usecols=['tier','Name','canonical_smiles','n_heavy','n_si','ref_D','SP_D','ET_D','ET_resid_D'])
cv['ENS_D']=(cv.ET_D+cv.ET_resid_D)/2
quat=Chem.MolFromSmarts('[CX4H0]([#6])([#6])([#6])'); arom=Chem.MolFromSmarts('a')
sel=[]
for s in cv.canonical_smiles:
    m=Chem.MolFromSmiles(s); sel.append(m is not None and m.HasSubstructMatch(quat) and not m.HasSubstructMatch(arom) and all(a.GetSymbol() in 'CHO' for a in m.GetAtoms()))
x=cv[np.array(sel)&(cv.n_si==0)&(cv.n_heavy>=8)].copy()
mols=[Chem.MolFromSmiles(s) for s in x.canonical_smiles]
x['dD_rd']=dp._dd(p,np.array([Crippen.MolMR(m) for m in mols])/vmod.predict(mols)[0])
for t in (1,2):
    y=x[x.tier==t]
    print(f'tier{t} branched (quaternary C) aliphatic C/H/O, >=8 heavy atoms: n={len(y)}  MAE rd {(y.dD_rd-y.ref_D).abs().mean():.2f} (bias {(y.dD_rd-y.ref_D).mean():+.2f})  SP {(y.SP_D-y.ref_D).abs().mean():.2f} (bias {(y.SP_D-y.ref_D).mean():+.2f})  ENS {(y.ENS_D-y.ref_D).abs().mean():.2f} (bias {(y.ENS_D-y.ref_D).mean():+.2f})')
pd.set_option('display.width',200)
print(x[x.tier==1][['Name','n_heavy','ref_D','dD_rd','SP_D','ENS_D']].round(2).sort_values('n_heavy').to_string(index=False))
big=x[x.n_heavy>=16]; print('>=16 heavy atoms n=',len(big),' MAE rd %.2f SP %.2f ENS %.2f'%((big.dD_rd-big.ref_D).abs().mean(),(big.SP_D-big.ref_D).abs().mean(),(big.ENS_D-big.ref_D).abs().mean()))
