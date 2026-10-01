"""Extrapolation check: dD models on large tier-2 (Y-MB) structures and on the monomer library."""
import numpy as np, pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen
import dispersion as dp
HSP=dp.HSP
vmod,p=dp.main()
cv=pd.read_csv(HSP/'results/benchmark/reference/model_cv_predictions.csv',usecols=['ikey14','tier','n_heavy','n_si','cls','ref_D','SP_D','ET_D','ET_resid_D','canonical_smiles'])
cv['ENS_D']=(cv.ET_D+cv.ET_resid_D)/2
t2=cv[(cv.tier==2)&(cv.n_si==0)].copy()
mols=[Chem.MolFromSmiles(s) for s in t2.canonical_smiles]
t2['dD_rd']=dp._dd(p,np.array([Crippen.MolMR(m) for m in mols])/vmod.predict(mols)[0])
for lo,hi in [(0,15),(15,30),(30,500)]:
    x=t2[(t2.n_heavy>lo)&(t2.n_heavy<=hi)]
    print(f'tier2 heavy atoms ({lo},{hi}] n={len(x)}  MAE dD_rd {(x.dD_rd-x.ref_D).abs().mean():.2f}  SP {(x.SP_D-x.ref_D).abs().mean():.2f}  ENS {(x.ENS_D-x.ref_D).abs().mean():.2f}  bias SP {(x.SP_D-x.ref_D).mean():+.2f} rd {(x.dD_rd-x.ref_D).mean():+.2f}')
m=pd.read_csv(HSP/'results/v2/monomers_hsp_ens.csv',usecols=['monomer_smiles','D','SP_D','has_si'],low_memory=False)
vm=[];unk=[];rd=[]
smi=m.monomer_smiles.tolist()
for i in range(0,len(smi),2000):
    mm=[Chem.MolFromSmiles(s) for s in smi[i:i+2000]]
    a,b=vmod.predict(mm); vm+=list(a); unk+=list(b); rd+=[Crippen.MolMR(x) for x in mm]
vm=np.array(vm); unk=np.array(unk); m['Vm']=vm; m['vm_unknown_atoms']=unk
m['dD_rd']=dp._dd(p,np.array(rd)/vm)
print(m[['D','SP_D','dD_rd']].describe().round(2).to_string()); print('unknown atom types in volume model:',(unk>0).sum())
print('ref tier1 dD range', cv[cv.tier==1].ref_D.quantile([0,.01,.5,.99,1]).round(1).to_dict(), 'tier2', cv[cv.tier==2].ref_D.quantile([0,.01,.5,.99,1]).round(1).to_dict())
m.to_csv(HSP/'results/benchmark/dispersion/monomers_dD_rd.csv',index=False)
