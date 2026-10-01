import sys; sys.path.insert(0,'.')
import pandas as pd, numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from hsp_calculator import compute_hsp
from solvent_incompatibility import _ra
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=sys.argv[1]
mols=[("1 cholesteryl NB ester","CC(CCC[C@H]([C@@]1([H])CC[C@]2([H])[C@]1(C)CC[C@@]3([H])[C@@]2([H])CC[C@]4([H])[C@]3(C)CC[C@H](OC(C5CC6C=CC5C6)=O)C4)C)C"),
("2 bis(2-ethylhexyl) NB diester","O=C(C1C(C(OCC(CC)CCCC)=O)C2C=CC1C2)OCC(CC)CCCC"),
("3 bis(2-hydroxyethyl) NB diester","O=C(C1C(C(OCCO)=O)C2C=CC1C2)OCCO"),
("4 bis(2-cyanoethyl) NB diester","O=C(C1C(C(OCCC#N)=O)C2C=CC1C2)OCCC#N"),
("5 tert-butyl NB ester","O=C(OC(C)(C)C)C1C2C=CC(C2)C1"),
("6 2-hydroxyethyl NB ester","O=C(OCCO)C1C2C=CC(C2)C1")]
REF={"Water":(15.5,16.0,42.3),"Diiodomethane":(17.8,3.9,5.5),"n-Hexadecane":(16.3,0.0,0.0),"PTFE":(12.7,0.0,0.0)}
rows=[]
for name,smi in mols:
    r=compute_hsp(smi); m=Chem.MolFromSmiles(smi)
    d=dict(id=name,smiles=smi,MW=round(Descriptors.MolWt(m),1),dD=r.delta_D,dP=r.delta_P,dH=r.delta_H,
           n_unmatched=r.n_unmatched_atoms,unavail=(r.n_unavail_d,r.n_unavail_p,r.n_unavail_hb),
           nb_ring_approx=r.approx_ring_correction,second_order=r.used_2nd_order,err=r.error,groups=r.group_counts)
    for k,v in REF.items(): d['Ra_'+k]=round(float(_ra(r.delta_D,r.delta_P,r.delta_H,*v)),2)
    rows.append(d)
df=pd.DataFrame(rows); df.to_csv(f'{out}/hsp6.csv',index=False)
pd.set_option('display.width',250); pd.set_option('display.max_colwidth',200)
print(df.drop(columns=['smiles','groups']).round(2).to_string(index=False))
for d in rows: print(d['id'],d['groups'])
ms=[Chem.RemoveHs(Chem.MolFromSmiles(s)) for _,s in mols]
leg=[f"{r['id']}\nδD {r['dD']:.1f} δP {r['dP']:.1f} δH {r['dH']:.1f}\nRa(PTFE) {r['Ra_PTFE']:.1f}" for r in rows]
img=Draw.MolsToGridImage(ms,molsPerRow=3,subImgSize=(420,340),legends=[l.replace('\n',' | ') for l in leg])
img.save(f'{out}/structures.png')
fig,ax=plt.subplots(1,3,figsize=(15,4.6))
pairs=[('dD','dP'),('dD','dH'),('dP','dH')]
for a,(x,y) in zip(ax,pairs):
    for i,r in df.iterrows():
        a.scatter(r[x],r[y],s=60,zorder=3); a.annotate(r['id'].split()[0],(r[x],r[y]),xytext=(4,4),textcoords='offset points')
    rx={'dD':0,'dP':1,'dH':2}
    for k,v in REF.items():
        if k=='Water': continue
        a.scatter(v[rx[x]],v[rx[y]],marker='*',s=160,c='k'); a.annotate(k,(v[rx[x]],v[rx[y]]),xytext=(4,-10),textcoords='offset points',fontsize=8)
    a.set_xlabel(f'δ{x[1]} (MPa$^{{1/2}}$)'); a.set_ylabel(f'δ{y[1]} (MPa$^{{1/2}}$)'); a.grid(alpha=.3)
fig.suptitle('Stefanis–Panayiotou HSP, 6 norbornene monomers (water off-scale: 15.5/16.0/42.3)')
fig.tight_layout(); fig.savefig(f'{out}/hansen_space.png',dpi=150)
