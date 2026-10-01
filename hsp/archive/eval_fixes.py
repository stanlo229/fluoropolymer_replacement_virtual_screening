"""Effect of the four worked-example second-order fixes on tier-1 (HSPiP master) error and on library negatives."""
import sys, logging; logging.disable(logging.WARNING)
import pandas as pd, numpy as np
from rdkit import RDLogger; RDLogger.DisableLog('rdApp.*')
from sp_fixes import compute
S='/tmp/claude-3105613/-project-6101772-stanlo-Repos-MarkushOCSR/aff4d277-6549-4ae7-b122-1046cb5cf735/scratchpad/si/reference_sp.csv'
r=pd.read_csv(S); r=r[r.sp_domain&(r.unmatched==0)&r.sp_D.notna()&~r.has_si]
ALL=('olefin','arom_oh','arom_n','string')
V={'current':((),'per_ring',False),'current+floor':((),'per_ring',True),'fixes+floor':(ALL,'per_ring',True),
   'fixes+per_system+floor':(ALL,'per_system',True),'fixes+isolated_only+floor':(ALL,'isolated_only',True)}
rows=[]
for vn,(fx,rm,fl) in V.items():
    p=np.array([compute(s,fx,rm,fl)[:3] for s in r.smiles],dtype=float)
    for t in ['master','10k_YMB']:
        for cls in ['all','fused/bridged aliphatic','mono/isolated rings']:
            k=(r.tier==t)&((r.ring_class==cls) if cls!='all' else True)
            e=np.abs(p[k.values]-r.loc[k,['dD','dP','dH']].values)
            ra=np.sqrt(4*(p[k.values,0]-r.loc[k,'dD'])**2+(p[k.values,1]-r.loc[k,'dP'])**2+(p[k.values,2]-r.loc[k,'dH'])**2)
            rows.append(dict(variant=vn,tier=t,cls=cls,n=int(k.sum()),MAE_D=e[:,0].mean(),MAE_P=e[:,1].mean(),MAE_H=e[:,2].mean(),med_Ra=np.median(ra),neg=(p[k.values,1:]<0).any(1).mean()))
out=pd.DataFrame(rows).round(3); print(out.to_string(index=False)); out.to_csv('fix_eval_reference.csv',index=False)
lib=pd.read_csv('../results/monomers_hsp.csv',usecols=['monomer_smiles']).sample(10000,random_state=0)
for vn,(fx,rm,fl) in V.items():
    q=[compute(s,fx,rm,False) for s in lib.monomer_smiles]
    neg=np.mean([i[3].get('raw_negative',False) for i in q if i[0] is not None]); ua=np.mean([i[3].get('unavail',0)>0 for i in q if i[0] is not None])
    print(f'library 10k sample {vn:28s} raw-negative {neg:.3f}   any-unavailable-term {ua:.3f}')
