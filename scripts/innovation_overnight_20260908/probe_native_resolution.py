"""Same frozen support features, native-pixel labels: 16 vs 32 grid audit."""
from pathlib import Path
import sys, json, time
from datetime import datetime, timezone
import numpy as np
import cv2
import torch
from sklearn.metrics import average_precision_score

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts/innovation_v14_decisive_validation_20260905'))
from industrial_ad.innovation_v10_portfolio.common import resize_patches
from v14_common import assert_fit_ids_are_support
CATS=['bracket_black','bracket_brown','bracket_white','connector','metal_plate','tubes']
BASE=ROOT/'outputs/dynamic_fusion/v14_p1_support'
OUT=ROOT/'experiments/dynamic_fusion/innovation_overnight_20260908/native_resolution'
torch.set_num_threads(2)
cv2.setNumThreads(1)

def unit(t): return torch.nn.functional.normalize(t,dim=-1)

def vectors(d,c,g):
    out=[]
    for x in (d,c):
        t=torch.from_numpy(np.asarray(x,dtype=np.float32))
        if g==16:
            t=t.reshape(*t.shape[:-3],16,2,16,2,t.shape[-1]).mean(dim=(-2,-4))
        out.append(unit(t.reshape(-1,g*g,t.shape[-1])))
    return unit(torch.cat(out,dim=-1))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(shot=2,seed=0,categories=CATS,episodes=[0,3,6],grids=[16,32],
                  fitting='none',memory='other support image only',mask='native 1024 binary; no mask downsampling',
                  interpolation='bilinear score maps, identical for both grids',
                  status='measurement/probe, not frozen A1 parity or confirmation',
                  decision='report all per-category and family differences; no tuning or winner selection')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    start=time.time(); rows=[]
    for cat in CATS:
        if datetime.now(timezone.utc)>=datetime(2026,9,7,23,0,tzinfo=timezone.utc): break
        with np.load(BASE/'v14_p1_support_dino_s0_k2'/f'{cat}.npz',allow_pickle=False) as z:
            refs=z['ref_rel'].astype(str).tolist(); assert_fit_ids_are_support(refs,cat,2)
            d=z['clean_feat']; ds=z['syn_feat'][:,[0,3,6]]; masks=z['syn_masks'][:,[0,3,6]]
        with np.load(BASE/'v14_p1_support_clip_s0_k2'/f'{cat}.npz',allow_pickle=False) as z:
            assert z['ref_rel'].astype(str).tolist()==refs
            c=resize_patches(z['clean_feat'],(32,32))
            x=z['syn_feat'][:,[0,3,6]]
            cs=resize_patches(x.reshape(-1,*x.shape[-3:]),(32,32)).reshape(2,3,32,32,-1)
        predictions={}
        for g in [16,32]:
            bank=vectors(d,c,g); query=vectors(ds,cs,g).reshape(2,3,g*g,-1)
            for h in range(2):
                s=1-(query[h]@bank[1-h].T).max(dim=-1).values
                for e,family in enumerate(['cutpaste','local_erasure','thin_scratch']):
                    y=masks[h,e]>0
                    score=cv2.resize(s[e].numpy().reshape(g,g),(y.shape[1],y.shape[0]),interpolation=cv2.INTER_LINEAR)
                    ap=float(average_precision_score(y.ravel(),score.ravel()))
                    rows.append(dict(category=cat,held_ref=refs[h],family=family,episode=[0,3,6][e],grid=g,
                                     n_positive=int(y.sum()),prevalence=float(y.mean()),pixel_ap=ap,ap_over_prevalence=ap/float(y.mean())))
        (OUT/f'{cat}.json').write_text(json.dumps(rows[-12:],indent=2),encoding='utf-8')
        print(f'done {cat}',flush=True)
    agg=[]
    for family in ['cutpaste','local_erasure','thin_scratch']:
        entry={'family':family}
        for g in [16,32]:
            rs=[r for r in rows if r['family']==family and r['grid']==g]
            entry[f'ap_{g}']=float(np.mean([r['pixel_ap'] for r in rs])) if rs else None
            entry[f'n_{g}']=len(rs)
        if entry['ap_16'] is not None: entry['delta_32_minus_16']=entry['ap_32']-entry['ap_16']
        agg.append(entry)
    (OUT/'RESULTS.json').write_text(json.dumps(dict(protocol=protocol,rows=rows,summary=agg,elapsed_s=time.time()-start),indent=2),encoding='utf-8')
    print(json.dumps(agg),flush=True)

if __name__=='__main__': main()
