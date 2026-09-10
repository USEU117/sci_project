"""Predeclared contrast/polarity stress test on support-only synthetic scratches."""
from pathlib import Path
import os, sys, json, time, gc
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
import cv2
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT/'scripts/innovation_followup_20260908',ROOT/'scripts/innovation_v14_decisive_validation_20260905',ROOT/'scripts/innovation_v12_new_observables',ROOT/'methods/anomalydino'):
    sys.path.insert(0,str(p))
from src.backbones import get_model
from src.utils import dists2map
from v14_common import DATA_ROOT, support_paths, assert_fit_ids_are_support
from ntof_render import render_synthetic
from audit_synthetic_signal import seed
torch.set_num_threads(2); cv2.setNumThreads(1)
OUT=ROOT/'experiments/dynamic_fusion/innovation_followup_20260908/scratch_shift'
CATS=['bracket_black','metal_plate']
VARIANTS=['legacy_dark','bright110','dark20','bright20']
def vectors(f):
    return torch.nn.functional.normalize(torch.as_tensor(f,device='cuda',dtype=torch.float32).reshape(-1,f.shape[-1]),dim=-1)
def score(q,b):
    q=vectors(q); b=vectors(b)
    ds=[]
    for i in range(0,len(q),512):
        ds.append(torch.sqrt(torch.clamp(2-2*(q[i:i+512]@b.T).max(dim=1).values,min=0)))
    return torch.cat(ds).cpu().numpy()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(categories=CATS,shot=2,seed=0,scratch_seed=0,edges=[448,896],variants=VARIANTS,
                  role='support-only stress test; frozen model; memory=other clean support',
                  map='Euclidean unit-feature NN -> identical dists2map to1024',
                  hypothesis='does resolution benefit persist across contrast sign and strength?',
                  limits='4 underlying masks; not generalization proof; no parameter selection')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    assert torch.cuda.is_available()
    start=time.perf_counter(); rows=[]; parity=[]
    for edge in protocol['edges']:
        model=get_model('dinov2_vitb14','cuda:0',smaller_edge_size=edge)
        model.model.eval()
        for cat in CATS:
            refs=support_paths(cat,2);assert_fit_ids_are_support(refs,cat,2)
            if edge==448:
                path=ROOT/'outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2'/f'{cat}.npz'
            else:
                path=ROOT/'outputs/dynamic_fusion/overnight_20260908_full896/features'/f'{cat}.npz'
            with np.load(path,allow_pickle=False) as z:
                assert z['ref_rel'].astype(str).tolist()==refs
                bank=z['clean_feat']; old_syn=z['syn_feat']
                old_masks=z['syn_masks']
            for h,rel in enumerate(refs):
                x=cv2.cvtColor(cv2.imread(str(DATA_ROOT/rel)),cv2.COLOR_BGR2RGB)
                legacy,mask=render_synthetic(x,'thin_scratch',seed(cat,rel,'thin_scratch',0))
                e=6 if edge==448 else 0
                assert np.array_equal(mask,old_masks[h,e])
                pos=mask>0
                for variant in VARIANTS:
                    if time.perf_counter()-start>1200: raise RuntimeError('20-minute probe budget reached; see partial')
                    y=legacy.copy() if variant=='legacy_dark' else x.copy()
                    if variant!='legacy_dark':
                        delta={'bright110':110,'dark20':-20,'bright20':20}[variant]
                        y[pos]=np.clip(x[pos].astype(np.int16)+delta,0,255).astype(np.uint8)
                    tensor,grid=model.prepare_image(y)
                    torch.cuda.synchronize();t=time.perf_counter()
                    with torch.inference_mode(): f=model.extract_features(tensor).astype(np.float32).reshape(*grid,-1)
                    torch.cuda.synchronize(); enc=time.perf_counter()-t
                    if variant=='legacy_dark': parity.append(dict(category=cat,ref=rel,edge=edge,max_feature_error=float(np.abs(f-old_syn[h,e]).max())))
                    with torch.inference_mode(): s=score(f,bank[1-h]).reshape(*grid)
                    m=dists2map(s,(1024,1024))
                    rows.append(dict(category=cat,ref=rel,memory_ref=refs[1-h],edge=edge,variant=variant,
                                     pixel_ap=float(average_precision_score(pos.ravel(),m.ravel())),
                                     mask_pixels=int(pos.sum()),contrast=float(np.abs(y[pos].astype(np.int16)-x[pos].astype(np.int16)).max(axis=1).mean()),encode_s=enc))
                    (OUT/'PARTIAL.json').write_text(json.dumps(dict(status='running',protocol=protocol,rows=rows,parity=parity),indent=2),encoding='utf-8')
            print(f'done {edge} {cat}',flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    summary=[]
    for v in VARIANTS:
        r={'variant':v,'underlying_masks':4}
        for edge in [448,896]: r[f'ap_{edge}']=float(np.mean([x['pixel_ap'] for x in rows if x['edge']==edge and x['variant']==v]))
        r['delta']=r['ap_896']-r['ap_448'];summary.append(r)
    (OUT/'RESULTS.json').write_text(json.dumps(dict(status='complete',protocol=protocol,rows=rows,parity=parity,summary=summary,elapsed_s=time.perf_counter()-start),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
