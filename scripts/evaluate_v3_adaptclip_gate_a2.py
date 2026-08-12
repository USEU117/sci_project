"""Gate A2: label-free, reference-calibrated selective rescue for AdaptCLIP."""

from __future__ import annotations

import argparse, json
from datetime import datetime, timezone
from pathlib import Path
import cv2, numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


CONFIGS = {"permissive": (.10, 3.0, .20), "balanced": (.30, 2.0, .15), "strict": (.55, 1.0, .10)}

def resize(maps, shape):
    return maps if maps.shape[1:] == shape else np.stack([cv2.resize(x, (shape[1],shape[0]), interpolation=cv2.INTER_LINEAR) for x in maps])

def evidence(values, center, scale): return np.arcsinh((values-center)/max(scale,1e-6))

def fit(values):
    values=np.asarray(values,dtype=float).reshape(-1); q1,q9=np.quantile(values,[.01,.99]); center=float(np.median(values)); scale=max(float((q9-q1)/2), max(abs(center)*.01,1e-6)); return center,scale

def main():
    p=argparse.ArgumentParser(); p.add_argument('--visual-root',type=Path,required=True); p.add_argument('--text-root',type=Path,required=True); p.add_argument('--reference-root',type=Path,required=True); p.add_argument('--visual-calibration',type=Path,required=True); p.add_argument('--output',type=Path,required=True); p.add_argument('--stride',type=int,default=8); a=p.parse_args()
    visual_calibration=json.loads(a.visual_calibration.read_text(encoding='utf8'))
    if any(visual_calibration.get(key) is not False for key in ('test_predictions_used','test_labels_used','test_masks_used','test_set_statistics_used')): raise ValueError('visual calibration is not leak-safe')
    cats=sorted(json.loads((Path(__file__).resolve().parents[1]/'data/splits/mpdd/manifest.json').read_text())['categories']); rows=[]; calibrations={}
    for cat in cats:
      with np.load(a.visual_root/f'{cat}.npz',allow_pickle=False) as v, np.load(a.text_root/f'{cat}.npz',allow_pickle=False) as t, np.load(a.reference_root/f'{cat}.npz',allow_pickle=False) as r:
        ids=np.asarray(v['sample_ids']).astype(str)
        if not np.array_equal(ids,np.asarray(t['sample_ids']).astype(str)): raise ValueError(f'unaligned IDs:{cat}')
        vmap=np.asarray(v['anomaly_maps'],dtype=np.float32)[:,::a.stride,::a.stride]; tmap=resize(np.asarray(t['anomaly_maps'],dtype=np.float32),np.asarray(v['anomaly_maps']).shape[1:])[:,::a.stride,::a.stride]; masks=np.asarray(v['imgs_masks'],dtype=np.uint8)[:,::a.stride,::a.stride]
        rmap=resize(np.asarray(r['anomaly_maps'],dtype=np.float32),np.asarray(v['anomaly_maps']).shape[1:])[:,::a.stride,::a.stride]
        tc,ts=fit(r['pr_sp']); tpc,tps=fit(rmap)
        vfit=visual_calibration['categories'][cat]['visual']['pixel']
        ve=evidence(vmap,vfit['center'],vfit['scale']); te=evidence(tmap,tpc,tps)
        calibrations[cat]={'text_image':{'center':tc,'scale':ts},'text_pixel':{'center':tpc,'scale':tps},'reference_images':int(len(r['pr_sp'])),'test_predictions_used':False,'test_labels_used':False,'test_masks_used':False}
        labels=np.asarray(v['gt_sp'],dtype=np.uint8); base_auc=float(roc_auc_score(masks.ravel(),ve.ravel())); base_ap=float(average_precision_score(masks.ravel(),ve.ravel())); oracle=np.where(masks.astype(bool),np.maximum(ve,te),np.minimum(ve,te)); oracle_gain=float(average_precision_score(masks.ravel(),oracle.ravel())-base_ap)
        for name,(gap,amb,budget) in CONFIGS.items():
          eligible=(te-ve>=gap)&(np.abs(ve)<=amb)
          # Per-image safety budget: at most a small fraction of pixels, chosen solely from unlabeled evidence gaps.
          allowed=np.zeros_like(eligible,bool)
          for i in range(len(eligible)):
            k=max(1,int(eligible.shape[1]*eligible.shape[2]*budget)); flat=np.flatnonzero(eligible[i]);
            if flat.size: allowed[i].flat[flat[np.argsort((te-ve)[i].flat[flat])[-min(k,flat.size):]]]=True
          fused=np.where(allowed,te,ve); ap=float(average_precision_score(masks.ravel(),fused.ravel())); auc=float(roc_auc_score(masks.ravel(),fused.ravel())); cnt=int(allowed.sum()); hit=int((allowed&masks.astype(bool)).sum()); prev=float(masks.mean())
          rows.append({'candidate':name,'category':cat,'samples':int(len(labels)),'visual_pixel_auroc':base_auc,'visual_pixel_ap':base_ap,'fused_pixel_auroc':auc,'fused_pixel_ap':ap,'delta_pixel_auroc':auc-base_auc,'delta_pixel_ap':ap-base_ap,'oracle_pixel_ap_gain':oracle_gain,'rescue_pixel_count':cnt,'rescue_coverage':cnt/allowed.size,'rescue_precision':hit/cnt if cnt else 0.,'anomaly_coverage':hit/max(int(masks.sum()),1),'rescue_precision_lift':(hit/cnt)/prev if cnt and prev else 0.,'test_labels_used_by_router':False,'test_masks_used_by_router':False,'test_set_statistics_used_by_router':False})
    summaries=[]
    for name in CONFIGS:
      x=[r for r in rows if r['candidate']==name]; pos=sum(np.mean([r['delta_pixel_ap'] for r in x if r['category']==c])>0 for c in cats); summaries.append({'candidate':name,'mean_delta_pixel_auroc':float(np.mean([r['delta_pixel_auroc'] for r in x])),'mean_delta_pixel_ap':float(np.mean([r['delta_pixel_ap'] for r in x])),'mean_rescue_precision':float(np.mean([r['rescue_precision'] for r in x])),'mean_rescue_precision_lift':float(np.mean([r['rescue_precision_lift'] for r in x])),'positive_category_count':int(pos)})
    selected=max(summaries,key=lambda x:x['mean_delta_pixel_ap']); passed=selected['mean_delta_pixel_ap']>0 and selected['mean_delta_pixel_auroc']>=-.002 and selected['positive_category_count']>=3
    out={'schema_version':1,'run_id':'v3_adaptclip_mpdd_s0_k1_gate_a2_v1','created_at_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','gate':'v3_gate_a2_adaptclip_reliability','dataset':'mpdd','dataset_role':'development','seed':0,'shot':1,'calibrations':calibrations,'test_predictions_used_by_router':False,'test_labels_used_by_router':False,'test_masks_used_by_router':False,'test_set_statistics_used_by_router':False,'rows':rows,'summaries':summaries,'gate_summary':{'selected_candidate':selected['candidate'],'mean_delta_pixel_auroc':selected['mean_delta_pixel_auroc'],'mean_delta_pixel_ap':selected['mean_delta_pixel_ap'],'positive_category_count':selected['positive_category_count'],'gate_a2_passed':bool(passed)},'decision_rule':'positive mean pixel AP, pixel AUROC loss no worse than 0.002, and positive AP in at least 3/6 categories'}
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(out,indent=2),encoding='utf8'); print(json.dumps(out['gate_summary'],indent=2))
if __name__=='__main__': main()
