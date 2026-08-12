"""Fast image-level sensitivity metrics for the pre-registered router sweep."""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from industrial_ad.fusion import BranchPrediction, ConfidenceRouter, load_category_calibrations
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_cache import load_cache, resize_maps
CATEGORIES=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']; TEMPS=(.20,.35,.50); MARGINS=(.05,.10,.15)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',type=Path,default=ROOT/'experiments/dynamic_fusion/20260805_sensitivity'); a=p.parse_args(); out=a.output_dir if a.output_dir.is_absolute() else ROOT/a.output_dir; out.mkdir(parents=True,exist_ok=True); rows=[]
 for shot in (2,4):
  cal=json.loads((ROOT/f'outputs/dynamic_fusion/normal_reference_predictions/20260804_visa_s0_k{shot}_real_reference_v1_q99/calibration.json').read_text(encoding='utf-8'))
  for cat in CATEGORIES:
   with np.load(ROOT/f'outputs/anomalydino/unified_matrix/seed_0_shot_1/predictions/{cat}.npz',allow_pickle=False) as vd: v_ids=vd['sample_ids']; v_sp=vd['pr_sp']; labels=vd['gt_sp']
   with np.load(ROOT/f'outputs/anomalyclip/visa_all_518_cached/{cat}.npz',allow_pickle=False) as td: t_sp=td['pr_sp']
   with np.load(ROOT/f'outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified/{cat}.sample_ids.npz',allow_pickle=False) as sd: t_ids=sd['sample_ids']
   order=build_alignment_plan(v_ids,t_ids).candidate_order; dummy=np.zeros((len(v_sp),1,1),dtype=np.float32); vb=BranchPrediction(v_ids,v_sp,dummy); tb=BranchPrediction(v_ids,t_sp[order],dummy.copy()); calv,calt=load_category_calibrations(cal,cat)
   for temp in TEMPS:
    for margin in MARGINS:
     r=ConfidenceRouter(temperature=temp,min_weight=.05,decision_margin=margin,visual_calibration=calv,text_calibration=calt).fuse(vb,tb); d,c=np.unique(r.decisions,return_counts=True); cm=dict(zip(d.tolist(),c.tolist())); rows.append({'shot':shot,'category':cat,'temperature':temp,'decision_margin':margin,'image_auroc':roc_auc_score(labels,r.image_scores),'image_ap':average_precision_score(labels,r.image_scores),'visual_count':cm.get('visual',0),'text_count':cm.get('text',0),'weighted_count':cm.get('weighted_fusion',0),'image_weight_mean':float(r.visual_weights.mean())})
 with (out/'image_sensitivity_results.csv').open('w',encoding='utf-8',newline='') as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 print(f'wrote {out/"image_sensitivity_results.csv"} ({len(rows)} rows)')
if __name__=='__main__': main()
