"""Generate and evaluate the selected seed-0 router candidate."""
from __future__ import annotations
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from industrial_ad.fusion import BranchPrediction, ConfidenceRouter, load_category_calibrations
from industrial_ad.fusion.alignment import build_alignment_plan
from run_dynamic_fusion_cache import load_cache, resize_maps
CATEGORIES=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-root',type=Path,default=ROOT/'outputs/dynamic_fusion/selected_candidate_20260805'); p.add_argument('--experiment-root',type=Path,default=ROOT/'experiments/dynamic_fusion/selected_candidate_20260805'); a=p.parse_args(); out=a.output_root if a.output_root.is_absolute() else ROOT/a.output_root; exp=a.experiment_root if a.experiment_root.is_absolute() else ROOT/a.experiment_root; out.mkdir(parents=True,exist_ok=True); exp.mkdir(parents=True,exist_ok=True); logs=[]
 for shot in (2,4):
  cal=json.loads((ROOT/f'outputs/dynamic_fusion/normal_reference_predictions/20260804_visa_s0_k{shot}_real_reference_v1_q99/calibration.json').read_text(encoding='utf-8')); target=out/f'k{shot}'; target.mkdir(exist_ok=True)
  for cat in CATEGORIES:
   v=load_cache(ROOT/f'outputs/anomalydino/unified_matrix/seed_0_shot_1/predictions/{cat}.npz'); t=load_cache(ROOT/f'outputs/anomalyclip/visa_all_518_cached/{cat}.npz',ROOT/f'outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified/{cat}.sample_ids.npz'); order=build_alignment_plan(v['sample_ids'],t['sample_ids']).candidate_order; vm=np.asarray(v['anomaly_maps']); vm=vm[:,0] if vm.ndim==4 and vm.shape[1]==1 else vm; tm=np.asarray(t['anomaly_maps'])[order]; tm=tm[:,0] if tm.ndim==4 and tm.shape[1]==1 else tm; tm=resize_maps(tm,vm.shape[1:]); calv,calt=load_category_calibrations(cal,cat); r=ConfidenceRouter(temperature=.5,min_weight=.05,decision_margin=.15,visual_calibration=calv,text_calibration=calt).fuse(BranchPrediction(v['sample_ids'],v['pr_sp'],vm),BranchPrediction(v['sample_ids'],t['pr_sp'][order],tm)); masks=np.asarray(v['imgs_masks']); masks=masks[:,0] if masks.ndim==4 and masks.shape[1]==1 else masks; np.savez_compressed(target/f'{cat}.npz',gt_sp=v['gt_sp'],pr_sp=r.image_scores,imgs_masks=masks,anomaly_maps=r.pixel_maps,sample_ids=r.sample_ids,visual_weights=r.visual_weights,visual_pixel_weights=r.visual_pixel_weights,route_decisions=r.decisions,temperature=np.asarray(.5),decision_margin=np.asarray(.15),min_weight=np.asarray(.05)); del v,t,vm,tm,r
  eval_dir=target/'evaluation'; eval_dir.mkdir(exist_ok=True); cmd=[sys.executable,str(ROOT/'scripts/evaluate_unified.py'),'--cache-dir',str(target),'--output-dir',str(eval_dir),'--workers','1']; logs.append(' '.join(cmd)); subprocess.run(cmd,check=True,cwd=ROOT)
 (exp/'command.txt').write_text('\n'.join(logs),encoding='utf-8'); (exp/'report.json').write_text(json.dumps({'status':'passed','temperature':.5,'decision_margin':.15,'min_weight':.05,'shots':[2,4],'test_predictions_used_by_router':False,'test_labels_used_by_router':False},indent=2),encoding='utf-8'); print('selected candidate evaluation passed')
if __name__=='__main__': main()
