"""Read-only support renderer audit; no model fit, no test labels."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import cv2
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/innovation_v14_decisive_validation_20260905'))
sys.path.insert(0,str(ROOT/'scripts/innovation_v12_new_observables'))
from v14_common import CATEGORIES, DATA_ROOT, support_paths, assert_fit_ids_are_support
from ntof_render import render_synthetic
cv2.setNumThreads(1)
OUT=ROOT/'experiments/dynamic_fusion/innovation_followup_20260908/synthetic_audit'
def seed(cat,rel,kind,i):
    h=0
    for ch in f'v14::{cat}::{rel}::{kind}::s{i}': h=(h*131+ord(ch))&0xffffffff
    return h
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    p=dict(shot=2,categories=CATEGORIES,families=['cutpaste','local_erasure','thin_scratch'],
           seeds=[0,1,2],pixel_thresholds=[0,2,8],role='support-only label-to-actual-change audit; no detector evaluation')
    (OUT/'PROTOCOL.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    rows=[]
    for cat in CATEGORIES:
        refs=support_paths(cat,2); assert_fit_ids_are_support(refs,cat,2)
        with np.load(ROOT/'outputs/dynamic_fusion/v14_p1_support/v14_p1_support_dino_s0_k2'/f'{cat}.npz',allow_pickle=False) as z:
            assert z['ref_rel'].astype(str).tolist()==refs
            masks=z['syn_masks']
        for h,rel in enumerate(refs):
            x=cv2.cvtColor(cv2.imread(str(DATA_ROOT/rel)),cv2.COLOR_BGR2RGB)
            assert x.shape==(1024,1024,3)
            for f,kind in enumerate(p['families']):
                for i in p['seeds']:
                    y,m=render_synthetic(x,kind,seed(cat,rel,kind,i))
                    assert np.array_equal(m,masks[h,f*3+i])
                    pos=m>0
                    delta=np.abs(y.astype(np.int16)-x.astype(np.int16)).max(axis=-1)
                    rows.append(dict(category=cat,ref=rel,family=kind,seed=i,mask_pixels=int(pos.sum()),
                        mask_sha256=hashlib.sha256(m.tobytes()).hexdigest(),mask_parity=True,
                        changed_outside=int(np.count_nonzero(delta[~pos])),
                        mean_contrast=float(delta[pos].mean()),
                        **{f'coverage_gt_{t}':float((delta[pos]>t).mean()) for t in p['pixel_thresholds']}))
        print(cat,flush=True)
    summary=[]
    for kind in p['families']:
        rs=[r for r in rows if r['family']==kind]
        summary.append(dict(family=kind,n=len(rs),mean_contrast=float(np.mean([r['mean_contrast'] for r in rs])),
                            **{f'mean_coverage_gt_{t}':float(np.mean([r[f'coverage_gt_{t}'] for r in rs])) for t in p['pixel_thresholds']},
                            min_coverage_gt8=min(r['coverage_gt_8'] for r in rs),
                            n_below_half_gt8=sum(r['coverage_gt_8']<.5 for r in rs)))
    z=dict(protocol=p,rows=rows,summary=summary,total_unique=len(rows),outside_changes=sum(r['changed_outside'] for r in rows))
    (OUT/'RESULTS.json').write_text(json.dumps(z,indent=2),encoding='utf-8')
    lines=['# 合成标注与实际图像变化审计','',
           '6类k2，108个unique synthetic masks，全部重新渲染并与缓存逐字节核验。阈值为每像素最大通道差的固定0/2/8灰度级，仅诊断对比度，不替换原mask。','',
           '|族|n|平均对比度|差>0覆盖|差>2覆盖|差>8覆盖|>8覆盖低于50%的episode|',
           '|---|---:|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['family']}|{r['n']}|{r['mean_contrast']:.3f}|{r['mean_coverage_gt_0']:.4f}|{r['mean_coverage_gt_2']:.4f}|{r['mean_coverage_gt_8']:.4f}|{r['n_below_half_gt8']}|")
    lines+=['',f"mask外变化像素总数：{z['outside_changes']}。",'',
        '代码层面的覆盖边界：thin_scratch仅暗化110、方向45–135度、2–4厚度参数；erasure为近中灰噪声；cutpaste限定从左半图拷到右半图。这些配方的AP不能代表全部真实划痕、擦除或装配缺陷。',
        '后续入口：固定新协议测试亮/暗反转、低对比度、不同方向的留出合成扰动，检验分辨率收益是否依赖单一渲染配方。不得按test mask位置放置合成缺陷。',
        '标注与像素变化一致不等于物理缺陷真实性，也未检查物体/背景语义归属。']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
