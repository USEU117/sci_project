"""Support-only measurement audit; no feature fitting or candidate selection."""
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/dynamic_fusion/innovation_overnight_20260908/resolution_audit'
CACHE = ROOT / 'outputs/dynamic_fusion/v14_p1_support'
CATS = ['bracket_black', 'bracket_brown', 'bracket_white', 'connector', 'metal_plate', 'tubes']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = dict(role='support synthetic mask measurement only', shots=[2,4], grids=[16,32,64],
                    thresholds=dict(majority=0.5, any_overlap=0),
                    note='any-overlap only diagnoses visibility; it does not replace pixel-level ground truth or prove model improvement')
    (OUT/'PROTOCOL.json').write_text(json.dumps(protocol, indent=2), encoding='utf-8')
    rows=[]
    sources=[]
    start=time.time()
    for shot in protocol['shots']:
        for cat in CATS:
            path=CACHE/f'v14_p1_support_dino_s0_k{shot}'/f'{cat}.npz'
            if not path.exists():
                sources.append(dict(path=str(path), status='missing'))
                continue
            with np.load(path, allow_pickle=False) as z:
                refs=z['ref_rel'].astype(str)
                assert all('/test/' not in '/'+r.replace('\\','/') for r in refs)
                masks=z['syn_masks']
                kinds=z['syn_kinds'].astype(str).tolist()
                seeds=int(z['syn_seeds'])
                sources.append(dict(path=str(path.relative_to(ROOT)), refs=refs.tolist(), shape=list(masks.shape),
                                    mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest()))
                for h in range(len(refs)):
                    for e in range(masks.shape[1]):
                        m=(masks[h,e]>0).astype(np.float32)
                        for grid in protocol['grids']:
                            fraction=cv2.resize(m,(grid,grid),interpolation=cv2.INTER_AREA)
                            old=cv2.resize(masks[h,e].astype(np.uint8),(grid,grid),interpolation=cv2.INTER_AREA)>127
                            majority=fraction>0.5
                            overlap=fraction>0
                            rows.append(dict(shot=shot,category=cat,ref=refs[h],episode=e,family=kinds[e//seeds],grid=grid,
                                source_pixels=int(m.sum()),source_fraction=float(m.mean()),majority_cells=int(majority.sum()),
                                legacy_cells=int(old.sum()),overlap_cells=int(overlap.sum()),max_cell_fraction=float(fraction.max()),
                                majority_area_ratio=float(majority.mean()/m.mean()) if m.mean() else None))
            print(f'done k{shot} {cat}',flush=True)
    summary=[]
    for shot in protocol['shots']:
        for family in ['cutpaste','local_erasure','thin_scratch']:
            for grid in protocol['grids']:
                rs=[r for r in rows if r['shot']==shot and r['family']==family and r['grid']==grid]
                if not rs: continue
                summary.append(dict(shot=shot,family=family,grid=grid,n=len(rs),
                    nonempty_source=sum(r['source_pixels']>0 for r in rs),
                    visible_majority=sum(r['majority_cells']>0 for r in rs),
                    visible_legacy=sum(r['legacy_cells']>0 for r in rs),
                    visible_any_overlap=sum(r['overlap_cells']>0 for r in rs),
                    mean_max_cell_fraction=float(np.mean([r['max_cell_fraction'] for r in rs]))))
    (OUT/'RESULTS.json').write_text(json.dumps(dict(protocol=protocol,sources=sources,summary=summary,rows=rows,elapsed_s=time.time()-start),indent=2),encoding='utf-8')
    lines=['# 分辨率与合成小缺陷可评价性实测','',
           '只读取normal support合成mask；没有拟合，没有test图。各shot有重复support，不视为独立样本。', '',
           '|shot|族|网格|样本|原图非空|旧阈值可见|多数覆盖可见|任意覆盖可见|',
           '|---|---|---|---|---|---|---|---|']
    for r in summary:
        lines.append(f"|{r['shot']}|{r['family']}|{r['grid']}|{r['n']}|{r['nonempty_source']}|{r['visible_legacy']}|{r['visible_majority']}|{r['visible_any_overlap']}|")
    lines += ['', '结论边界：mask在网格上消失意味着该口径不能检验此类缺陷，不能据此判断提取器是否含有信号。',
              '任意覆盖会改变目标定义，不能直接替换旧指标并宣称提升；后续应在原始/统一像素mask上评价上采样预测，并单列细小缺陷。',
              '创新入口：在匹配前做图像裁块重编码，是否比仅插值预测/特征真正增加可分性。需相同输入预算对照。', '',
              '复现：`.venv-anomalyclip/Scripts/python.exe scripts/innovation_overnight_20260908/audit_resolution.py`']
    (OUT/'REPORT_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__': main()
