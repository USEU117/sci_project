"""Pre-registered MPDD seed0 pilot for normal-calibrated boost-only SafeTCRR."""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
from industrial_ad.innovation_v6_dgsafe import maps as a1io  # noqa: E402
from industrial_ad.innovation_v8_tcrr_probe import normal_calibrated_region_boost_map, robust01  # noqa: E402

CFG = json.loads((ROOT / "configs/innovation_v9_ncsafe_tcrr/r0_protocol.json").read_text(encoding="utf-8"))
TEST = ROOT / "outputs/dynamic_fusion/innovation_v8_tcrr_probe/text_maps"
REF = ROOT / "outputs/dynamic_fusion/innovation_v9_ncsafe_tcrr/reference_text_maps"
OUT = ROOT / "experiments/dynamic_fusion/innovation_v9_ncsafe_tcrr/R0_mpdd_seed0"
CATS = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]

def m(scores, labels):
    y=labels.ravel().astype(int); s=scores.ravel().astype(float)
    return float(average_precision_score(y,s)), float(roc_auc_score(y,s))

def main():
    a1io.assert_development_only(); OUT.mkdir(parents=True, exist_ok=True)
    frozen={}; total=boosted=0
    for cat in CATS:
        with np.load(TEST/f"{cat}.npz",allow_pickle=False) as z:
            tids=np.asarray(z["sample_ids"]); tm=np.asarray(z["anomaly_maps"],dtype=np.float32)
        test56=np.stack([cv2.resize(x,(448,448),interpolation=cv2.INTER_LINEAR)[::8,::8] for x in tm])
        for shot in CFG["shots"]:
            with np.load(REF/f"s0_k{shot}"/f"{cat}.npz",allow_pickle=False) as z:
                refs=np.asarray(z["anomaly_maps"],dtype=np.float32)
            refs56=np.stack([cv2.resize(x,(448,448),interpolation=cv2.INTER_LINEAR)[::8,::8] for x in refs])
            a=a1io.load_a1_patch_map(cat,0,shot); perm=a1io.align_perm(tids,a["sample_ids"])
            raw=a1io.a1_maps448(a["patch_map"])[:,::8,::8].astype(np.float32)
            prop=np.stack([robust01(x) for x in raw]); txt=test56[perm]
            made={"safe":[],"rotate180":[],"halfroll":[]}; audit=[]
            for i in range(len(raw)):
                variants={"safe":txt[i],"rotate180":np.rot90(txt[i],2),"halfroll":np.roll(txt[i],(28,28),axis=(0,1))}
                for name,t in variants.items():
                    out,aud,_=normal_calibrated_region_boost_map(raw[i],prop[i],t,refs56,quantile=.95,min_cells=4,z_start=3,z_full=6,max_factor=1.5)
                    made[name].append(out)
                    if name=="safe": audit.extend(aud)
            total += len(audit); boosted += sum(x["factor"]>1.0+1e-12 for x in audit)
            frozen[(cat,shot)]={"ids":a["sample_ids"],"a1":raw,**{k:np.stack(v) for k,v in made.items()}}
    rows=[]
    for cat in CATS:
        for shot in CFG["shots"]:
            f=frozen[(cat,shot)]; gt=a1io.gt_masks_for(f["ids"])[:,::8,::8]; bap,bauc=m(f["a1"],gt)
            row={"category":cat,"shot":shot,"a1_pixel_ap":bap,"a1_pixel_auroc":bauc}
            for name in ("safe","rotate180","halfroll"):
                ap,auc=m(f[name],gt); row[f"{name}_delta_pixel_ap"]=ap-bap; row[f"{name}_delta_pixel_auroc"]=auc-bauc
            rows.append(row)
    mean=lambda field,rr=rows: float(np.mean([r[field] for r in rr]))
    gain=mean("safe_delta_pixel_ap"); auc=mean("safe_delta_pixel_auroc")
    cats={c:mean("safe_delta_pixel_ap",[r for r in rows if r["category"]==c]) for c in CATS}
    controls={n:mean(f"{n}_delta_pixel_ap") for n in ("rotate180","halfroll")}; sep=gain-max(controls.values())
    frac=boosted/total if total else 0.; g=CFG["gate"]
    checks={"gain_ge_0003":gain>=g["macro_pixel_ap_gain_ge"],"positive_cats_ge_4":sum(v>0 for v in cats.values())>=g["positive_category_count_ge"],
            "worst_cat_ge_minus001":min(cats.values())>=g["worst_category_macro_pixel_ap_gain_ge"],"auroc_ge_minus0001":auc>=g["macro_pixel_auroc_loss_ge"],
            "control_separation_ge_0001":sep>=g["genuine_gain_minus_best_control_gain_ge"],
            "boost_fraction_in_range":g["boosted_region_fraction_between"][0]<=frac<=g["boosted_region_fraction_between"][1]}
    passed=all(checks.values()); report={"program":CFG["program"],"created_at_utc":datetime.now(timezone.utc).isoformat(),"protocol":CFG,"rows":rows,
      "summary":{"pixel_ap_gain":gain,"pixel_auroc_gain":auc,"category_gain":cats,"control_gain":controls,"separation":sep,"regions":total,"boosted_regions":boosted,"boosted_fraction":frac},"gate_checks":checks,"gate_passed":passed}
    (OUT/"R0_RESULT.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# NC-SafeTCRR v9 R0", "",f"- Pixel-AP gain: {gain:+.6f}",f"- Pixel-AUROC gain: {auc:+.6f}",f"- category gains: {cats}",f"- controls: {controls}; separation {sep:+.6f}",f"- boosted regions: {boosted}/{total} ({frac:.3%})",f"- gate: {'PASS' if passed else 'FAIL / ARCHIVE'}","",*[f"- {k}: {v}" for k,v in checks.items()]]
    (OUT/"R0_DECISION.md").write_text("\n".join(lines),encoding="utf-8"); print("\n".join(lines)); return 0 if passed else 2
if __name__=="__main__": raise SystemExit(main())
