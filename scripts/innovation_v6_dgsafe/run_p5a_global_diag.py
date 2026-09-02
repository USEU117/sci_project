"""P5-A-lite - S1-HGLC first diagnostic step: DINO CLS global image evidence
(task book 16 s.3.3 item 2: compare global scores vs A1 image score Image-AUROC/AP).

Only the DINO v2-B/14 CLS-token part of the S1 route is probed here (cheap, no
AnomalyCLIP/text changes). CLS embeddings of MPDD test images do not depend on
(seed, shot); per-config anomaly scores are built by cosine distance of a test
CLS to the k-shot normal-reference CLS set, evaluated per (cat, seed=0, shot).

Gate (16 s.3.3): a single global score should beat A1's own image score by an
average Image-AP margin of >= +0.010, otherwise archive S1-HGLC image-level part.
A1 image score = max of the frozen concat 448 map (CPU).
All MPDD development role; GT labels evaluator-side only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)
METHOD_ROOT = ROOT / "methods" / "anomalydino"
sys.path.insert(0, str(METHOD_ROOT))

from src.backbones import get_model  # noqa: E402

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402

OUT = maps.EXPERIMENT_ROOT / "p5a_global_diag"
A1_MAPS = maps.A1_MAPS_ROOT
MPDD = maps.MPDD_DATA_ROOT
MANIFEST = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
RES = 448


def load_cls(model, path) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    tensor, _ = model.prepare_image(img)
    with torch.inference_mode():
        x = tensor.unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")
        out = model.model.forward_features(x)
    cls = out["x_norm_clstoken"].squeeze().float().cpu().numpy()
    return cls.astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    maps.assert_development_only()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cats = sorted(p.stem for p in A1_MAPS.glob(f"s{args.seed}_k1/*.npz"))
    model = get_model("dinov2_vitb14", args.device, smaller_edge_size=RES)
    model.model.eval()

    cls_dir = OUT / "cls_cache"
    cls_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for cat in cats:
        # all test images of the category (order = compact map order)
        npz = np.load(A1_MAPS / f"s{args.seed}_k1" / f"{cat}.npz", allow_pickle=False)
        sample_ids = [str(s) for s in npz["sample_ids"]]
        concat = np.asarray(npz["concat_patch_map"], dtype=np.float32)
        a1_img_scores = maps.a1_maps448(concat).reshape(len(concat), -1).max(axis=1)
        labels = np.asarray([1 if "/good/" not in s else 0 for s in sample_ids])
        # cache CLS per test image (config-independent)
        test_cls = np.zeros((len(sample_ids), 768), dtype=np.float64)
        for i, sid in enumerate(sample_ids):
            p = MPDD / sid
            test_cls[i] = load_cls(model, p)
        # references per shot (nested sets); CLS by filename
        ref_cache = {}
        for shot in args.shots:
            refs = manifest["categories"][cat][str(args.seed)][str(shot)]
            ref_cls = np.stack([ref_cache.setdefault(r, load_cls(model, MPDD / r))
                                for r in refs])
            cos = test_cls @ ref_cls.T / (
                np.linalg.norm(test_cls, axis=1, keepdims=True) *
                np.linalg.norm(ref_cls, axis=1) + 1e-12)
            s_global = 1.0 - cos.max(axis=1)      # cosine distance to nearest ref
            from sklearn.metrics import average_precision_score, roc_auc_score
            img_ap_a1 = float(average_precision_score(labels, a1_img_scores))
            img_ap_g = float(average_precision_score(labels, s_global))
            img_auroc_a1 = float(roc_auc_score(labels, a1_img_scores))
            img_auroc_g = float(roc_auc_score(labels, s_global))
            rows.append({"category": cat, "shot": shot, "n_test": len(labels),
                         "a1_image_ap": round(img_ap_a1, 4),
                         "dino_cls_image_ap": round(img_ap_g, 4),
                         "a1_image_auroc": round(img_auroc_a1, 4),
                         "dino_cls_image_auroc": round(img_auroc_g, 4),
                         "delta_image_ap": round(img_ap_g - img_ap_a1, 4)})
            print(f"[{cat}] k{shot} A1_imgAP={img_ap_a1:.4f} "
                  f"CLS_imgAP={img_ap_g:.4f} dAP={img_ap_g - img_ap_a1:+.4f}",
                  flush=True)

    per_cat = {}
    for cat in cats:
        r = [x for x in rows if x["category"] == cat]
        mm = lambda f: float(np.mean([x[f] for x in r]))
        per_cat[cat] = {"a1_image_ap": round(mm("a1_image_ap"), 4),
                        "dino_cls_image_ap": round(mm("dino_cls_image_ap"), 4),
                        "delta_image_ap": round(mm("delta_image_ap"), 4)}
    pool = lambda f: float(np.mean([per_cat[c][f] for c in cats]))
    verdict = {"pooled_a1_image_ap": round(pool("a1_image_ap"), 4),
               "pooled_dino_cls_image_ap": round(pool("dino_cls_image_ap"), 4),
               "pooled_delta_image_ap": round(pool("delta_image_ap"), 4),
               "s1_gate_global_ge_a1_plus_001": bool(pool("delta_image_ap") >= 0.010),
               "note": "gate per task book 16 s3.3 (single global vs A1 image AP)"}
    report = {"program": "innovation_v6_dgsafe", "phase": "p5a_global_diag",
              "dataset": "mpdd", "role": "development", "seed": args.seed,
              "model": "dinov2_vitb14 CLS @448", "rows": rows,
              "per_category": per_cat, "verdict": verdict,
              "created_at_utc": datetime.now(timezone.utc).isoformat()}
    (OUT / "P5A_GLOBAL_DIAG.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
