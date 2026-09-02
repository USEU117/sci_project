"""S0 Wave 0 — identity replay of exported SubspaceAD maps (task book 16 2.2/2.7).

For each exported {cat}_s0_k{shot}.npz (amap_raw 48x48 fp16 + sample_ids):
  1. sample-set/order identity vs frozen A1 compact maps;
  2. rebuild 672 map via official post_process_map, recompute pooled Pixel-AP
     (672, stride-8 flatten) and compare with the frozen audit per_config.jsonl;
     pass = per-category |delta| <= 5e-4.

GT is loaded evaluator-side from MPDD ground-truth dirs; never touches the model.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "methods" / "SubspaceAD")):
    if p not in sys.path:
        sys.path.insert(0, p)

from industrial_ad.innovation_v6_dgsafe import maps  # noqa: E402
from src.subspacead.post_process.scoring import post_process_map  # noqa: E402

AUDIT_PER_CONFIG = (ROOT / "experiments/dynamic_fusion/v4_vision_text_20260819"
                    / "06_v2_g2_audit/per_config.jsonl")
OUT_ROOT = maps.EXPERIMENT_ROOT / "Wave0_replay"
TOL = maps.PROTOCOL["identity_replay_tolerance_pixel_ap"]
RES672 = 672
STRIDE = 8
MPDD_DATA_ROOT = ROOT / "data" / "mpdd_raw" / "MPDD"


def audit_pixel_ap() -> dict:
    out = {}
    for line in AUDIT_PER_CONFIG.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if int(r["seed"]) == 0:
            out[(r["category"], int(r["shot"]))] = r["pixel_ap"]
    return out


def gt_masks_official_pil(sample_ids, res=(RES672, RES672)):
    """Replicate the frozen audit evaluator GT path (official handler: PIL open,
    convert L, NEAREST resize to (W,H)=(res)) so the replay reproduces the audit
    numbers bit-for-bit (cv2 INTER_NEAREST differs on non-integer downscales)."""
    from PIL import Image as PILImage
    out = np.zeros((len(sample_ids), res[1], res[0]), dtype=np.uint8)
    for i, sid in enumerate(sample_ids):
        parts = Path(sid).parts
        if len(parts) < 4 or parts[1] != "test":
            raise ValueError(f"unexpected sample_id: {sid}")
        cat, defect, stem = parts[0], parts[2], Path(parts[3]).stem
        if defect == "good":
            continue
        p = Path(MPDD_DATA_ROOT) / cat / "ground_truth" / defect / f"{stem}_mask.png"
        if not p.exists():
            raise FileNotFoundError(p)
        mask = PILImage.open(p).convert("L").resize((res[1], res[0]),
                                                    PILImage.Resampling.NEAREST)
        out[i] = (np.asarray(mask) > 0).astype(np.uint8)
    return out


def main() -> int:
    maps.assert_development_only()
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", type=Path, default=maps.EXPERIMENT_ROOT / "sub_maps_s0")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ref = audit_pixel_ap()
    cats = sorted({k[0] for k in ref})
    rows, t0 = [], time.time()
    for cat in cats:
        for shot in (1, 2, 4):
            sub = maps.load_sub_raw(args.export_dir, 0, shot, cat)
            # 1) sample identity vs frozen A1 compact map
            a1 = maps.load_a1_patch_map(cat, 0, shot)
            perm = maps.align_perm(sub["sample_ids"], a1["sample_ids"])
            if not np.array_equal(sub["sample_ids"][perm], a1["sample_ids"]):
                raise RuntimeError(f"sample order mismatch {cat} s0/k{shot}")
            amap_raw = sub["amap_raw"][perm]               # order == A1 compact
            # 2) rebuild 672 map and recompute pooled Pixel-AP (audit protocol)
            y_parts, s_parts = [], []
            gt672 = gt_masks_official_pil(sub["sample_ids"][perm])
            for k in range(len(amap_raw)):
                amap_final = post_process_map(np.asarray(amap_raw[k], dtype=np.float32), RES672)
                s_parts.append(amap_final.flatten()[::STRIDE])
                y_parts.append(gt672[k].flatten()[::STRIDE])
            y = np.concatenate(y_parts).astype(np.int64)
            s = np.concatenate(s_parts).astype(np.float64)
            from sklearn.metrics import average_precision_score
            replay_ap = float(average_precision_score(y, s))
            ref_ap = ref[(cat, shot)]
            rows.append({"category": cat, "shot": shot,
                         "audit_pixel_ap": ref_ap, "replay_pixel_ap": replay_ap,
                         "abs_diff": round(abs(ref_ap - replay_ap), 6),
                         "n_samples": int(len(amap_raw))})
    # per-category mean |diff| (doc: per-class error <= 5e-4)
    per_cat = {}
    ok_all = True
    for cat in cats:
        d = [r["abs_diff"] for r in rows if r["category"] == cat]
        m = float(np.mean(d))
        per_cat[cat] = round(m, 6)
        ok_all = ok_all and m <= TOL
    report = {
        "program": "innovation_v6_dgsafe", "phase": "Wave0_identity_replay",
        "dataset": "mpdd", "role": "development", "seed": 0,
        "protocol": "exported amap_raw(48) -> official post_process_map(672) -> "
                    "pooled Pixel-AP flatten[::8] vs frozen audit per_config",
        "tolerance": TOL,
        "passed": ok_all, "per_category_mean_abs_diff": per_cat,
        "rows": rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (OUT_ROOT / "WAVE0_REPLAY.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Wave0 replay passed={ok_all} (tol {TOL})")
    for cat in cats:
        print(f"  {cat}: mean|diff|={per_cat[cat]}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
