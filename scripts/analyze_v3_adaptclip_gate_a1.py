"""Evaluator-only Gate A1 for AdaptCLIP versus the frozen MPDD visual branch."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import cv2
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from skimage import measure


def rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return (rankdata(values, method="average") - 1.0) / max(len(values) - 1.0, 1.0)


def map01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    flat = values.reshape(len(values), -1)
    low, high = np.quantile(flat, .01, axis=1), np.quantile(flat, .99, axis=1)
    return np.clip((values - low[:, None, None]) / np.maximum(high - low, 1e-6)[:, None, None], 0, 1)


def auc_ap(labels: np.ndarray, values: np.ndarray) -> tuple[float, float]:
    return float(roc_auc_score(labels.reshape(-1), values.reshape(-1))), float(average_precision_score(labels.reshape(-1), values.reshape(-1)))


def resize_maps(maps: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if maps.shape[1:] == shape:
        return maps
    return np.stack([
        cv2.resize(item, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        for item in maps
    ]).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--adaptclip-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixel-stride", type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / "data/splits/mpdd/manifest.json").read_text(encoding="utf-8"))
    rows = []
    for category in sorted(manifest["categories"]):
        with np.load(args.visual_root / f"{category}.npz", allow_pickle=False) as visual, np.load(args.adaptclip_root / f"{category}.npz", allow_pickle=False) as text:
            ids_v = np.asarray(visual["sample_ids"]).astype(str)
            ids_t = np.asarray(text["sample_ids"]).astype(str)
            if not np.array_equal(ids_v, ids_t):
                raise ValueError(f"unaligned sample IDs: {category}")
            labels = np.asarray(visual["gt_sp"], dtype=np.uint8)
            if not np.array_equal(labels, np.asarray(text["gt_sp"], dtype=np.uint8)):
                raise ValueError(f"labels differ: {category}")
            v_image, t_image = rank01(visual["pr_sp"]), rank01(text["pr_sp"])
            image_oracle = np.where(labels.astype(bool), np.maximum(v_image, t_image), np.minimum(v_image, t_image))
            v_maps = map01(np.asarray(visual["anomaly_maps"], dtype=np.float32))
            t_maps = map01(resize_maps(np.asarray(text["anomaly_maps"], dtype=np.float32), v_maps.shape[1:]))
            masks = np.asarray(visual["imgs_masks"], dtype=np.uint8)
            stride = args.pixel_stride
            v_maps, t_maps, masks = v_maps[:, ::stride, ::stride], t_maps[:, ::stride, ::stride], masks[:, ::stride, ::stride]
            pixel_oracle = np.where(masks.astype(bool), np.maximum(v_maps, t_maps), np.minimum(v_maps, t_maps))
            regions = better_regions = 0
            for mask, vmap, tmap in zip(masks, v_maps, t_maps):
                connected = measure.label(mask.astype(bool))
                for rid in range(1, int(connected.max()) + 1):
                    selected = connected == rid
                    regions += 1
                    better_regions += int(float(tmap[selected].mean()) > float(vmap[selected].mean()))
            vi_auc, vi_ap = auc_ap(labels, v_image)
            ti_auc, ti_ap = auc_ap(labels, t_image)
            oi_auc, oi_ap = auc_ap(labels, image_oracle)
            vp_auc, vp_ap = auc_ap(masks, v_maps)
            tp_auc, tp_ap = auc_ap(masks, t_maps)
            op_auc, op_ap = auc_ap(masks, pixel_oracle)
            rows.append({"category": category, "samples": int(len(labels)), "visual_image_auroc": vi_auc, "adaptclip_image_auroc": ti_auc, "oracle_image_auroc": oi_auc, "visual_image_ap": vi_ap, "adaptclip_image_ap": ti_ap, "oracle_image_ap": oi_ap, "visual_pixel_auroc": vp_auc, "adaptclip_pixel_auroc": tp_auc, "oracle_pixel_auroc": op_auc, "visual_pixel_ap": vp_ap, "adaptclip_pixel_ap": tp_ap, "oracle_pixel_ap": op_ap, "oracle_image_delta_auroc_vs_visual": oi_auc-vi_auc, "oracle_pixel_delta_ap_vs_visual": op_ap-vp_ap, "anomaly_regions": regions, "adaptclip_better_regions": better_regions, "adaptclip_better_region_fraction": better_regions/max(regions,1)})
    total_regions = sum(r["anomaly_regions"] for r in rows)
    better_regions = sum(r["adaptclip_better_regions"] for r in rows)
    image_gain_categories = sum(r["oracle_image_delta_auroc_vs_visual"] > .005 for r in rows)
    pixel_gain_categories = sum(r["oracle_pixel_delta_ap_vs_visual"] > .005 for r in rows)
    summary = {"categories": len(rows), "mean_oracle_image_delta_auroc_vs_visual": float(np.mean([r["oracle_image_delta_auroc_vs_visual"] for r in rows])), "mean_oracle_pixel_delta_ap_vs_visual": float(np.mean([r["oracle_pixel_delta_ap_vs_visual"] for r in rows])), "image_headroom_positive_categories": image_gain_categories, "pixel_headroom_positive_categories": pixel_gain_categories, "anomaly_region_count": total_regions, "adaptclip_better_region_count": better_regions, "adaptclip_better_region_fraction": better_regions/max(total_regions,1)}
    passed = image_gain_categories >= 2 or (pixel_gain_categories >= 2 and summary["adaptclip_better_region_fraction"] >= .10)
    result = {"schema_version": 1, "run_id": "v3_adaptclip_mpdd_s0_k1_gate_a1_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "gate": "v3_gate_a1_adaptclip_oracle_headroom", "dataset": "mpdd", "dataset_role": "development", "seed": 0, "shot": 1, "analysis_type": "label_informed_evaluator_only_upper_bound", "router_labels_used": False, "router_masks_used": False, "test_set_statistics_used_by_router": False, "pixel_analysis_stride": args.pixel_stride, "rows": rows, "summary": summary, "gate_a1_passed": bool(passed), "decision_rule": "continue only if oracle image AUROC gain >0.005 on at least 2/6 categories, or oracle pixel AP gain >0.005 on at least 2/6 categories and AdaptCLIP is better on at least 10% of anomaly regions"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"gate_a1_passed": result["gate_a1_passed"], **summary}, indent=2))


if __name__ == "__main__":
    main()
