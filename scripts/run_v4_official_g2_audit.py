"""Official SubspaceAD (V2) full G2 audit driver for MPDD (resumable, memory-lean).

Reuses official repo components (extractor / PCAModel / scoring / post_process /
datasets handler) but loads the giant model ONCE and loops over
6 categories x 3 seeds x 1/2/4-shot. References are pinned to
data/splits/mpdd/manifest.json (seed/shot) so results are comparable to the frozen
V4 G2 matrix (matched feature-DINO-only KNN baselines).

6GB/sandbox adaptations (documented in 05_v2_smoke/smoke_protocol.md):
- extractor loads weights via numpy memmap (safetensors >4GB crashes here), fp16.
- P-AUROC / P-AP on a stride-8 subsample of the 672x672 map; AU-PRO full-res but
  computed with preallocated arrays (no internal np.stack) to stay within RAM.
- Resumable: per_config.jsonl rows already present are skipped.

Usage:
  .venv-anomalyclip\\Scripts\\python.exe scripts/run_v4_official_g2_audit.py
      --model-dir %TEMP%\\dinov2_giant_local
      --outdir experiments/dynamic_fusion/v4_vision_text_20260819/06_v2_g2_audit
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import label as cc_label, generate_binary_structure
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO = PROJECT_ROOT / "methods" / "SubspaceAD"
sys.path.insert(0, str(REPO))

from src.subspacead.core.extractor import FeatureExtractor  # noqa: E402
from src.subspacead.core.pca import PCAModel  # noqa: E402
from src.subspacead.data.datasets import get_dataset_handler  # noqa: E402
from src.subspacead.data.transforms import get_augmentation_transform  # noqa: E402
from src.subspacead.post_process.scoring import (  # noqa: E402
    calculate_anomaly_scores,
    post_process_map,
)
from src.subspacead.utils.common import min_max_norm  # noqa: E402


LAYERS = [-12, -13, -14, -15, -16, -17, -18]
OFFICIAL_COMMIT = "ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_all(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def fit_pca(extractor, train_paths, aug_transform, image_res, batch_size, ev, aug_count):
    """Two-pass streaming PCA fit on k-shot references (+augmentations)."""
    all_imgs = []
    for path in train_paths:
        pil = Image.open(path).convert("RGB")
        all_imgs.append(pil)
        if aug_transform is not None:
            for _ in range(aug_count):
                all_imgs.append(aug_transform(pil))
    temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
        [all_imgs[0]], image_res, LAYERS, "mean", docrop=False
    )
    feature_dim = temp_tokens.shape[-1]
    num_aug = len(all_imgs)
    total_tokens = num_aug * h_p * w_p
    num_batches = math.ceil(num_aug / batch_size)

    def generator():
        for i in range(0, num_aug, batch_size):
            if i > 0 and i % 20 == 0:
                torch.cuda.empty_cache()  # defrag allocator cache between cov batches
            batch = all_imgs[i : i + batch_size]
            tokens, _, _ = extractor.extract_tokens(
                batch, image_res, LAYERS, "mean", docrop=False
            )
            yield tokens.reshape(-1, feature_dim)

    pca_model = PCAModel(k=None, ev=ev, whiten=False)
    pca_model.dtype = torch.float32  # official hardcodes float64; fp32 halves cov/batch memory
    pca_params = pca_model.fit(generator, feature_dim, total_tokens, num_batches)
    del all_imgs
    gc.collect()
    return pca_params, (h_p, w_p), feature_dim


def aupro_from_arrays(preds, gts, fpr_limit=0.3, num_thresholds=300, connectivity=8):
    """MVTec-AD AUPRO on preallocated (N,H,W) arrays (no internal stacking)."""
    preds = np.asarray(preds, dtype=np.float32)
    gts = np.asarray(gts, dtype=np.uint8)
    if not np.isfinite(preds).all():
        return float("nan")
    structure = generate_binary_structure(2, 2 if connectivity == 8 else 1)
    region_sorted_scores = []
    for i in range(gts.shape[0]):
        if gts[i].sum() == 0:
            continue
        labeled, n = cc_label(gts[i], structure=structure)
        for r in range(1, n + 1):
            region_mask = labeled == r
            region_sorted_scores.append(np.sort(preds[i][region_mask]))
    if not region_sorted_scores:
        return float("nan")
    neg_scores = preds[gts == 0]
    if neg_scores.size == 0:
        return float("nan")
    neg_sorted = np.sort(neg_scores)
    n_neg = neg_sorted.size
    target_fprs = np.linspace(0.0, fpr_limit, num_thresholds + 1)[1:]
    q_idx = np.clip(np.floor((1.0 - target_fprs) * (n_neg - 1)).astype(np.int64), 0, n_neg - 1)
    thresholds = neg_sorted[q_idx]
    fp_counts = n_neg - np.searchsorted(neg_sorted, thresholds, side="left")
    fprs = fp_counts.astype(np.float64) / n_neg
    pros_accum = np.zeros(num_thresholds, dtype=np.float64)
    for rs in region_sorted_scores:
        area = rs.size
        pros_accum += (rs.size - np.searchsorted(rs, thresholds, side="left")) / area
    pros = pros_accum / len(region_sorted_scores)
    order = np.argsort(fprs, kind="stable")
    fprs_s = np.concatenate([[0.0], fprs[order]])
    pros_s = np.concatenate([[0.0], pros[order]])
    if fprs_s[-1] > fpr_limit:
        cut = np.searchsorted(fprs_s, fpr_limit, side="right")
        f0, f1 = fprs_s[cut - 1], fprs_s[cut]
        p0, p1 = pros_s[cut - 1], pros_s[cut]
        p_at = p0 + (p1 - p0) * (fpr_limit - f0) / (f1 - f0) if f1 > f0 else p0
        fprs_s = np.concatenate([fprs_s[:cut], [fpr_limit]])
        pros_s = np.concatenate([pros_s[:cut], [p_at]])
    elif fprs_s[-1] < fpr_limit:
        fprs_s = np.concatenate([fprs_s, [fpr_limit]])
        pros_s = np.concatenate([pros_s, [pros_s[-1]]])
    return float(np.trapz(pros_s, fprs_s) / fpr_limit)


def evaluate_category(extractor, handler, pca_params, image_res, batch_size, stride=8, cat_name="",
                      export_dir=None, data_root=None, export_meta=None):
    test_paths = handler.get_test_paths()
    n = len(test_paths)
    H = W = image_res
    px_true, px_pred, px_pred_norm = [], [], []
    pro_maps = np.empty((n, H, W), dtype=np.float32)
    pro_gts = np.zeros((n, H, W), dtype=np.uint8)
    img_true, img_pred = [], []
    export_ids, export_raw = [], []   # --export-maps payloads (raw residual grid)
    t_img = time.time()
    for j, p in enumerate(test_paths):
        pil = Image.open(p).convert("RGB")
        is_anomaly = "good" not in str(p) and "Normal" not in str(p)
        tokens, (h_p, w_p), _ = extractor.extract_tokens(
            [pil], image_res, LAYERS, "mean", docrop=False
        )
        scores = calculate_anomaly_scores(
            tokens.reshape(-1, tokens.shape[-1]), pca_params, "reconstruction", 0
        )
        amap = scores.reshape(h_p, w_p)
        amap_final = post_process_map(amap, image_res)
        amap_norm = min_max_norm(amap_final)

        img_true.append(1 if is_anomaly else 0)
        img_pred.append(float(np.max(amap_final)))

        if export_dir is not None:
            export_ids.append(str(Path(p).relative_to(data_root).as_posix()))
            export_raw.append(np.asarray(amap, dtype=np.float16))

        gt = handler.get_ground_truth_mask(p, (W, H))
        gt = (gt > 0).astype(np.uint8)
        px_true.append(gt.flatten()[::stride])
        px_pred.append(amap_final.flatten()[::stride].astype(np.float32))
        px_pred_norm.append(amap_norm.flatten()[::stride].astype(np.float32))
        pro_maps[j] = amap_final
        pro_gts[j] = gt
        del tokens, scores, amap, amap_final, amap_norm, gt
        if j % 10 == 0:
            gc.collect()
        if j % 25 == 0:
            print(f"  [{cat_name}] eval {j}/{n}  {time.time()-t_img:.1f}s", flush=True)
    print(f"  [{cat_name}] eval done {n} imgs in {time.time()-t_img:.1f}s", flush=True)

    if export_dir is not None:
        exp_dir = Path(export_dir)
        exp_dir.mkdir(parents=True, exist_ok=True)
        raw = np.stack(export_raw)                       # (N,h_p,w_p) float16
        meta = export_meta or {}
        refs = [str(Path(r).relative_to(data_root).as_posix())
                for r in meta.get("ref_paths", [])]
        np.savez_compressed(
            exp_dir / f"{meta['category']}_s{meta['seed']}_k{meta['shot']}.npz",
            sample_ids=np.asarray(export_ids),
            amap_raw=raw,
            ref_ids=np.asarray(refs),
            image_res=np.asarray(image_res, dtype=np.int64),
            pca_ev=np.asarray(meta.get("pca_ev", 0.99), dtype=np.float64),
            aug_count=np.asarray(meta.get("aug_count", 30), dtype=np.int64),
            official_commit=np.asarray(meta.get("official_commit", OFFICIAL_COMMIT)),
        )
        print(f"  [{cat_name}] exported {raw.shape} residual maps to {exp_dir}", flush=True)
        del raw

    y_true = np.concatenate(px_true)
    y_pred = np.concatenate(px_pred)
    has_pos, has_neg = bool((y_true == 1).any()), bool((y_true == 0).any())
    px_ap = average_precision_score(y_true, y_pred) if (has_pos and has_neg) else float("nan")
    px_auroc = roc_auc_score(y_true, y_pred) if (has_pos and has_neg) else float("nan")
    del y_true, y_pred, px_true, px_pred, px_pred_norm
    gc.collect()
    aupro = aupro_from_arrays(pro_maps, pro_gts, fpr_limit=0.3, num_thresholds=300, connectivity=8)
    del pro_maps, pro_gts
    gc.collect()
    img_auroc = roc_auc_score(img_true, img_pred) if len(set(img_true)) > 1 else float("nan")
    img_aupr = average_precision_score(img_true, img_pred) if len(set(img_true)) > 1 else float("nan")
    return {
        "image_auroc": img_auroc,
        "image_aupr": img_aupr,
        "pixel_auroc": px_auroc,
        "pixel_ap": px_ap,
        "aupro": aupro,
        "n_test": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default=str(PROJECT_ROOT / "data/mpdd_raw/MPDD"))
    parser.add_argument("--model-dir", required=True, help="dir with config.json+model.safetensors")
    parser.add_argument("--image-res", type=int, default=672)
    parser.add_argument("--aug-count", type=int, default=30)
    parser.add_argument("--pca-ev", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--export-maps", type=Path, default=None,
                        help="optional dir to export raw reconstruction residual grids "
                             "(N,h_p,w_p float16 per config) + sample_ids for DG-SAFE")
    args = parser.parse_args()

    dataset_path = args.dataset_path
    categories = args.categories or sorted(
        f.name for f in Path(dataset_path).iterdir() if f.is_dir() and f.name != "split_csv"
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    per_config_path = outdir / "per_config.jsonl"

    done_rows = {}
    if per_config_path.exists():
        for line in per_config_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
                done_rows[(r["category"], r["seed"], r["shot"])] = r
            except Exception:
                pass
    print(f"Resume: {len(done_rows)} configs already done.", flush=True)

    g2_report = json.loads(
        (PROJECT_ROOT / "experiments/dynamic_fusion/v4_vision_text_20260819/02_visual_gate/g2_matrix/g2_matrix_report.json").read_text(encoding="utf-8")
    )
    baseline_ap, baseline_aupro = {}, {}
    for cfg in g2_report["configs"]:
        if abs(cfg["pca_ev"] - 0.99) > 1e-9:
            continue
        key = (cfg["seed"], cfg["shot"])
        for row in cfg["rows"]:
            baseline_ap[(row["category"], key[0], key[1])] = row["anomalydino"]["pixel_ap"]
            baseline_aupro[(row["category"], key[0], key[1])] = row["anomalydino"]["pixel_aupro"]

    aug_transform = get_augmentation_transform(["rotate"], args.image_res)
    extractor = FeatureExtractor(args.model_dir, half=True, need_saliency=False)

    rows, start_all = [], time.time()
    for cat in categories:
        for seed in args.seeds:
            for shot in args.shots:
                if (cat, seed, shot) in done_rows:
                    rows.append(done_rows[(cat, seed, shot)])
                    continue
                free_g, total_g = torch.cuda.mem_get_info()
                print(
                    f"  [{cat}] s{seed}/k{shot} start  GPU free {free_g/2**20:.0f}/{total_g/2**20:.0f} MiB",
                    flush=True,
                )
                seed_all(seed)
                os.environ["V4_SMOKE_SEED"] = str(seed)
                os.environ["V4_SMOKE_SHOT"] = str(shot)
                handler = get_dataset_handler("mpdd", dataset_path, cat)
                train_paths = handler.get_train_paths()
                t0 = time.time()
                pca_params, _h_pw, _fd = fit_pca(
                    extractor, train_paths, aug_transform, args.image_res,
                    args.batch_size, args.pca_ev, args.aug_count,
                )
                t_fit = time.time() - t0
                print(f"  [{cat}] fit done in {t_fit:.1f}s (seed {seed} shot {shot})", flush=True)
                res = evaluate_category(
                    extractor, handler, pca_params, args.image_res, args.batch_size,
                    cat_name=cat, export_dir=args.export_maps,
                    data_root=Path(dataset_path),
                    export_meta=None if args.export_maps is None else {
                        "category": cat, "seed": seed, "shot": shot,
                        "pca_ev": args.pca_ev, "aug_count": args.aug_count,
                        "official_commit": OFFICIAL_COMMIT,
                        "ref_paths": train_paths,
                    },
                )
                del pca_params
                torch.cuda.empty_cache()
                gc.collect()
                base = baseline_ap.get((cat, seed, shot), float("nan"))
                base_aupro = baseline_aupro.get((cat, seed, shot), float("nan"))
                row = {
                    "category": cat, "seed": seed, "shot": shot, "pca_ev": args.pca_ev,
                    "pixel_ap": res["pixel_ap"], "pixel_auroc": res["pixel_auroc"],
                    "aupro": res["aupro"], "image_auroc": res["image_auroc"],
                    "image_aupr": res["image_aupr"], "n_test": res["n_test"],
                    "matched_dino_knn_pixel_ap": base,
                    "matched_dino_knn_aupro": base_aupro,
                    "delta_ap_vs_dino": (res["pixel_ap"] - base) if base == base else None,
                    "delta_aupro_vs_dino": (res["aupro"] - base_aupro) if base_aupro == base_aupro else None,
                    "fit_seconds": round(t_fit, 1),
                }
                rows.append(row)
                with per_config_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)

    rows = [r for r in rows if r.get("pixel_ap") is not None]  # drop placeholders
    # ---- Gate G2 aggregation (pca_ev = 0.99, 9 configs) ----
    cfg_means = {}
    for seed in args.seeds:
        for shot in args.shots:
            sub = [r for r in rows if r["seed"] == seed and r["shot"] == shot]
            d = [r["delta_ap_vs_dino"] for r in sub if r["delta_ap_vs_dino"] is not None]
            cfg_means[f"s{seed}_k{shot}"] = {
                "mean_delta_ap": float(np.mean(d)),
                "mean_delta_aupro": float(np.mean(
                    [r["delta_aupro_vs_dino"] for r in sub if r["delta_aupro_vs_dino"] is not None]
                )),
                "n_non_negative_categories": int(sum(1 for x in d if x >= 0)),
                "positive_4of6": int(sum(1 for r in sub if (r["delta_ap_vs_dino"] or 0) > 0)),
            }
    deltas = [r["delta_ap_vs_dino"] for r in rows if r["delta_ap_vs_dino"] is not None]
    delta_aupros = [r["delta_aupro_vs_dino"] for r in rows if r["delta_aupro_vs_dino"] is not None]
    worst = {}
    for cat in categories:
        cat_rows = [r for r in rows if r["category"] == cat]
        worst[cat] = float(np.mean([r["delta_ap_vs_dino"] for r in cat_rows]))

    mean_delta_ap = float(np.mean(deltas))
    mean_delta_aupro = float(np.mean(delta_aupros))
    non_neg_configs = sum(1 for c in cfg_means.values() if c["mean_delta_ap"] >= 0)
    pos4_configs = sum(1 for c in cfg_means.values() if c["positive_4of6"] >= 4)
    worst_value = float(np.min(list(worst.values())))
    cond_a = mean_delta_ap >= 0.010
    cond_b = abs(mean_delta_ap) <= 0.003 and mean_delta_aupro >= 0.010
    hard = cond_a or cond_b
    gate = {
        "pca_ev": args.pca_ev,
        "n_configs": len(cfg_means),
        "mean_delta_ap": mean_delta_ap,
        "mean_delta_aupro": mean_delta_aupro,
        "cond_a_delta_ap_ge_0010": cond_a,
        "cond_b_ap_flat_aupro_ge_0010": cond_b,
        "non_negative_configs": non_neg_configs,
        "non_negative_ge_7_of_9": non_neg_configs >= 7,
        "positive_4of6_cat_configs": pos4_configs,
        "positive_cat_ge_7_of_9": pos4_configs >= 7,
        "worst_category_delta_ap": worst_value,
        "worst_category_ge_neg_0020": worst_value >= -0.020,
        "mean_aupro": float(np.mean([r["aupro"] for r in rows])),
        "per_category_mean_delta_ap": worst,
        "configs": cfg_means,
        "gate_passed": bool(hard and non_neg_configs >= 7 and pos4_configs >= 7 and worst_value >= -0.020),
    }
    report = {
        "run_id": "v4_v2_official_g2_audit",
        "official_commit": OFFICIAL_COMMIT,
        "model": "facebook/dinov2-with-registers-giant (fp16, local memmap)",
        "protocol": f"image_res {args.image_res}, aug {args.aug_count}x rotate, pca_ev {args.pca_ev}, "
                    f"agg mean layers -12..-18, reconstruction score, batch {args.batch_size}, manifest-pinned refs",
        "dataset_role": "development",
        "baseline": "matched feature-DINO-only KNN (frozen G2 matrix, pca 0.99 rows)",
        "elapsed_seconds": round(time.time() - start_all, 1),
        "gate": gate,
        "leakage_flags": {
            "test_predictions_used_for_parameter_fit": False,
            "test_labels_used_for_parameter_fit": False,
            "test_masks_used_for_parameter_fit": False,
            "test_dataset_statistics_used_for_calibration": False,
            "test_normal_selection_used": False,
        },
        "rows": rows,
    }
    (outdir / "g2_audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== GATE ===", flush=True)
    print(json.dumps(gate, indent=2), flush=True)


if __name__ == "__main__":
    main()
