"""探索动态融合新方向（VisA 单类协议，同 backbone vitb14）。

方向 A/B/C 的统一实验：在同一 dinov2_vitb14 特征上，
  1. KNN 距离（memory retrieval）异常图
  2. PCA 子空间重建残差（low-rank normal manifold）异常图
  3. 两者逐像素融合（max / mean / 加权和）与逐类 oracle

评估像素级 AUROC / AP / AUPRO，逐类输出，判断互补性与融合增益。

数据源：methods/winclip/datasets/VisA_pytorch/1cls/{cat}/
  train/good/ (正常参考), test/good/ (正常), test/bad/ (异常),
  ground_truth/bad/{stem}.png (逐像素掩码，与 test/bad 同 stem 对齐)。

DINO：本地 torch.hub 缓存 dinov2_vitb14（无网络）。纯探索脚本，结果写入
experiments/dynamic_fusion/explore_visa_fusion/report.json。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import faiss
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_unified import aupro_fast  # noqa: E402

DINOV2_HUB = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
DINOV2_CKPT = Path.home() / ".cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth"

MAP_SIZE = 448
STRIDE = 8
PATCH = 14
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def dists2map(dists: np.ndarray, img_shape: tuple[int, int]) -> np.ndarray:
    d = cv2.resize(dists, (img_shape[1], img_shape[0]), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(d, sigma=4)


def load_dinov2(device: str):
    if DINOV2_HUB.is_dir():
        sys.path.insert(0, str(DINOV2_HUB))
    from dinov2.hub.backbones import dinov2_vitb14

    model = dinov2_vitb14(pretrained=False)
    ckpt = torch.load(str(DINOV2_CKPT), map_location="cpu")
    model.load_state_dict(ckpt, strict=True)
    model.to(device).eval()
    return model


def extract_patches(model, images_rgb: np.ndarray, device: str) -> np.ndarray:
    """images_rgb: (N,H,W,3) uint8 -> (N, h, w, D) float32 patch features."""
    n = images_rgb.shape[0]
    out = []
    with torch.inference_mode():
        for i in range(0, n, 8):
            batch = images_rgb[i : i + 8]
            tensors = []
            for im in batch:
                im = cv2.resize(im, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_LINEAR)
                im = (im.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
                tensors.append(torch.from_numpy(im).permute(2, 0, 1))
            x = torch.stack(tensors).to(device)
            feat = model.forward_features(x)["x_norm_patchtokens"]  # (B, h*w, D)
            feat = feat.reshape(len(tensors), MAP_SIZE // PATCH, MAP_SIZE // PATCH, -1)
            out.append(feat.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_metrics(pixel_maps: np.ndarray, gt_masks: np.ndarray) -> dict:
    maps_strided = pixel_maps[:, ::STRIDE, ::STRIDE]
    masks_strided = gt_masks[:, ::STRIDE, ::STRIDE]
    flat_maps = maps_strided.ravel()
    flat_labels = (masks_strided.ravel() > 0.5).astype(np.int32)
    return {
        "pixel_auroc": float(roc_auc_score(flat_labels, flat_maps)),
        "pixel_ap": float(average_precision_score(flat_labels, flat_maps)),
        "pixel_aupro": float(aupro_fast(masks_strided, maps_strided)),
    }


def knn_maps(feat: np.ndarray, ref: np.ndarray) -> np.ndarray:
    d = feat.shape[-1]
    f = feat.reshape(-1, d).astype(np.float32)
    r = ref.reshape(-1, d).astype(np.float32)
    faiss.normalize_L2(f)
    faiss.normalize_L2(r)
    index = faiss.IndexFlatL2(d)
    index.add(r)
    dist, _ = index.search(f, k=1)
    dist = (dist[:, 0] / 2.0).reshape(feat.shape[0], feat.shape[1], feat.shape[2])
    return np.stack([dists2map(s, (MAP_SIZE, MAP_SIZE)) for s in dist]).astype(np.float32)


def pca_maps(feat: np.ndarray, ref: np.ndarray, pca_ev: float = 0.99, chunk: int = 25) -> np.ndarray:
    """PCA 子空间重建残差（float32、分块，避免 sklearn 的 float64 内存膨胀）。"""
    d = feat.shape[-1]
    n, gh, gw = feat.shape[0], feat.shape[1], feat.shape[2]
    f = feat.reshape(-1, d).astype(np.float32)
    r = ref.reshape(-1, d).astype(np.float32)
    pca = PCA(n_components=pca_ev, svd_solver="full", random_state=0)
    pca.fit(r)
    mean = pca.mean_.astype(np.float32)
    comps = pca.components_.astype(np.float32)  # (k, d)
    maps = np.empty((n, MAP_SIZE, MAP_SIZE), dtype=np.float32)
    step = chunk * gh * gw
    for i in range(0, n, chunk):
        x = f[i * gh * gw : (i + chunk) * gh * gw]
        xc = x - mean
        scores = xc @ comps.T
        recon = scores @ comps + mean
        residual = np.linalg.norm(x - recon, axis=1).astype(np.float32)
        for j in range(min(chunk, n - i)):
            maps[i + j] = dists2map(residual[j * gh * gw : (j + 1) * gh * gw].reshape(gh, gw),
                                    (MAP_SIZE, MAP_SIZE))
        del x, xc, scores, recon, residual
    return maps


def pct_norm(maps: np.ndarray, pct: float = 99.0) -> np.ndarray:
    """全局百分位裁剪 + min-max 到 [0,1]（跨整类所有测试图，保持逐方法排序不变）。"""
    hi = float(np.percentile(maps, pct))
    lo = float(maps.min())
    rng = hi - lo
    if rng <= 1e-12:
        return np.zeros_like(maps, dtype=np.float32)
    return (np.clip(maps, lo, hi) - lo) / rng


def list_files(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def read_rgb(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path,
                    default=ROOT / "methods/winclip/datasets/VisA_pytorch/1cls")
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--n-refs", type=int, default=64, help="正常参考图数量（来自 train/good）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--output-dir", type=Path,
                    default=ROOT / "experiments/dynamic_fusion/explore_visa_fusion")
    args = ap.parse_args()

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.categories:
        categories = args.categories
    else:
        categories = sorted(p.name for p in args.data_root.iterdir() if p.is_dir())

    model = load_dinov2(args.device)

    report_rows = []
    for cat in categories:
        cdir = args.data_root / cat
        train_good = cdir / "train" / "good"
        test_good = cdir / "test" / "good"
        test_bad = cdir / "test" / "bad"
        gt_bad = cdir / "ground_truth" / "bad"

        rng = np.random.default_rng(args.seed)
        train_files = list_files(train_good)
        ref_files = sorted(rng.choice(train_files, size=min(args.n_refs, len(train_files)), replace=False))

        ref_imgs = np.stack([read_rgb(p) for p in ref_files])
        print(f"[{cat}] extracting {len(ref_imgs)} refs ...", flush=True)
        ref_feat = extract_patches(model, ref_imgs, args.device)
        del ref_imgs
        gc.collect()

        good_files = list_files(test_good)
        bad_files = list_files(test_bad)
        test_files = good_files + bad_files
        labels = np.array([0] * len(good_files) + [1] * len(bad_files), dtype=np.int64)
        masks = []
        for f, lab in zip(test_files, labels):
            if lab == 1:
                m = cv2.imread(str(gt_bad / f"{f.stem}.png"), cv2.IMREAD_GRAYSCALE)
                m = cv2.resize(m, (MAP_SIZE, MAP_SIZE), interpolation=cv2.INTER_NEAREST)
                masks.append((m > 0).astype(np.uint8))
            else:
                masks.append(np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.uint8))
        masks = np.stack(masks)

        print(f"[{cat}] extracting {len(test_files)} test ...", flush=True)
        test_imgs = np.stack([read_rgb(p) for p in test_files])
        test_feat = extract_patches(model, test_imgs, args.device)
        del test_imgs
        gc.collect()

        print(f"[{cat}] scoring KNN/PCA ...", flush=True)
        km = knn_maps(test_feat, ref_feat)
        pm = pca_maps(test_feat, ref_feat)
        del test_feat, ref_feat
        gc.collect()

        m_knn = compute_metrics(km, masks)
        m_pca = compute_metrics(pm, masks)

        # 融合（逐像素、归一化后，逐个计算并即时释放）
        kn = pct_norm(km)
        pn = pct_norm(pm)
        corr = float(pearsonr(kn.ravel(), pn.ravel())[0])
        fusions = {}
        fusions["max"] = compute_metrics(np.maximum(kn, pn), masks)
        fusions["mean"] = compute_metrics((kn + pn) / 2.0, masks)
        for w in (0.3, 0.5, 0.7):
            fusions[f"w{w}"] = compute_metrics(w * kn + (1 - w) * pn, masks)

        row = {
            "category": cat,
            "n_refs": int(len(ref_files)),
            "n_test": int(len(test_files)),
            "n_anomaly": int(labels.sum()),
            "knn": m_knn,
            "pca": m_pca,
            "oracle_cat_ap": max(m_knn["pixel_ap"], m_pca["pixel_ap"]),
            "knn_pca_corr": corr,
            "fusion": fusions,
        }
        report_rows.append(row)
        print(json.dumps({
            "category": cat,
            "knn_ap": round(m_knn["pixel_ap"], 4),
            "pca_ap": round(m_pca["pixel_ap"], 4),
            "knn_pca_corr": round(corr, 4),
            "max_ap": round(fusions["max"]["pixel_ap"], 4),
            "mean_ap": round(fusions["mean"]["pixel_ap"], 4),
            "w0.5_ap": round(fusions["w0.5"]["pixel_ap"], 4),
            "oracle_cat_ap": round(max(m_knn["pixel_ap"], m_pca["pixel_ap"]), 4),
        }, ensure_ascii=False), flush=True)
        del km, pm, kn, pn, masks, labels
        gc.collect()
        torch.cuda.empty_cache()

    mean = {}
    for key in ("knn", "pca"):
        mean[key] = {
            "pixel_ap": float(np.mean([r[key]["pixel_ap"] for r in report_rows])),
            "pixel_auroc": float(np.mean([r[key]["pixel_auroc"] for r in report_rows])),
            "pixel_aupro": float(np.mean([r[key]["pixel_aupro"] for r in report_rows])),
        }
    for fname in ("max", "mean", "w0.3", "w0.5", "w0.7"):
        mean[f"fusion_{fname}"] = {
            "pixel_ap": float(np.mean([r["fusion"][fname]["pixel_ap"] for r in report_rows])),
            "pixel_auroc": float(np.mean([r["fusion"][fname]["pixel_auroc"] for r in report_rows])),
            "pixel_aupro": float(np.mean([r["fusion"][fname]["pixel_aupro"] for r in report_rows])),
        }
    mean["oracle_cat_ap"] = float(np.mean([r["oracle_cat_ap"] for r in report_rows]))
    mean["knn_pca_corr"] = float(np.mean([r["knn_pca_corr"] for r in report_rows]))

    report = {
        "run_id": "explore_visa_fusion",
        "backbone": "dinov2_vitb14 (torch.hub cached, no-registers)",
        "map": "dists2map bilinear448+gaussian4, stride8 eval",
        "n_refs": args.n_refs,
        "seed": args.seed,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "categories": report_rows,
        "mean": mean,
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== MEAN ===", flush=True)
    for k, v in mean.items():
        if isinstance(v, dict):
            print(f"  {k:<12} AP={v['pixel_ap']:.4f} AUROC={v['pixel_auroc']:.4f} AUPRO={v['pixel_aupro']:.4f}", flush=True)
        else:
            print(f"  {k:<12} {v:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
