"""P1-C: steady-state end-to-end efficiency benchmark for A1 (MVTec bottle s0/k1).

Standalone (does NOT modify any file under the frozen freeze manifest).

Modes (one process per branch, then a merge step), mirroring smoke_a1_one_class_one_image.py:
  --mode dino    (.venv-patchcore):     DINOv2 vitb14 448px feature extraction
  --mode clip    (.venv-anomalyclip):   AnomalyCLIP ViT-L/14@336 518px feature extraction
  --mode concat  (.venv-patchcore):     align + concat + KNN(k=1) scoring only (no model load)

Each branch: load model once, warm up 3 passes, then N repeated single-image passes.
Records per-pass wall time (mean/std/P50/P95), peak VRAM (torch) and peak process
working set (Windows GetProcessMemoryInfo via ctypes). concat mode reuses the features
saved by the two branch modes and times the align+concat+KNN pipeline only.

Output: outputs/p1_c_benchmark/{dino,clip,concat}_benchmark.json
"""

from __future__ import annotations

import argparse
import ctypes
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

N_REPEAT = 30
WARMUP = 3


# ----------------------------------------------------------------------
# Windows peak working-set (process peak RAM) via ctypes
# ----------------------------------------------------------------------
class _MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def _get_memory_counters() -> _MEMORY_COUNTERS | None:
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_MEMORY_COUNTERS), ctypes.c_ulong
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        counters = _MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(_MEMORY_COUNTERS)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters if ok else None
    except Exception:  # pragma: no cover - non-Windows fallback
        return None


def working_set_mb() -> float:
    counters = _get_memory_counters()
    return counters.WorkingSetSize / (1024 * 1024) if counters else 0.0


def peak_working_set_mb() -> float:
    counters = _get_memory_counters()
    return counters.PeakWorkingSetSize / (1024 * 1024) if counters else 0.0


def summarize(times: list[float]) -> dict:
    ts = sorted(times)
    p50 = ts[len(ts) // 2]
    p95 = ts[min(len(ts) - 1, int(len(ts) * 0.95))]
    return {
        "n_repeats": len(times),
        "mean_seconds": round(statistics.mean(times), 4),
        "std_seconds": round(statistics.stdev(times), 4) if len(times) > 1 else 0.0,
        "p50_seconds": round(p50, 4),
        "p95_seconds": round(p95, 4),
        "min_seconds": round(min(times), 4),
        "max_seconds": round(max(times), 4),
        "throughput_images_per_second": round(1.0 / statistics.mean(times), 3),
    }


def pick_bottle_io(manifest_path: Path, data_root: Path) -> tuple[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs = manifest["categories"]["bottle"]["0"]["1"]
    ref_id = str(refs[0])
    # First anomalous test image (mask present) for a realistic single-image pass.
    from v2_mpdd_prediction_common import index_dataset

    indexed = index_dataset("mvtec", data_root)
    sample = next((s for s in indexed["bottle"] if s.label == 1), indexed["bottle"][0])
    return ref_id, sample.sample_id


# ----------------------------------------------------------------------
# branch modes
# ----------------------------------------------------------------------
def run_dino(args: argparse.Namespace) -> int:
    import cv2
    import torch

    sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
    from src.backbones import get_model

    ref_id, sample_id = pick_bottle_io(args.manifest, args.data_root)
    model = get_model(args.model_name, args.device, smaller_edge_size=args.resolution)

    def extract(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        tensor, grid = model.prepare_image(image_rgb)
        tokens = model.extract_features(tensor).astype(np.float32)
        return tokens.reshape(grid[0], grid[1], -1), grid

    ref_rgb = cv2.cvtColor(cv2.imread(str(args.data_root / ref_id)), cv2.COLOR_BGR2RGB)
    test_rgb = cv2.cvtColor(cv2.imread(str(args.data_root / sample_id)), cv2.COLOR_BGR2RGB)

    # warm up
    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        extract(test_rgb)

    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        test_feat, grid = extract(test_rgb)
        times.append(time.perf_counter() - t0)

    ref_feat, ref_grid = extract(ref_rgb)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    out_dir = args.output_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "dino_bottle.npz",
        patch_features=test_feat.astype(np.float32)[None, ...],
        ref_patch_features=ref_feat.astype(np.float32)[None, ...],
        grid_size=np.asarray(grid, dtype=np.int64),
        sample_ids=np.asarray([sample_id]),
        gt_sp=np.asarray([1], dtype=np.int64),
        imgs_masks=np.zeros((1, 448, 448), dtype=np.uint8),
    )
    report = {
        "mode": "dino",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": "bottle", "seed": 0, "shot": 1,
        "model": args.model_name, "resolution": args.resolution,
        "sample_id": sample_id, "ref_id": ref_id,
        "peak_vram_mb": round(peak_vram, 2),
        "peak_ram_mb": round(peak_working_set_mb(), 1),
        "latency": summarize(times),
    }
    (args.output_dir / "dino_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def run_clip(args: argparse.Namespace) -> int:
    import torch

    from types import SimpleNamespace

    from PIL import Image

    sys.path.insert(0, str(ROOT / "methods" / "AnomalyCLIP-main"))
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    ref_id, sample_id = pick_bottle_io(args.manifest, args.data_root)

    design = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.text_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=args.device, design_details=design)
    model.eval()
    preprocess, _ = get_transform(SimpleNamespace(image_size=args.image_size))
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(args.device)
    model.to(args.device)
    model.visual.DAPM_replace(DPAM_layer=20)

    def extract(image_rgb: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        with Image.fromarray(image_rgb) as opened:
            tensor = preprocess(opened.convert("RGB"))
        image = tensor.reshape(1, 3, args.image_size, args.image_size).to(args.device)
        with torch.inference_mode():
            _, patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
        patch_token = patch_features[-1][0, 1:, :].float().cpu().numpy()
        seq_len = patch_token.shape[0]
        side = int(round(seq_len**0.5))
        if side * side != seq_len:
            raise RuntimeError(f"non-square patch sequence: {seq_len}")
        return patch_token.reshape(side, side, -1), (side, side)

    with Image.open(str(args.data_root / ref_id)) as opened:
        ref_rgb = np.asarray(opened.convert("RGB"))
    with Image.open(str(args.data_root / sample_id)) as opened:
        test_rgb = np.asarray(opened.convert("RGB"))

    torch.cuda.reset_peak_memory_stats()
    for _ in range(WARMUP):
        extract(test_rgb)

    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        test_feat, grid = extract(test_rgb)
        times.append(time.perf_counter() - t0)

    ref_feat, ref_grid = extract(ref_rgb)
    peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0
    out_dir = args.output_dir / "features"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "clip_bottle.npz",
        patch_features=test_feat.astype(np.float32)[None, ...],
        ref_patch_features=ref_feat.astype(np.float32)[None, ...],
        grid_size=np.asarray(grid, dtype=np.int64),
        sample_ids=np.asarray([sample_id]),
        gt_sp=np.asarray([1], dtype=np.int64),
        imgs_masks=np.zeros((1, 448, 448), dtype=np.uint8),
    )
    report = {
        "mode": "clip",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": "bottle", "seed": 0, "shot": 1,
        "model": "AnomalyCLIP ViT-L/14@336px", "image_size": args.image_size,
        "sample_id": sample_id, "ref_id": ref_id,
        "peak_vram_mb": round(peak_vram, 2),
        "peak_ram_mb": round(peak_working_set_mb(), 1),
        "latency": summarize(times),
    }
    (args.output_dir / "clip_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


# ----------------------------------------------------------------------
# concat + KNN scoring mode (CPU; reuses saved branch features)
# ----------------------------------------------------------------------
def run_concat(args: argparse.Namespace) -> int:
    from evaluate_a1_feature_fusion import fuse_category, load_features

    feat_dir = args.output_dir / "features"
    dino = load_features(feat_dir / "dino_bottle.npz")
    clip = load_features(feat_dir / "clip_bottle.npz")

    map_size = (448, 448)
    # warm up
    for _ in range(WARMUP):
        fuse_category(dino, clip, "concat", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)

    times = []
    for _ in range(N_REPEAT):
        t0 = time.perf_counter()
        fuse_category(dino, clip, "concat", pca_dim=0, whiten=False, map_size=map_size, dino_weight=0.5)
        times.append(time.perf_counter() - t0)

    report = {
        "mode": "concat_knn",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "category": "bottle", "seed": 0, "shot": 1,
        "knn_k": 1, "pca_dim": 0, "whiten": False, "dino_weight": 0.5, "map_size": list(map_size),
        "peak_ram_mb": round(peak_working_set_mb(), 1),
        "latency": summarize(times),
    }
    (args.output_dir / "concat_benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dino", "clip", "concat"), required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/splits/mvtec/manifest.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mvtec")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/p1_c_benchmark")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--features-list", type=int, nargs="+", default=[6, 12, 18, 24])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--text-n-ctx", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "dino":
        return run_dino(args)
    if args.mode == "clip":
        return run_clip(args)
    return run_concat(args)


if __name__ == "__main__":
    raise SystemExit(main())
