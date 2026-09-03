"""V12 NTOF (doc 22 s3) R0 feature export: illumination-intervention & synthetic-defect
variants of NORMAL MPDD images through the frozen per-branch feature extractors.

Reuses EXACTLY the frozen extraction pipelines used to build the A1 caches:
  - dino : methods/anomalydino get_model('dinov2_vitb14', smaller_edge_size=448) -> 32x32 grid
  - clip : methods/AnomalyCLIP-main ViT-L/14@336px image tower, image_size=518 -> 37x37 grid
Only NORMAL images are touched: the K reference (memory) images get the 15 fit
variants; the category's test/good images get 5 held-out illumination variants and
3 synthetic structure defects. No bad image, no GT, no MVTec AD 2.

Run (GPU, .venv-anomalyclip):
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r2_ntof_export.py --branch clip
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r2_ntof_export.py --branch dino
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v12_new_observables"))
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
sys.path.insert(0, str(ROOT / "methods" / "AnomalyCLIP-main"))

from v2_mpdd_prediction_common import index_dataset, sha256  # noqa: E402
import ntof_render as R  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
DATA_ROOT = ROOT / "data/mpdd_raw/MPDD"
OUT_ROOT = ROOT / "outputs/dynamic_fusion"
CAT_LIST = CATEGORIES


# ------------------------------------------------------------------ extractors

def make_dino_extractor(device: str):
    from src.backbones import get_model

    model = get_model("dinov2_vitb14", device, smaller_edge_size=448)

    def extract(rgb: np.ndarray):
        tensor, grid = model.prepare_image(rgb)
        toks = model.extract_features(tensor).astype(np.float32)
        return toks.reshape(grid[0], grid[1], -1)

    return extract


def make_clip_extractor(device: str):
    from types import SimpleNamespace

    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    design = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9,
              "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device="cpu", design_details=design)
    ck_path = ROOT / "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth"
    pl = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    ck = torch.load(str(ck_path), map_location="cpu")
    pl.load_state_dict(ck["prompt_learner"])
    model.to(device)
    pl.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    preprocess, _ = get_transform(SimpleNamespace(image_size=518))

    def extract(rgb: np.ndarray):
        from PIL import Image

        tensor = preprocess(Image.fromarray(rgb)).reshape(1, 3, 518, 518).to(device)
        with torch.inference_mode():
            _, pf = model.encode_image(tensor, [6, 12, 18, 24], DPAM_layer=20)
        tok = pf[-1][0, 1:, :].float().cpu().numpy()
        side = int(round(tok.shape[0] ** 0.5))
        return tok.reshape(side, side, -1)

    return extract


def load_image(path: Path) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


# ------------------------------------------------------------------ main

def export_category(cat: str, shot: int, extract, manifest: dict, out_dir: Path,
                    mask_dir: Path, limit: int | None, branch: str):
    indexed = index_dataset("mpdd", DATA_ROOT)[cat]
    refs = manifest["categories"][cat]["0"][str(shot)]
    goods = [s for s in indexed if "/good/" in s.sample_id]
    if limit:
        goods = goods[:limit]

    ref_orig = []
    ref_var = []
    for rel in refs:
        x0 = load_image(DATA_ROOT / rel)
        f0 = extract(x0)
        ref_orig.append(f0)
        v = []
        for key in R.REF_KEYS:
            img, _m = R.render_by_key(x0, key, cat, rel)
            v.append(extract(img))
        ref_var.append(np.stack(v))

    good_rel = [s.sample_id for s in goods]
    held_all, syn_all, syn_masks = [], [], []
    for s in goods:
        x0 = load_image(Path(s.image_path))
        hv, sv, sm = [], [], []
        for key in R.HELD_KEYS:
            img, _m = R.render_by_key(x0, key, cat, s.sample_id)
            hv.append(extract(img))
        for key in R.SYN_KEYS:
            img, m = R.render_by_key(x0, key, cat, s.sample_id)
            sv.append(extract(img))
            sm.append(m)
        held_all.append(np.stack(hv))
        syn_all.append(np.stack(sv))
        syn_masks.append(np.stack(sm))

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{cat}.npz",
        ref_rel=np.asarray(refs),
        ref_orig_feat=np.asarray(ref_orig, dtype=np.float32),
        ref_var_feat=np.asarray(ref_var, dtype=np.float32),
        good_rel=np.asarray(good_rel),
        good_held_feat=np.asarray(held_all, dtype=np.float32),
        good_syn_feat=np.asarray(syn_all, dtype=np.float32),
        grid_size=np.asarray(ref_orig[0].shape[:2], dtype=np.int64),
        branch=np.asarray(branch),
    )
    mask_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        mask_dir / f"{cat}.npz",
        good_rel=np.asarray(good_rel),
        syn_masks=np.asarray(syn_masks, dtype=np.uint8),
    )
    return {
        "category": cat, "refs": len(refs), "goods": len(goods),
        "ref_variants": len(R.REF_KEYS), "held_variants": len(R.HELD_KEYS),
        "syn_variants": len(R.SYN_KEYS),
        "output": str((out_dir / f"{cat}.npz").resolve()),
        "sha256": sha256(out_dir / f"{cat}.npz"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("dino", "clip"), required=True)
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out_dir = OUT_ROOT / f"ntof_features_{args.branch}_s0_k{args.shot}"
    mask_dir = OUT_ROOT / f"ntof_syn_masks_s0_k{args.shot}"
    cats = [args.category] if args.category else CAT_LIST

    extract = make_dino_extractor(args.device) if args.branch == "dino" \
        else make_clip_extractor(args.device)

    rows = []
    with torch.inference_mode():
        for cat in cats:
            print(f"[export {args.branch}] {cat}", flush=True)
            r = export_category(cat, args.shot, extract, manifest, out_dir, mask_dir,
                                args.limit, args.branch)
            rows.append(r)
            print(f"    {r['refs']} refs x{r['ref_variants']} + {r['goods']} goods x"
                  f"{r['held_variants'] + r['syn_variants']} -> {r['sha256'][:12]}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": args.branch, "shot": args.shot, "device": args.device,
        "manifest_sha256": sha256(MANIFEST),
        "variant_keys_ref": R.REF_KEYS, "variant_keys_held": R.HELD_KEYS,
        "variant_keys_syn": R.SYN_KEYS,
        "categories": rows,
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
