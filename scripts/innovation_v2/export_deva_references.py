"""A2 innovation_v2 — Route D (DEVA) normal-reference augmentation export.

Task book section 8.2. Augments ONLY the normal reference images of MPDD
(seed0 x shot {1,2,4}) and re-extracts frozen DINOv2 + AnomalyCLIP features,
inverse-warped back to the original grid. Cache layout
(outputs/dynamic_fusion/innovation_v2_deva/):

    <category>_<pack>.npz  -> aug_d (N,H,W,768), aug_c (N,37,37,768),
                             valid_d (N,H,W), valid_c (N,37,37),
                             source_ref (N,), identity_d (1,H,W,768),
                             identity_c (1,37,37,768)
    manifest.json         -> per (category, pack, transform, ref) metadata

GPU batch=1; only normal references are re-exported (never test images).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
METHOD_DINO = ROOT / "methods" / "anomalydino"
METHOD_CLIP = ROOT / "methods" / "AnomalyCLIP-main"
for p in (str(METHOD_DINO), str(METHOD_CLIP), str(ROOT / "src"), str(ROOT / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

# Load the DINO backbone and the AnomalyCLIP preprocess via explicit file paths
# to avoid the `src` / `utils` name clashes with <repo>/src.
import importlib.util  # noqa: E402

def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_dino_backbones = _load_module("adino_backbones", METHOD_DINO / "src" / "backbones.py")
get_model = _dino_backbones.get_model
_aclip_utils = _load_module("aclip_utils", METHOD_CLIP / "utils.py")

from industrial_ad.innovation_v2 import common  # noqa: E402
from industrial_ad.innovation_v2.equivariant_augmentation import (  # noqa: E402
    apply_transform, inverse_warp_features, make_transforms,
)

DINO_SIZE = 448
CLIP_SIZE = 518
STRIDE = 14.0  # DINO grid: 448/32 ; CLIP grid: 518/37

CHECKPOINT = ROOT / "methods" / "AnomalyCLIP-main" / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth"


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_clip(device: str):
    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner

    design = {"Prompt_length": 12, "learnabel_text_embedding_depth": 9,
              "learnabel_text_embedding_length": 4}
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=design)
    model.eval()
    preprocess, _ = _aclip_utils.get_transform(SimpleNamespace(image_size=CLIP_SIZE))
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), design)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    return model, preprocess


def clip_extract(model, preprocess, image_rgb: np.ndarray, device: str) -> np.ndarray:
    img = Image.fromarray(image_rgb)
    tensor = preprocess(img).unsqueeze(0).to(device)
    with torch.inference_mode():
        _, patch_features = model.encode_image(tensor, [6, 12, 18, 24], DPAM_layer=20)
    # deepest requested layer, drop [CLS], keep HxW patch tokens
    tokens = patch_features[-1][0, 1:, :].float().cpu().numpy()
    seq = tokens.shape[0]
    side = int(round(seq ** 0.5))
    if side * side != seq:
        raise RuntimeError(f"non-square patch sequence: {seq}")
    return tokens.reshape(side, side, -1).astype(np.float32)


def dino_extract(model, image_rgb: np.ndarray, device: str) -> np.ndarray:
    tensor, grid = model.prepare_image(image_rgb)
    tokens = model.extract_features(tensor).astype(np.float32)
    return tokens.reshape(*grid, -1), grid


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "mpdd_raw" / "MPDD")
    parser.add_argument("--out-dir", type=Path,
                        default=ROOT / "outputs" / "dynamic_fusion" / "innovation_v2_deva")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--category", default=None, help="restrict to one category")
    parser.add_argument("--packs", nargs="*", default=["geometry", "photometric", "combined"])
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if not CHECKPOINT.is_file():
        raise SystemExit(f"missing AnomalyCLIP checkpoint: {CHECKPOINT}")

    setup_seed(111)
    dino_model = get_model("dinov2_vitb14", args.device, smaller_edge_size=DINO_SIZE)
    clip_model, preprocess = load_clip(args.device)

    manifest = common.manifest_for("mpdd")
    cfg = common.load_config(ROOT / "configs" / "innovation_v2" / "route_d_deva.yaml")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    transforms = {p: make_transforms(p, cfg) for p in args.packs}
    meta_rows = []

    categories = [args.category] if args.category else sorted(manifest["categories"].keys())
    for cat in categories:
        for shot in (1, 2, 4):
            refs = common.reference_ids_for(manifest, cat, 0, shot)
            # identity re-extraction for this (cat, shot): frozen-check + equivariance refs
            d_ids, c_ids = [], []
            for ri, relative in enumerate(refs):
                native = cv2.cvtColor(cv2.imread(str(args.data_root / relative)),
                                      cv2.COLOR_BGR2RGB)
                d_id, _ = dino_extract(dino_model, native, args.device)
                img518 = cv2.resize(native, (CLIP_SIZE, CLIP_SIZE),
                                    interpolation=cv2.INTER_LINEAR)
                c_id = clip_extract(clip_model, preprocess, img518, args.device)
                d_ids.append(d_id)
                c_ids.append(c_id)
            np.savez_compressed(
                args.out_dir / f"{cat}_k{shot}_identity.npz",
                identity_d=np.stack(d_ids).astype(np.float32),
                identity_c=np.stack(c_ids).astype(np.float32))
            for pack in args.packs:
                dino_blocks = []
                clip_blocks = []
                vd_blocks = []
                vc_blocks = []
                src_refs = []
                for ri, relative in enumerate(refs):
                    native = cv2.cvtColor(cv2.imread(str(args.data_root / relative)),
                                          cv2.COLOR_BGR2RGB)
                    for t in transforms[pack]:
                        img448 = cv2.resize(native, (DINO_SIZE, DINO_SIZE),
                                            interpolation=cv2.INTER_LINEAR)
                        aug448 = apply_transform(img448, t, DINO_SIZE)
                        d_aug, _ = dino_extract(dino_model, aug448, args.device)
                        wd, vd = inverse_warp_features(
                            d_aug, cv2.invertAffineTransform(t.M_at(float(DINO_SIZE))),
                            STRIDE)
                        img518 = cv2.resize(native, (CLIP_SIZE, CLIP_SIZE),
                                            interpolation=cv2.INTER_LINEAR)
                        aug518 = apply_transform(img518, t, CLIP_SIZE)
                        c_aug = clip_extract(clip_model, preprocess, aug518, args.device)
                        wc, vc = inverse_warp_features(
                            c_aug, cv2.invertAffineTransform(t.M_at(float(CLIP_SIZE))),
                            STRIDE)
                        dino_blocks.append(wd)
                        clip_blocks.append(wc)
                        vd_blocks.append(vd)
                        vc_blocks.append(vc)
                        src_refs.append(ri)
                        meta_rows.append({
                            "category": cat, "shot": shot, "ref_index": ri,
                            "ref": relative, "pack": pack, "transform": t.name,
                            "tx_frac": t.tx_frac, "ty_frac": t.ty_frac,
                            "angle_deg": t.angle_deg, "contrast": t.contrast,
                            "brightness_offset": t.brightness_offset,
                        })
                if not dino_blocks:
                    continue
                np.savez_compressed(
                    args.out_dir / f"{cat}_k{shot}_{pack}.npz",
                    aug_d=np.stack(dino_blocks).astype(np.float32),
                    aug_c=np.stack(clip_blocks).astype(np.float32),
                    valid_d=np.stack(vd_blocks).astype(bool),
                    valid_c=np.stack(vc_blocks).astype(bool),
                    source_ref=np.asarray(src_refs, dtype=np.int32),
                    category=np.asarray([cat]),
                    shot=np.asarray([shot], dtype=np.int32),
                    pack=np.asarray([pack]))
                print(f"[done] {cat} k{shot} pack={pack} aug={len(dino_blocks)}")
            print(f"[done] {cat} k{shot} identity")

    manifest_out = {
        "schema_version": 1,
        "program": "innovation_v2",
        "route": "D_DEVA",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dino_model": "dinov2_vitb14",
        "dino_size": DINO_SIZE,
        "clip_model": "ViT-L/14@336px (AnomalyCLIP)",
        "clip_size": CLIP_SIZE,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": common.sha256_file(CHECKPOINT),
        "config_sha256": cfg["_config_sha256"],
        "transforms": {p: [vars(t) for t in tr] for p, tr in transforms.items()},
        "rows": meta_rows,
        "n_rows": len(meta_rows),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest_out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"exported_rows": len(meta_rows)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
