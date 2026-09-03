"""V12-EARLY-FUSION Stage 0 multi-layer feature exporter (doc 23 Stage0).

Reuses the frozen per-branch extraction pipelines of the A1 caches and returns
SEVERAL intermediate layers per forward:
  dino : dinov2_vitb14 get_intermediate_layers block indices [6, 9, 11] -> 3 x 32x32 grids
  clip : AnomalyCLIP image tower encode_image features_list [6, 12, 18, 24] -> 4 x 37x37 grids
Deepest layer (dino 11 / clip 24) must reproduce the frozen A1 caches (parity gate <1e-5).

Run (GPU, .venv-anomalyclip), one branch at a time:
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_multilayer_export.py --branch dino --shot 1
  .venv-anomalyclip\\Scripts\\python.exe scripts\\innovation_v12_new_observables\\run_r3_ef_multilayer_export.py --branch clip --shot 1
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
sys.path.insert(0, str(ROOT / "methods" / "anomalydino"))
sys.path.insert(0, str(ROOT / "methods" / "AnomalyCLIP-main"))

from v2_mpdd_prediction_common import index_dataset, sha256  # noqa: E402

CATEGORIES = ["bracket_black", "bracket_brown", "bracket_white", "connector", "metal_plate", "tubes"]
MANIFEST = ROOT / "data/splits/mpdd/manifest.json"
DATA_ROOT = ROOT / "data/mpdd_raw/MPDD"
OUT_ROOT = ROOT / "outputs/dynamic_fusion/v12_early_fusion"
DINO_LAYERS = [6, 9, 11]
CLIP_LAYERS = [6, 12, 18, 24]


def load_image(path: Path) -> np.ndarray:
    import cv2

    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def make_dino(device: str):
    from src.backbones import get_model

    wrapper = get_model("dinov2_vitb14", device, smaller_edge_size=448)
    model = wrapper.model  # dinov2 hub module (has get_intermediate_layers)

    def extract(rgb: np.ndarray):
        tensor, grid = wrapper.prepare_image(rgb)  # [3,Hc,Wc] + (gh,gw)
        gh, gw = grid
        batch = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            mids = model.get_intermediate_layers(batch, n=[6, 9])   # list path (hub-normalized)
            final = model.get_intermediate_layers(batch, n=1)[0]    # EXACT A1 final-layer path
            feats = []
            for o in list(mids) + [final]:
                a = o.float().cpu().numpy()
                if a.ndim == 3:
                    a = a[0]
                if a.shape[0] == gh * gw + 1:
                    a = a[1:]  # drop cls if present
                a = a.reshape(gh, gw, -1)
                feats.append(a)
        return feats  # L6, L9, L11 each (gh,gw,d)

    return extract


def make_clip(device: str):
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
            _, pf = model.encode_image(tensor, CLIP_LAYERS, DPAM_layer=20)
        side = 37
        return [pf[k][0, 1:, :].float().cpu().numpy().reshape(side, side, -1)
                for k in range(len(CLIP_LAYERS))]

    return extract


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("dino", "clip"), required=True)
    parser.add_argument("--shot", type=int, default=1, choices=[1, 2, 4])
    parser.add_argument("--category", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    layer_ids = DINO_LAYERS if args.branch == "dino" else CLIP_LAYERS
    out_dir = OUT_ROOT / f"ml_{args.branch}_s0_k{args.shot}"
    out_dir.mkdir(parents=True, exist_ok=True)
    extract = make_dino(args.device) if args.branch == "dino" else make_clip(args.device)
    cats = [args.category] if args.category else CATEGORIES

    rows = []
    for cat in cats:
        refs = manifest["categories"][cat]["0"][str(args.shot)]
        samples = index_dataset("mpdd", DATA_ROOT)[cat]
        refs_layers = []
        grid = None
        for rel in refs:
            outs = extract(load_image(DATA_ROOT / rel))
            refs_layers.append(outs)
            grid = tuple(outs[0].shape[:2])
        ref_blocks = np.stack([np.stack([r[l] for r in refs_layers])
                               for l in range(len(layer_ids))])  # [L,K,H,W,d]

        feat_blocks = []
        masks = []
        sample_ids = []
        for s in samples:
            outs = extract(load_image(Path(s.image_path)))
            feat_blocks.append(np.stack(outs))  # [L,H,W,d]
            if s.mask_path is None:
                m = np.zeros((grid[0] * 14, grid[1] * 14), dtype=np.uint8)
            else:
                import cv2 as _cv2

                m = _cv2.imread(str(s.mask_path), _cv2.IMREAD_GRAYSCALE)
                m = _cv2.resize(m, (grid[1] * 14, grid[0] * 14),
                                interpolation=_cv2.INTER_NEAREST)
                m = (m > 0).astype(np.uint8)
            masks.append(m)
            sample_ids.append(s.sample_id)
        feat = np.stack(feat_blocks)  # [N,L,H,W,d]

        out = out_dir / f"{cat}.npz"
        np.savez_compressed(
            out,
            patch_features=feat.astype(np.float32),
            ref_patch_features=ref_blocks.astype(np.float32),
            imgs_masks=np.asarray(masks, dtype=np.uint8),
            sample_ids=np.asarray(sample_ids),
            layer_ids=np.asarray(layer_ids, dtype=np.int64),
            grid_size=np.asarray(grid, dtype=np.int64),
            branch=np.asarray(args.branch),
            seed=np.asarray(0), shot=np.asarray(args.shot),
        )
        rows.append({"category": cat, "layers": layer_ids, "n": len(samples),
                     "refs": len(refs), "grid": list(grid),
                     "output": str(out.resolve()), "sha256": sha256(out)})
        print(f"[ef-export {args.branch} k{args.shot}] {cat} {rows[-1]['sha256'][:10]}", flush=True)

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "branch": args.branch, "shot": args.shot, "layer_ids": layer_ids,
        "manifest_sha256": sha256(MANIFEST), "categories": rows,
    }
    (out_dir / "export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
