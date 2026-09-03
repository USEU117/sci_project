"""S1-HGLC (task book 16 s.3) - stage-1 GPU export of the two *global* signals
that A1's frozen cache does not contain:

  1) AnomalyCLIP image-level abnormal *text* probability p_abn  (cross-modal)
  2) AnomalyCLIP CLIP global image embedding (ViT-L/14@336, @518 input)

mirroring the official/frozen AnomalyCLIP inference pathway exactly
(methods/AnomalyCLIP-main, checkpoint 9_12_4_multiscale_visa/epoch_15.pth,
DAPM_replace(20), encode_image -> image_features @ learned text prompts ->
softmax/0.07 -> abnormal class probability), as implemented by the previously
validated scripts/export_v4_text_maps.py text branch.

Label-free contract: this script reads test *sample ids* from the frozen A1
compact cache (order alignment) and normal *reference* paths from the MPDD
manifest.  It never reads ground-truth masks or labels (image labels are
derived evaluator-side from "/good/" in the sample id by the diagnostic stage).

Outputs (per category, npz cache + one export_report.json):
  sample_ids (N,) test ids in frozen A1 s0_k1 order
  clip_global_test (N, 768) fp16  normalized CLIP global embedding
  text_prob_test  (N,)     fp16  abnormal-class text probability (higher = anomaly)
  ref_ids (M,) + clip_global_refs (M, 768)  for the k-shot reference union

Run env: .venv-anomalyclip (torch 2.0.0+cu118).  No GT, no sklearn, no cv2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = ROOT / "methods" / "AnomalyCLIP-main"
for p in (str(METHOD_ROOT),):
    if p not in sys.path:
        sys.path.insert(0, p)

import AnomalyCLIP_lib  # noqa: E402
from prompt_ensemble import AnomalyCLIP_PromptLearner  # noqa: E402
from utils import get_transform  # noqa: E402

A1_MAPS = ROOT / "submission_repro_20260827" / "predictions_compact" / "maps" / "mpdd"
MPDD = ROOT / "data" / "mpdd_raw" / "MPDD"
MANIFEST = ROOT / "data" / "splits" / "mpdd" / "manifest.json"
OUT_ROOT = ROOT / "experiments" / "dynamic_fusion" / "innovation_v6_dgsafe" / "s1_hglc"
CACHE = OUT_ROOT / "cache"
CHECKPOINT = (METHOD_ROOT / "checkpoints" / "9_12_4_multiscale_visa" / "epoch_15.pth")

CKPT_SHA256 = "415c5dcb52668b8c33fb9c1a351c686d632b919df5b384d63fa9ce7a2338ced4"
IMAGE_SIZE = 518
LAYERS = [6, 12, 18, 24]
DPAM_LAYER = 20
DESIGN = {"Prompt_length": 12,
          "learnabel_text_embedding_depth": 9,
          "learnabel_text_embedding_length": 4}
SHOTS = (1, 2, 4)
SEED = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--categories", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--swap-text", action="store_true",
                    help="swap normal/abnormal text embeddings (direction sanity)")
    args = ap.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cats = sorted(A1_MAPS.glob(f"s{args.seed}_k1/*.npz"))
    cats = [p.stem for p in cats]
    if args.categories is not None:
        bad = sorted(set(args.categories).difference(cats))
        if bad:
            raise SystemExit(f"unknown categories: {bad}")
        cats = [c for c in cats if c in args.categories]

    assert sha256_file(CHECKPOINT) == CKPT_SHA256, "checkpoint hash mismatch (frozen)"
    CACHE.mkdir(parents=True, exist_ok=True)

    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=args.device,
                                    design_details=DESIGN)
    model.eval()
    preprocess, _ = get_transform(SimpleNamespace(image_size=IMAGE_SIZE))
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), DESIGN)
    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    prompt_learner.load_state_dict(ckpt["prompt_learner"])
    prompt_learner.to(args.device)
    model.to(args.device)
    model.visual.DAPM_replace(DPAM_layer=DPAM_LAYER)

    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    with torch.inference_mode():
        text_features = model.encode_text_learn(
            prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    if args.swap_text:
        text_features = text_features[:, [1, 0], :]
    text_features = text_features.to(args.device)
    text_hash = tensor_sha256(text_features)

    def embed(img_path: Path):
        with Image.open(img_path) as opened:
            t = preprocess(opened.convert("RGB"))
        image = t.reshape(1, 3, IMAGE_SIZE, IMAGE_SIZE).to(args.device)
        with torch.inference_mode():
            image_features, _ = model.encode_image(image, LAYERS, DPAM_layer=DPAM_LAYER)
        gf = image_features / image_features.norm(dim=-1, keepdim=True)
        probs = (gf @ text_features.permute(0, 2, 1) / 0.07).softmax(-1)
        return gf[0].float().cpu().numpy().astype(np.float32), float(probs[0, 0, 1].item())

    report_rows = []
    t_all = time.time()
    for cat in cats:
        npz_a1 = np.load(A1_MAPS / f"s{args.seed}_k1" / f"{cat}.npz", allow_pickle=False)
        sample_ids = [str(s) for s in npz_a1["sample_ids"]]
        # reference union across shots (normal-only)
        refs = []
        for shot in SHOTS:
            for r in manifest["categories"][cat][str(args.seed)][str(shot)]:
                if r not in refs:
                    refs.append(r)
        cg = np.zeros((len(sample_ids), 768), dtype=np.float32)
        tp = np.zeros(len(sample_ids), dtype=np.float32)
        for i, sid in enumerate(sample_ids):
            cg[i], tp[i] = embed(MPDD / sid)
        ref_cg = np.zeros((len(refs), 768), dtype=np.float32)
        for j, r in enumerate(refs):
            ref_cg[j], _ = embed(MPDD / r)
        out = CACHE / f"{cat}.npz"
        np.savez_compressed(
            out,
            sample_ids=np.asarray(sample_ids),
            clip_global_test=cg.astype(np.float16),
            text_prob_test=tp.astype(np.float16),
            ref_ids=np.asarray(refs),
            clip_global_refs=ref_cg.astype(np.float16),
            seed=np.asarray(args.seed),
            image_size=np.asarray(IMAGE_SIZE),
            checkpoint_sha256=np.asarray(CKPT_SHA256),
            text_embedding_sha256=np.asarray(text_hash),
            text_swapped=np.asarray(bool(args.swap_text)),
        )
        report_rows.append({"category": cat, "n_test": len(sample_ids),
                            "n_refs": len(refs),
                            "clip_global_mean_abs": round(float(np.abs(cg).mean()), 5),
                            "text_prob_mean": round(float(tp.mean()), 5),
                            "text_prob_good_vs_defect_mean_diff": None,
                            "output": str(out.resolve())})
        print(f"[{cat}] n={len(sample_ids)} refs={len(refs)} "
              f"mean_text_p={tp.mean():.4f} {time.time()-t_all:.0f}s", flush=True)

    report = {
        "program": "innovation_v6_dgsafe", "phase": "s1_hglc_export",
        "dataset": "mpdd", "role": "development", "seed": args.seed,
        "shots_manifest_used": list(SHOTS),
        "task_book_section": "16 s.3.2/3.3 (image-level global signals)",
        "checkpoint": str(CHECKPOINT.resolve()),
        "checkpoint_sha256": CKPT_SHA256,
        "text_embedding_sha256": text_hash,
        "text_swapped": bool(args.swap_text),
        "image_size": IMAGE_SIZE, "dapm_layer": DPAM_LAYER,
        "design": DESIGN,
        "categories": report_rows,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_total_s": round(time.time() - t_all, 1),
        "note": "label-free export; image labels derived evaluator-side in "
                "run_s1_hglc_diag.py from '/good/' in sample_id",
    }
    (CACHE / "s1_hglc_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"text_embedding_sha256": text_hash,
                      "swapped": bool(args.swap_text), "categories": len(cats)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
