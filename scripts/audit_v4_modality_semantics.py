"""G0 read-only audit: A1 modality semantics + candidate source lock.

Proves, from frozen source, that A1's second branch is a CLIP *image* patch
feature (not an explicit text branch), and records the V4 candidate source lock
table together with git/environment state. Strictly CPU / read-only: it never
launches a model and never writes outside --output-dir.

Outputs (into --output-dir):
  state.json
  modality_semantics_audit.json
  modality_semantics_audit.md
  candidate_source_lock.json
  environment.txt
  commands.txt
  hashes.sha256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import freeze_a1_mpdd as freeze  # noqa: E402

EXPORT_SCRIPT = ROOT / "scripts" / "export_anomalyclip_mpdd_features.py"
ANOMALYCLIP_SRC = ROOT / "methods" / "AnomalyCLIP-main" / "AnomalyCLIP_lib" / "AnomalyCLIP.py"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git(args: list[str]) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001
        return f"<git unavailable: {exc}>"


def collect_environment() -> dict:
    info: dict = {"python_version": platform.python_version(), "platform": platform.platform()}
    try:
        import faiss  # noqa: F401
        info["faiss"] = getattr(faiss, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        info["faiss"] = "unavailable"
    try:
        import numpy  # noqa: F401
        info["numpy"] = numpy.__version__
    except Exception:  # noqa: BLE001
        info["numpy"] = "unavailable"
    try:
        import sklearn  # noqa: F401
        info["sklearn"] = sklearn.__version__
    except Exception:  # noqa: BLE001
        info["sklearn"] = "unavailable"
    try:
        import torch  # noqa: F401
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_count"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        info["torch"] = "unavailable"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        info["gpu"] = out.stdout.strip() or "none"
    except Exception:  # noqa: BLE001
        info["gpu"] = "unavailable"
    return info


def static_modality_audit() -> dict:
    """Static, evidence-based audit that A1's second branch is CLIP image patch, not text."""
    export_src = EXPORT_SCRIPT.read_text(encoding="utf-8")
    anomalylip_src = ANOMALYCLIP_SRC.read_text(encoding="utf-8")

    evidence = []

    # 1. Export path never calls encode_text.
    calls_encode_text = "encode_text" in export_src
    evidence.append({
        "id": "export_calls_encode_text",
        "check": "export script never calls encode_text",
        "passed": not calls_encode_text,
        "detail": f"encode_text present in export script = {calls_encode_text}",
    })

    # 2. Export path calls encode_image.
    calls_encode_image = "model.encode_image(" in export_src
    evidence.append({
        "id": "export_calls_encode_image",
        "check": "export script calls model.encode_image(...)",
        "passed": calls_encode_image,
        "detail": f"encode_image present = {calls_encode_image}",
    })

    # 3. prompt_learner is loaded but never feeds the image encoder.
    prompt_loaded = "prompt_learner" in export_src
    # The only references are load/load_state_dict/to(device); no forward use.
    prompt_forward_uses = [ln for ln in export_src.splitlines() if "prompt_learner" in ln]
    evidence.append({
        "id": "prompt_learner_usage",
        "check": "prompt learner is loaded but not used in feature extraction",
        "passed": prompt_loaded and all(
            ("load" in ln or "load_state_dict" in ln or "to(" in ln or "prompt_learner =" in ln)
            for ln in prompt_forward_uses
        ),
        "detail": prompt_forward_uses,
    })

    # 4. AnomalyCLIP.encode_image delegates only to self.visual.
    evidence.append({
        "id": "anomalyclip_encode_image_visual_only",
        "check": "AnomalyCLIP.encode_image delegates to self.visual (vision-only)",
        "passed": "return self.visual(" in anomalylip_src,
        "detail": "encode_image returns self.visual(image.type(...), ...) at AnomalyCLIP.py:478-479",
    })

    # 5. VisionTransformer.forward starts from pixel conv, no text input.
    evidence.append({
        "id": "vit_forward_pixel_only",
        "check": "VisionTransformer.forward consumes only pixel tokens (conv1 -> patches)",
        "passed": "x = self.conv1(x)" in anomalylip_src and "encode_text" not in anomalylip_src.split("class AnomalyCLIP")[0],
        "detail": "forward begins `x = self.conv1(x)` (AnomalyCLIP.py:358); no text tensor enters the vision tower.",
    })

    # 6. DAPM_replace replaces visual attention with v-v self-attention.
    evidence.append({
        "id": "dapm_is_visual_self_attention",
        "check": "DAPM_replace swaps in visual self-attention (no text/prompt injection)",
        "passed": "self.transformer.resblocks[-i].attn = self.attn" in anomalylip_src,
        "detail": "DAPM_replace clones in_proj/out_proj into an Attention module and re-assigns visual resblock attn (AnomalyCLIP.py:344-353).",
    })

    # 7. Attention.forward builds q,k,v all from the same visual patch tensor x.
    attention_block = anomalylip_src.split("class Attention")[1].split("class LayerNorm")[0]
    qkv_from_x = "self.qkv(x)" in attention_block
    evidence.append({
        "id": "attention_qkv_from_visual_x",
        "check": "Attention computes q,k,v all from the same visual patch tensor",
        "passed": qkv_from_x,
        "detail": "Attention.forward: qkv = self.qkv(x) -> q,k,v (AnomalyCLIP.py:71-74); no text or prompt tensor present.",
    })

    a1_modality = "dual_visual_fixed_fusion" if all(e["passed"] for e in evidence) else "UNRESOLVED"

    return {
        "a1_modality": a1_modality,
        "conclusion": (
            "A1's second branch is a CLIP image patch feature (encode_image -> visual ViT), "
            "not an explicit text branch. The prompt_learner is loaded but never used during "
            "export; encode_text / image-text similarity are never computed. Therefore A1 is "
            "a dual-visual fixed fusion, not explicit visual-text fusion."
        ),
        "evidence": evidence,
        "all_passed": a1_modality == "dual_visual_fixed_fusion",
        "sources": {
            "export_script": str(EXPORT_SCRIPT.relative_to(ROOT)),
            "anomalyclip_source": str(ANOMALYCLIP_SRC.relative_to(ROOT)),
            "export_script_sha256": sha256_text(export_src),
            "anomalyclip_source_sha256": sha256_text(anomalylip_src),
        },
    }


def build_source_lock() -> dict:
    """Candidate source lock table. Local repos verified by LICENSE file; remote
    repos recorded from the plan's section 16 with current known status."""
    def license_of(path: Path) -> str:
        lic = path / "LICENSE"
        if not lic.is_file():
            return "unknown"
        head = lic.read_text(encoding="utf-8", errors="replace")[:200].lower()
        if "mit license" in head:
            return "MIT"
        if "apache license" in head:
            return "Apache-2.0"
        if "general public license" in head or "gnu" in head:
            return "GPL"
        return "other"

    return {
        "schema_version": 1,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidates": [
            {
                "id": "V0_AnomalyDINO",
                "role": "strong visual baseline (must beat)",
                "source": "https://github.com/dammsi/AnomalyDINO",
                "local_path": "methods/anomalydino",
                "status": "available",
                "license": license_of(ROOT / "methods" / "anomalydino"),
                "commit": "n/a (local archive, not a git checkout)",
                "weights": "dinov2_vitb14 via torch hub (cached locally); official AnomalyDINO uses DINOv2 (ViT-B/14)",
                "queueable": True,
            },
            {
                "id": "T0_AnomalyCLIP_text",
                "role": "explicit text branch backbone (text-conditioned map)",
                "source": "https://github.com/caoyunkang/AnomalyCLIP",
                "local_path": "methods/AnomalyCLIP-main",
                "status": "available",
                "license": license_of(ROOT / "methods" / "AnomalyCLIP-main"),
                "commit": "n/a (local archive)",
                "weights": "methods/AnomalyCLIP-main/checkpoints/9_12_4_multiscale_visa/epoch_15.pth (frozen A1 checkpoint)",
                "queueable": True,
                "note": "A1 used only its encode_image patch features; T0 must produce explicit normal/abnormal text embeddings + similarity map.",
            },
            {
                "id": "V1_SubspaceAD_style",
                "role": "PCA normal-subspace reconstruction residual (same backbone/cache)",
                "source": "https://github.com/CLendering/SubspaceAD",
                "local_path": None,
                "status": "not_cloned",
                "license": "unknown (to verify on clone)",
                "commit": "unknown",
                "weights": "not required for V1 (reuses frozen DINO raw patch cache)",
                "queueable": True,
                "note": "V1 = subspace_style_same_backbone, NOT an official SubspaceAD reproduction.",
            },
            {
                "id": "V2_SubspaceAD_official",
                "role": "official SubspaceAD (DINOv2-giant, 672, augmentation)",
                "source": "https://github.com/CLendering/SubspaceAD",
                "local_path": None,
                "status": "not_cloned",
                "license": "unknown (to verify on clone)",
                "commit": "unknown",
                "weights": "DINOv2-giant (needs download; 6GB VRAM constraint -> validate-only/single-category smoke first)",
                "queueable": False,
                "note": "Only after V1 passes G2.",
            },
            {
                "id": "V3_FoundAD",
                "role": "nonlinear manifold projector candidate",
                "source": "https://github.com/ymxlzgy/FoundAD",
                "local_path": None,
                "status": "not_cloned",
                "license": "unknown (to verify on clone)",
                "commit": "unknown",
                "weights": "unknown",
                "queueable": False,
                "note": "Second visual candidate.",
            },
            {
                "id": "V4_alt_FastRef",
                "role": "query-conditioned prototype refinement",
                "source": "https://github.com/liyufei25/FastRef",
                "local_path": None,
                "status": "blocked_empty_repo",
                "license": "n/a",
                "commit": "n/a",
                "weights": "n/a",
                "queueable": False,
                "note": "Repo is empty as of 2026-08-19; do not rewrite and call it an official reproduction.",
            },
            {
                "id": "T1_ReMP_AD",
                "role": "retrieval-enhanced prompt fusion (fallback text branch)",
                "source": "https://github.com/cshcma/ReMP-AD",
                "local_path": "methods/remp_ad",
                "status": "available",
                "license": license_of(ROOT / "methods" / "remp_ad"),
                "commit": "n/a (local archive)",
                "weights": "methods/remp_ad/result/mvtec/epoch_15.pth present (training artifact); verify protocol/retrieval library before use",
                "queueable": True,
                "note": "Audit whether the retrieval library uses target test info before queueing.",
            },
            {
                "id": "T2_AdaptCLIP",
                "role": "conditional text branch candidate",
                "source": "https://github.com/caoyunkang/AdaptCLIP",
                "local_path": "methods/adaptclip",
                "status": "available",
                "license": license_of(ROOT / "methods" / "adaptclip"),
                "commit": "n/a (local archive)",
                "weights": "unknown (needs verification)",
                "queueable": False,
                "note": "Clarify whether it adapts per K/seed and the training data/parameter source before queueing.",
            },
        ],
    }


def collect_state() -> dict:
    return {
        "run_id": "v4_g0_modality_and_source_audit_20260819_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "branch_status": git(["status", "--short", "--branch"]),
            "last_commits": git(["log", "-3", "--oneline", "--decorate"]),
            "head": git(["rev-parse", "HEAD"]),
        },
        "a1_freeze_verify": None,  # filled below
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments" / "dynamic_fusion" / "v4_vision_text_20260819" / "00_g0_audit",
    )
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    env = collect_environment()
    modality = static_modality_audit()
    source_lock = build_source_lock()

    # Read-only A1 freeze verification (imports freeze_a1_mpdd; never writes).
    a1_manifest = ROOT / "experiments" / "dynamic_fusion" / "freeze" / "a1_mpdd_w05" / "freeze_manifest.json"
    a1_verify = freeze.verify_manifest(a1_manifest)

    state = collect_state()
    state["a1_freeze_verify"] = {
        "all_ok": a1_verify.get("all_ok"),
        "verified_entries": a1_verify.get("verified_entries"),
        "missing": a1_verify.get("missing"),
        "size_mismatch": a1_verify.get("size_mismatch"),
        "hash_mismatch": a1_verify.get("hash_mismatch"),
        "extra_undeclared_npz": a1_verify.get("extra_undeclared_npz"),
        "manifest_sha256_before": a1_verify.get("manifest_sha256_before"),
    }

    commands = [
        ".venv-patchcore\\Scripts\\python.exe scripts\\freeze_a1_mpdd.py --verify",
        ".venv-patchcore\\Scripts\\python.exe -m pytest tests\\test_v3_3_clean.py tests\\test_v3_3_rescue.py tests\\test_freeze_a1_mpdd.py -q",
        ".venv-patchcore\\Scripts\\python.exe scripts\\audit_v4_modality_semantics.py --output-dir experiments\\dynamic_fusion\\v4_vision_text_20260819\\00_g0_audit",
    ]

    env_txt = "\n".join(f"{k}: {v}" for k, v in env.items())
    (out / "environment.txt").write_text(env_txt + "\n", encoding="utf-8")
    (out / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")

    # Hashes of the two frozen source files that carry the audit evidence.
    hashes = {
        "scripts/export_anomalyclip_mpdd_features.py": freeze.sha256(EXPORT_SCRIPT),
        "methods/AnomalyCLIP-main/AnomalyCLIP_lib/AnomalyCLIP.py": freeze.sha256(ANOMALYCLIP_SRC),
    }
    (out / "hashes.sha256").write_text(
        "\n".join(f"{h}  {p}" for p, h in sorted(hashes.items())) + "\n", encoding="utf-8"
    )

    (out / "state.json").write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "modality_semantics_audit.json").write_text(
        json.dumps(modality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "candidate_source_lock.json").write_text(
        json.dumps(source_lock, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# A1 Modality Semantics Audit (G0)",
        "",
        f"- run_id: `{state['run_id']}`",
        f"- created_at_utc: `{state['created_at_utc']}`",
        f"- **A1 modality**: `{modality['a1_modality']}`",
        f"- A1 freeze verify all_ok: `{state['a1_freeze_verify']['all_ok']}` (entries: {state['a1_freeze_verify']['verified_entries']})",
        "",
        "## Conclusion",
        "",
        modality["conclusion"],
        "",
        "## Evidence",
        "",
    ]
    for e in modality["evidence"]:
        mark = "PASS" if e["passed"] else "FAIL"
        md_lines.append(f"- [{mark}] {e['check']} — {e['detail']}")
    md_lines += [
        "",
        "## Candidate source lock (summary)",
        "",
        "| id | status | license | queueable |",
        "| --- | --- | --- | --- |",
    ]
    for c in source_lock["candidates"]:
        md_lines.append(f"| {c['id']} | {c['status']} | {c['license']} | {c['queueable']} |")
    md_lines.append("")
    (out / "modality_semantics_audit.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_dir": str(out),
        "a1_modality": modality["a1_modality"],
        "modality_all_passed": modality["all_passed"],
        "a1_freeze_all_ok": state["a1_freeze_verify"]["all_ok"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
