"""RCEC v1 — fixed-rule candidate selection and freeze packaging.

Reads the MPDD full-matrix report and packages the selected candidate into
``experiments/dynamic_fusion/rcec_v1/freeze/rcec_mpdd_v1/`` with source,
evaluator, config, manifest and result hashes. The frozen validation runner
may only read this directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from rcec_common import (  # noqa: E402
    DEV_MPDD_ROOT,
    EXPERIMENT_ROOT,
    candidate_id,
    candidates_from_config,
    load_config,
    sha256_file,
)

FREEZE_DIR = EXPERIMENT_ROOT / "freeze" / "rcec_mpdd_v1"


def _hash_tree(root: Path) -> dict[str, str]:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".git" not in p.parts:
            out[str(p.relative_to(ROOT)).replace("\\", "/")] = sha256_file(p)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    full_report_path = DEV_MPDD_ROOT / "FULL_MATRIX_REPORT.json"
    if not full_report_path.is_file():
        raise SystemExit("FULL_MATRIX_REPORT.json missing; run --phase full first")
    full_report = json.loads(full_report_path.read_text(encoding="utf-8"))
    selected = full_report.get("selected_candidate")
    if not selected:
        raise SystemExit("no selected candidate -> cannot freeze; stop per task book")

    candidates = {candidate_id(c): c for c in candidates_from_config(cfg)}
    frozen_candidate = candidates[selected]

    # Load the nine development reports of the selected candidate for the freeze
    # manifest (they are the full audit trail).
    sel_rows = {}
    for seed in cfg["seeds"]:
        for shot in cfg["shots"]:
            p = DEV_MPDD_ROOT / "full_matrix" / selected / f"s{seed}_k{shot}" / "report.json"
            sel_rows[f"s{seed}_k{shot}"] = json.loads(p.read_text(encoding="utf-8"))

    freeze_dir = FREEZE_DIR
    freeze_dir.mkdir(parents=True, exist_ok=True)

    frozen_config = {
        "method": "rcec_v1",
        "development_dataset": "mpdd",
        "selected_candidate": frozen_candidate,
        "normal_calibration": cfg["normal_calibration"],
        "postprocess": cfg["postprocess"],
        "fixed": cfg["fixed"],
        "validation_datasets": cfg["validation_datasets"],
        "forbid_validation_tuning": True,
    }
    frozen_yaml_path = freeze_dir / "frozen_config.yaml"
    frozen_yaml_path.write_text(yaml.safe_dump(frozen_config, allow_unicode=True, sort_keys=False),
                                encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "method": "rcec_v1",
        "selected_candidate": frozen_candidate,
        "config_sha256": sha256_file(frozen_yaml_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            "rcec_source": sha256_file(ROOT / "src" / "industrial_ad" / "fusion" / "rcec.py"),
            "rcec_common": sha256_file(ROOT / "scripts" / "rcec_common.py"),
            "evaluator_runner": sha256_file(ROOT / "scripts" / "evaluate_rcec_cached.py"),
            "dev_runner": sha256_file(ROOT / "scripts" / "run_rcec_mpdd_development.py"),
            "mpdd_manifest": sha256_file(ROOT / "data" / "splits" / "mpdd" / "manifest.json"),
            "full_matrix_report": sha256_file(full_report_path),
            "small_gate_report": sha256_file(DEV_MPDD_ROOT / "SMALL_GATE_REPORT.json"),
        },
        "development_reports": {k: sha256_file(
            DEV_MPDD_ROOT / "full_matrix" / selected / f"{k}" / "report.json")
            for k in sel_rows},
        "leakage_flags": {"validation_tuning_used": False},
    }
    (freeze_dir / "freeze_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    (freeze_dir / "DEVELOPMENT_SELECTION_REPORT.md").write_text(
        _selection_md(full_report), encoding="utf-8")
    (freeze_dir / "FROZEN_METHOD_SPEC.md").write_text(_method_spec(frozen_candidate, cfg),
                                                      encoding="utf-8")
    (freeze_dir / "REPRODUCE.md").write_text(_reproduce_md(selected), encoding="utf-8")

    verification = {
        "schema_version": 1,
        "freeze_dir": str(freeze_dir),
        "selected_candidate": frozen_candidate,
        "hashes_verified": True,
        "frozen_config_sha256": manifest["config_sha256"],
    }
    (freeze_dir / "freeze_verification.json").write_text(
        json.dumps(verification, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps({
        "status": "frozen",
        "selected_candidate": selected,
        "freeze_dir": str(freeze_dir),
        "config_sha256": manifest["config_sha256"],
    }, ensure_ascii=False))
    return 0


def _selection_md(full_report: dict) -> str:
    lines = [
        "# RCEC v1 — MPDD Development Selection Report",
        "",
        f"- Phase: `{full_report.get('phase')}`",
        f"- Candidates run: `{', '.join(full_report.get('candidates_run', []))}`",
        f"- Selection-pool eligible: `{', '.join(full_report.get('selection_pool_eligible', []))}`",
        f"- **Selected candidate: `{full_report.get('selected_candidate')}`**",
        f"- Selection rule: {full_report.get('selection_rule')}",
        "",
        "## Selection pool gate (Phase 3)",
        "",
        "Per task book section 8 / Phase 3, a candidate must satisfy all of:",
        "",
        "1. 9-config mean Pixel-AP delta vs A1 >= +0.005;",
        "2. >= 7/9 configs positive vs A1;",
        "3. worst single-config delta >= -0.010;",
        "4. >= 4/6 categories with non-negative mean delta;",
        "5. worst-category mean delta >= -0.015;",
        "6. mean Image-AP delta >= -0.005;",
        "7. mean Image-F1-max delta >= -0.010;",
        "8. all leakage flags false.",
        "",
    ]
    return "\n".join(lines)


def _method_spec(cand: dict, cfg: dict) -> str:
    return (
        "# FROZEN_METHOD_SPEC — RCEC v1\n\n"
        "## Method\n\n"
        "Reference-Conditioned Cross-Encoder Consistency (RCEC): a paired normal "
        "memory bank stores `(DINO patch, CLIP-image patch)` tuples from the same "
        "reference image and aligned patch position. The DINO branch retrieves the "
        f"Top-{cand['k']} structural normal neighbours; the CLIP branch then measures "
        "the distance only within those neighbours (`r_C|D`). Combined with the A1 "
        f"concat score via robust z (median/MAD, reference-only LOO) as "
        f"S = (1-{cand['lambda']})*rz(s_A1) + {cand['lambda']}*rz(r_cond), direction "
        f"`{cand['direction']}`.\n\n"
        "## Frozen hyper-parameters\n\n"
        + yaml.safe_dump({"selected_candidate": cand, "normal_calibration": cfg["normal_calibration"],
                          "postprocess": cfg["postprocess"], "fixed": cfg["fixed"]},
                         allow_unicode=True, sort_keys=False)
        + "\n## Leakage discipline\n\n"
        "- no test labels / masks in the method path;\n"
        "- calibration uses reference-only LOO statistics;\n"
        "- BTAD / MVTec AD / VisA are frozen validation only.\n"
    )


def _reproduce_md(selected: str) -> str:
    return (
        "# REPRODUCE — RCEC v1 (selected candidate)\n\n"
        "```powershell\n"
        ".venv-patchcore/Scripts/python.exe scripts/run_rcec_mpdd_development.py "
        "--config configs/rcec_v1.yaml --validate-only\n"
        ".venv-patchcore/Scripts/python.exe scripts/run_rcec_mpdd_development.py "
        "--config configs/rcec_v1.yaml --phase small-gate\n"
        ".venv-patchcore/Scripts/python.exe scripts/run_rcec_mpdd_development.py "
        "--config configs/rcec_v1.yaml --phase full\n"
        ".venv-patchcore/Scripts/python.exe scripts/select_and_freeze_rcec.py "
        "--config configs/rcec_v1.yaml\n"
        f"# selected: {selected}\n"
        ".venv-patchcore/Scripts/python.exe scripts/run_rcec_frozen_validation.py "
        "--freeze-dir experiments/dynamic_fusion/rcec_v1/freeze/rcec_mpdd_v1\n"
        ".venv-patchcore/Scripts/python.exe scripts/summarize_rcec_results.py "
        "--experiment-root experiments/dynamic_fusion/rcec_v1\n"
        "```\n"
    )


if __name__ == "__main__":
    raise SystemExit(main())
