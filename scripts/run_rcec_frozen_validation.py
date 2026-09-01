"""RCEC v1 — one-shot frozen validation on BTAD / MVTec AD / VisA.

The runner ONLY reads the freeze directory; it never accepts method parameter
overrides. It verifies the freeze hashes first, then runs the frozen config on
each validation dataset x seed x shot exactly once. Results are written to
``experiments/dynamic_fusion/rcec_v1/frozen_validation/``.
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
    EXPERIMENT_ROOT,
    check_no_config_overrides,
    evaluate_config,
    sha256_file,
)

VAL_ROOT = EXPERIMENT_ROOT / "frozen_validation"


def _verify_freeze(freeze_dir: Path) -> dict:
    manifest = json.loads(
        (freeze_dir / "freeze_manifest.json").read_text(encoding="utf-8"))
    for rel, expected in manifest["files"].items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            raise SystemExit(
                f"freeze hash mismatch: {rel} expected {expected} got {actual}")
    frozen_yaml = freeze_dir / "frozen_config.yaml"
    if sha256_file(frozen_yaml) != manifest["config_sha256"]:
        raise SystemExit("frozen_config.yaml hash mismatch vs manifest")
    return json.loads(frozen_yaml.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze-dir", type=Path, required=True)
    args = parser.parse_args()
    freeze_dir = args.freeze_dir if args.freeze_dir.is_absolute() else ROOT / args.freeze_dir

    frozen = _verify_freeze(freeze_dir)
    cand = frozen["selected_candidate"]
    check_no_config_overrides(
        {"direction": cand["direction"], "k": cand["k"], "lambda": cand["lambda"]}, cand)
    if frozen.get("forbid_validation_tuning", True) is not True:
        raise SystemExit("frozen config does not forbid validation tuning")

    datasets = frozen["validation_datasets"]
    cfg_for_eval = {
        "normal_calibration": frozen["normal_calibration"],
        "postprocess": frozen["postprocess"],
        "fixed": frozen["fixed"],
    }
    config_sha = sha256_file(freeze_dir / "frozen_config.yaml")

    summary = {}
    for ds in datasets:
        for seed in (0, 1, 2):
            for shot in (1, 2, 4):
                out_dir = VAL_ROOT / ds / f"s{seed}_k{shot}"
                marker = out_dir / "report.json"
                meta = out_dir / "marker.json"
                if marker.is_file() and meta.is_file():
                    saved = json.loads(meta.read_text(encoding="utf-8"))
                    if saved.get("config_sha256") == config_sha:
                        report = json.loads(marker.read_text(encoding="utf-8"))
                    else:
                        raise SystemExit(
                            f"existing report for {ds}/s{seed}_k{shot} has a different "
                            "config hash; refusing to mix configs")
                else:
                    report = evaluate_config(ds, seed, shot, cand, cfg_for_eval)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    marker.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
                    meta.write_text(
                        json.dumps({"config_sha256": config_sha,
                                    "created_at_utc": datetime.now(timezone.utc).isoformat()}),
                        encoding="utf-8")
                d = report["metrics"]["delta_rcec_vs_a1"]["pixel"]["pixel_ap"]
                summary.setdefault(ds, []).append(d)
                print(f"[frozen] {ds}/s{seed}_k{shot} delta_pixel_ap={d}", flush=True)

    out = {
        "schema_version": 1,
        "phase": "frozen_validation",
        "frozen_config_sha256": config_sha,
        "selected_candidate": cand,
        "summary": {ds: {"mean_delta": round(float(sum(v)) / len(v), 6),
                         "n_positive": int(sum(1 for x in v if x > 0)),
                         "n_configs": len(v)} for ds, v in summary.items()},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (VAL_ROOT / "FROZEN_VALIDATION_REPORT.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
