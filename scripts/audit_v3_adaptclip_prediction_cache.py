"""Acceptance audit for the bounded V3 AdaptCLIP MPDD prediction cache.

This evaluator-side audit checks the cache contract before any Gate A result is
computed.  It never uses labels or masks to make a routing decision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scalar(payload: np.lib.npyio.NpzFile, field: str):
    return payload[field].item()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptclip-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--metadata-audit", type=Path, required=True)
    parser.add_argument("--runner-log", type=Path, required=True)
    parser.add_argument("--runner-err", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.metadata_audit.read_text(encoding="utf-8"))
    expected_categories = sorted(source["categories"] and json.loads(
        (Path(__file__).resolve().parents[1] / "data" / "splits" / "mpdd" / "manifest.json").read_text(encoding="utf-8")
    )["categories"])
    rows = []
    all_ids: list[str] = []
    failures: list[str] = []
    for category in expected_categories:
        adaptclip_path = args.adaptclip_root / f"{category}.npz"
        visual_path = args.visual_root / f"{category}.npz"
        row = {"category": category, "adaptclip_path": str(adaptclip_path), "visual_path": str(visual_path)}
        if not adaptclip_path.is_file() or not visual_path.is_file():
            row["status"] = "failed"
            row["error"] = "missing_cache"
            rows.append(row)
            failures.append(f"{category}:missing_cache")
            continue
        with np.load(adaptclip_path, allow_pickle=False) as adaptclip, np.load(visual_path, allow_pickle=False) as visual:
            ids = np.asarray(adaptclip["sample_ids"]).astype(str)
            visual_ids = np.asarray(visual["sample_ids"]).astype(str)
            maps = np.asarray(adaptclip["anomaly_maps"])
            masks = np.asarray(adaptclip["imgs_masks"])
            scores = np.asarray(adaptclip["pr_sp"])
            labels = np.asarray(adaptclip["gt_sp"])
            expected_n = len(visual_ids)
            checks = {
                "sample_count_matches_visual": len(ids) == expected_n,
                "sample_ids_unique": len(ids) == len(set(ids.tolist())),
                "sample_ids_exactly_match_visual": np.array_equal(ids, visual_ids),
                "scores_finite": bool(np.isfinite(scores).all()),
                "maps_finite": bool(np.isfinite(maps).all()),
                "score_shape_valid": scores.shape == (len(ids),),
                "label_shape_valid": labels.shape == (len(ids),),
                "map_shape_valid": maps.ndim == 3 and maps.shape[0] == len(ids),
                "mask_shape_matches_map": masks.shape == maps.shape,
                "dataset_is_mpdd": scalar(adaptclip, "dataset") == "mpdd",
                "dataset_role_is_development": scalar(adaptclip, "dataset_role") == "development",
                "branch_is_adaptclip_text_v3": scalar(adaptclip, "branch") == "adaptclip_text_v3",
                "router_label_leakage_false": bool(scalar(adaptclip, "test_labels_used_by_router")) is False,
                "router_mask_leakage_false": bool(scalar(adaptclip, "test_masks_used_by_router")) is False,
                "router_set_statistics_leakage_false": bool(scalar(adaptclip, "test_set_statistics_used_by_router")) is False,
            }
            row.update(checks, samples=len(ids), map_shape=list(maps.shape), sha256=sha256(adaptclip_path))
            row["status"] = "passed" if all(checks.values()) else "failed"
            if row["status"] == "failed":
                failures.append(category)
            all_ids.extend(ids.tolist())
        rows.append(row)

    total = sum(int(row.get("samples", 0)) for row in rows)
    runner_text = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (args.runner_log, args.runner_err)
        if path.is_file()
    ).lower()
    runner_failed = "traceback" in runner_text or "out of memory" in runner_text or "cuda oom" in runner_text
    overall = {
        "schema_version": 1,
        "run_id": "v3_adaptclip_mpdd_s0_k1_gate_a_v2_acceptance",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "mpdd",
        "dataset_role": "development",
        "seed": 0,
        "shot": 1,
        "categories_expected": len(expected_categories),
        "categories_found": len(rows) - sum("missing_cache" in item for item in failures),
        "test_images_expected": int(source["test_images"]),
        "test_images_found": total,
        "all_sample_ids_globally_unique": len(all_ids) == len(set(all_ids)),
        "traceback_or_oom_in_runner_log": runner_failed,
        "rows": rows,
        "failures": failures,
    }
    overall["status"] = "passed" if not failures and total == source["test_images"] and overall["all_sample_ids_globally_unique"] and not runner_failed else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps({key: overall[key] for key in ("status", "categories_found", "test_images_found", "all_sample_ids_globally_unique", "failures")}, indent=2))
    return 0 if overall["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
