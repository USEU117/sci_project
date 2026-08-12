"""Create small deterministic fixtures for the normal-reference pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from industrial_ad.fusion.alignment import canonicalize_sample_ids


def stable_offset(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / (2**32)


def write_branch(
    output: Path,
    dataset: str,
    branch: str,
    category: str,
    seed: int,
    shot: int,
    source_ids: np.ndarray,
    views: int,
    branch_offset: float,
) -> None:
    sample_ids: list[str] = []
    repeated_sources: list[str] = []
    augmentation_ids: list[str] = []
    scores: list[float] = []
    maps: list[np.ndarray] = []
    grid_y, grid_x = np.mgrid[0:8, 0:8]
    grid = (grid_y + grid_x).astype(np.float32) / 100.0
    for source_id in source_ids.tolist():
        base = stable_offset(f"{branch}:{category}:{source_id}") * 0.1
        for view in range(views):
            sample_ids.append(f"{source_id}::view-{view}")
            repeated_sources.append(source_id)
            augmentation_ids.append(f"deterministic-view-{view}")
            score = branch_offset + base + view * 0.025
            scores.append(score)
            maps.append(grid + score + view * 0.005)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        sample_ids=np.asarray(sample_ids),
        source_ids=np.asarray(repeated_sources),
        augmentation_ids=np.asarray(augmentation_ids),
        image_scores=np.asarray(scores, dtype=np.float32),
        pixel_maps=np.asarray(maps, dtype=np.float32),
        dataset=np.asarray(dataset),
        branch=np.asarray(branch),
        category=np.asarray(category),
        seed=np.asarray(seed),
        shot=np.asarray(shot),
        score_direction=np.asarray("higher_is_more_anomalous"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-dir", type=Path, required=True)
    parser.add_argument("--text-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--views", type=int, default=4)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    for category, category_data in manifest["categories"].items():
        selected = category_data[str(args.seed)][str(args.shot)]
        source_ids = canonicalize_sample_ids(np.asarray(selected))
        write_branch(
            args.visual_dir / f"{category}.npz",
            manifest["dataset"],
            "synthetic_visual",
            category,
            args.seed,
            args.shot,
            source_ids,
            args.views,
            0.1,
        )
        write_branch(
            args.text_dir / f"{category}.npz",
            manifest["dataset"],
            "synthetic_text",
            category,
            args.seed,
            args.shot,
            source_ids,
            args.views,
            5.0,
        )
    print(
        json.dumps(
            {
                "categories": len(manifest["categories"]),
                "seed": args.seed,
                "shot": args.shot,
                "views_per_source": args.views,
            }
        )
    )


if __name__ == "__main__":
    main()
