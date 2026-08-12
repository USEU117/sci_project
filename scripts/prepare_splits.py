"""Create deterministic nested K-shot normal-reference manifests.

The script intentionally uses only the Python standard library so it can run
before any ML environment is installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def image_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def normal_candidates(root: Path, dataset: str, category: str) -> list[str]:
    """Return normal-reference paths for supported industrial AD layouts."""
    if dataset == "visa" and (root / "meta.json").is_file():
        meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        return sorted(
            item["img_path"]
            for item in meta["train"][category]
            if int(item.get("anomaly", 0)) == 0
        )
    category_root = root / category
    candidates = [
        category_root / "train" / "good",
        category_root / "train" / "ok",
    ]
    normal_dir = next((path for path in candidates if path.is_dir()), None)
    if normal_dir is None:
        return []
    return [p.relative_to(root).as_posix() for p in image_files(normal_dir)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset", required=True, choices=["mvtec", "visa", "mpdd", "btad"]
    )
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=Path("data/splits"))
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {root}")
    if any(s <= 0 for s in args.shots):
        raise SystemExit("shots must be positive")
    shots = sorted(set(args.shots))
    out_dir = args.output / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "root": str(root),
        "shots": shots,
        "seeds": sorted(set(args.seeds)),
        "nested": True,
        "categories": {},
    }

    if args.dataset == "visa" and (root / "meta.json").is_file():
        visa_meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
        categories = sorted(visa_meta["train"])
    else:
        categories = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not categories:
        raise SystemExit(f"No category directories found under {root}")
    for category in categories:
        candidates = normal_candidates(root, args.dataset, category)
        if len(candidates) < max(shots):
            raise SystemExit(
                f"{args.dataset}/{category}: {len(candidates)} normal train images, "
                f"need at least {max(shots)}"
            )
        rel_candidates = [
            p if isinstance(p, str) else p.relative_to(root).as_posix()
            for p in candidates
        ]
        manifest["categories"][category] = {}
        for seed in sorted(set(args.seeds)):
            shuffled = rel_candidates[:]
            random.Random(seed).shuffle(shuffled)
            selected = {}
            for shot in shots:
                selected[str(shot)] = shuffled[:shot]
            manifest["categories"][category][str(seed)] = selected

    if args.dataset in {"mpdd", "btad"}:
        selected_paths = sorted(
            {
                relative
                for seed_map in manifest["categories"].values()
                for shot_map in seed_map.values()
                for paths in shot_map.values()
                for relative in paths
            }
        )
        manifest["selected_file_sha256"] = {
            relative: hashlib.sha256(
                root.joinpath(*Path(relative).parts).read_bytes()
            ).hexdigest()
            for relative in selected_paths
        }

    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    payload_bytes = payload.encode("utf-8")
    destination = out_dir / "manifest.json"
    # Write the exact bytes that are hashed. Text-mode writes on Windows may
    # translate LF to CRLF and make the recorded digest disagree with the file.
    destination.write_bytes(payload_bytes)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    (out_dir / "manifest.sha256").write_bytes(
        f"{digest}  manifest.json\n".encode("ascii")
    )
    print(f"Wrote {destination}")
    print(f"SHA256 {digest}")
    print(f"Categories: {len(categories)}; shots: {shots}; seeds: {sorted(set(args.seeds))}")


if __name__ == "__main__":
    main()
