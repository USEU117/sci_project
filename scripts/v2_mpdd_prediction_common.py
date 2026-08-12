"""Shared, leakage-explicit MPDD/BTAD indexing for V2 prediction exporters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class TestSample:
    category: str
    anomaly_type: str
    image_path: Path
    mask_path: Path | None
    sample_id: str
    label: int


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def index_mpdd(data_root: Path) -> dict[str, list[TestSample]]:
    indexed: dict[str, list[TestSample]] = {}
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        test_root = category_dir / "test"
        if not test_root.is_dir():
            continue
        samples: list[TestSample] = []
        for type_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            label = 0 if type_dir.name == "good" else 1
            for image_path in sorted(
                path for path in type_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                mask_path = None
                if label:
                    candidate = category_dir / "ground_truth" / type_dir.name / f"{image_path.stem}_mask.png"
                    if not candidate.is_file():
                        raise FileNotFoundError(f"ground-truth mask missing: {candidate}")
                    mask_path = candidate
                relative = image_path.relative_to(data_root).as_posix()
                samples.append(
                    TestSample(
                        category=category_dir.name,
                        anomaly_type=type_dir.name,
                        image_path=image_path,
                        mask_path=mask_path,
                        sample_id=relative,
                        label=label,
                    )
                )
        if samples:
            indexed[category_dir.name] = samples
    return indexed


def index_btad(data_root: Path) -> dict[str, list[TestSample]]:
    indexed: dict[str, list[TestSample]] = {}
    for category_dir in sorted(path for path in data_root.iterdir() if path.is_dir()):
        test_root = category_dir / "test"
        if not test_root.is_dir():
            continue
        samples: list[TestSample] = []
        for type_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
            label = 0 if type_dir.name == "ok" else 1
            for image_path in sorted(
                path for path in type_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ):
                mask_path = None
                if label:
                    mask_root = category_dir / "ground_truth" / "ko"
                    candidates = sorted(
                        path for path in mask_root.glob(f"{image_path.stem}.*")
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                    )
                    if len(candidates) != 1:
                        raise FileNotFoundError(
                            f"expected exactly one ground-truth mask for {image_path}, found {len(candidates)}"
                        )
                    mask_path = candidates[0]
                relative = image_path.relative_to(data_root).as_posix()
                samples.append(TestSample(category_dir.name, type_dir.name, image_path, mask_path, relative, label))
        if samples:
            indexed[category_dir.name] = samples
    return indexed


def index_dataset(dataset: str, data_root: Path) -> dict[str, list[TestSample]]:
    if dataset == "mpdd":
        return index_mpdd(data_root)
    if dataset == "btad":
        return index_btad(data_root)
    raise ValueError(f"unsupported dataset: {dataset}")


def validate_dataset_gate_inputs(dataset: str, data_root: Path, manifest: dict, seed: int, shot: int) -> dict:
    if manifest.get("dataset") != dataset:
        raise ValueError(f"manifest dataset differs: expected {dataset}")
    indexed = index_dataset(dataset, data_root)
    if set(indexed) != set(manifest["categories"]):
        raise ValueError(f"dataset categories differ from the frozen {dataset.upper()} manifest")
    normal_dir = "good" if dataset == "mpdd" else "ok"
    reference_paths = 0
    for category in sorted(indexed):
        selected = manifest["categories"][category][str(seed)][str(shot)]
        reference_paths += len(selected)
        for relative in selected:
            path = data_root / relative
            if not path.is_file():
                raise FileNotFoundError(f"normal reference missing: {path}")
            normalized = Path(relative).as_posix()
            if f"/train/{normal_dir}/" not in f"/{normalized}":
                raise ValueError(f"reference is not from train/{normal_dir}: {relative}")
    return {
        "categories": len(indexed),
        "test_images": sum(len(samples) for samples in indexed.values()),
        "anomalous_images": sum(sample.label for samples in indexed.values() for sample in samples),
        "normal_references": reference_paths,
    }


def validate_mpdd_gate_inputs(data_root: Path, manifest: dict, seed: int, shot: int) -> dict:
    return validate_dataset_gate_inputs("mpdd", data_root, manifest, seed, shot)
