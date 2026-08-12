from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def make_dataset(root: Path, categories: int, normal_name: str) -> None:
    for index in range(categories):
        category = root / f"category_{index:02d}"
        for image_index in range(5):
            path = category / "train" / normal_name / f"{image_index:03d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"normal-{index}-{image_index}".encode())
        normal_test = category / "test" / normal_name / "normal.png"
        abnormal_test = category / "test" / "defect" / "bad.png"
        mask = category / "ground_truth" / "defect" / "bad_mask.png"
        for path in (normal_test, abnormal_test, mask):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(path.as_posix().encode())


@pytest.mark.parametrize(
    ("dataset", "categories", "normal_name"),
    [("mpdd", 6, "good"), ("btad", 3, "ok")],
)
def test_new_dataset_manifest_and_validation_contract(
    tmp_path: Path, dataset: str, categories: int, normal_name: str
) -> None:
    data_root = tmp_path / dataset
    split_root = tmp_path / "splits"
    report_root = tmp_path / "reports"
    make_dataset(data_root, categories, normal_name)
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_dataset.py"),
            "--dataset",
            dataset,
            "--root",
            str(data_root),
            "--output",
            str(report_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_splits.py"),
            "--dataset",
            dataset,
            "--root",
            str(data_root),
            "--output",
            str(split_root),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest_path = split_root / dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["categories"]) == categories
    assert manifest["selected_file_sha256"]
    for seed_map in manifest["categories"].values():
        for selections in seed_map.values():
            assert set(selections["1"]).issubset(selections["2"])
            assert set(selections["2"]).issubset(selections["4"])
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_splits.py"),
            str(manifest_path),
            "--output",
            str(report_root / f"{dataset}_split_validation.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_safe_archive_preparation_discovers_single_dataset_root(tmp_path: Path) -> None:
    source = tmp_path / "source"
    make_dataset(source / "MPDD", 6, "good")
    archive = tmp_path / "mpdd.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as handle:
        for path in (source / "MPDD").rglob("*"):
            if path.is_file():
                handle.write(path, path.relative_to(source).as_posix())
    destination = tmp_path / "extracted"
    report = tmp_path / "archive_report.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_v2_dataset_archive.py"),
            "--dataset",
            "mpdd",
            "--archive",
            str(archive),
            "--destination",
            str(destination),
            "--source-url",
            "https://example.invalid/mpdd.zip",
            "--source-kind",
            "mirror",
            "--output",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["category_count"] == 6
    assert Path(payload["dataset_root"]).name == "MPDD"
