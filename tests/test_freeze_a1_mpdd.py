"""CPU tests for the S1 read-only freeze verifier (scripts/freeze_a1_mpdd.py).

Verifies that --verify never rewrites the manifest and that it reports
missing / size mismatch / hash mismatch / extra undeclared .npz entries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import freeze_a1_mpdd as freeze  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a small fake artifact tree under tmp_path; returns (root, manifest_path)."""
    (tmp_path / "code").mkdir(parents=True)
    (tmp_path / "cache" / "s0_k1" / "anomalydino_visual").mkdir(parents=True)
    (tmp_path / "cache" / "s0_k1" / "anomalyclip_text").mkdir(parents=True)

    code_file = tmp_path / "code" / "foo.py"
    code_file.write_bytes(b"FOO")

    a_npz = tmp_path / "cache" / "s0_k1" / "anomalydino_visual" / "a.npz"
    a_npz.write_bytes(b"AAA")
    b_npz = tmp_path / "cache" / "s0_k1" / "anomalyclip_text" / "b.npz"
    b_npz.write_bytes(b"BBB")

    def entry(rel: str) -> dict:
        path = tmp_path / rel
        return {"relative_path": rel, "size_bytes": path.stat().st_size, "sha256": freeze.sha256(path)}

    manifest = {
        "status": "frozen",
        "code": [entry("code/foo.py")],
        "checkpoints": [],
        "manifests": [],
        "evaluators": [],
        "feature_caches": {
            "s0_k1": {
                "anomalydino_visual": [entry("cache/s0_k1/anomalydino_visual/a.npz")],
                "anomalyclip_text": [entry("cache/s0_k1/anomalyclip_text/b.npz")],
            }
        },
        "baseline_prediction_caches": {},
    }
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path, manifest_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_verify_passes_and_leaves_manifest_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    root, manifest_path = build_tree(tmp_path)
    assert root == tmp_path

    before = freeze.sha256(manifest_path)
    report = freeze.verify_manifest(manifest_path)
    after = freeze.sha256(manifest_path)

    assert report["all_ok"] is True
    assert report["verified_entries"] == 3
    assert report["missing"] == []
    assert report["size_mismatch"] == []
    assert report["hash_mismatch"] == []
    assert report["extra_undeclared_npz"] == []
    assert after == before, "verify must not rewrite the manifest"


def test_verify_detects_tampered_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    _, manifest_path = build_tree(tmp_path)

    # same size (3 bytes), different content -> hash mismatch only
    (tmp_path / "code" / "foo.py").write_bytes(b"BAR")
    report = freeze.verify_manifest(manifest_path)

    assert report["all_ok"] is False
    assert len(report["hash_mismatch"]) == 1
    assert report["hash_mismatch"][0]["relative_path"] == "code/foo.py"
    assert report["size_mismatch"] == []


def test_verify_detects_missing_file(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    _, manifest_path = build_tree(tmp_path)

    (tmp_path / "cache" / "s0_k1" / "anomalyclip_text" / "b.npz").unlink()
    report = freeze.verify_manifest(manifest_path)

    assert report["all_ok"] is False
    assert len(report["missing"]) == 1
    assert report["missing"][0]["relative_path"] == "cache/s0_k1/anomalyclip_text/b.npz"


def test_verify_detects_size_mismatch(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    root, manifest_path = build_tree(tmp_path)

    # corrupt declared size for a.npz (real file stays intact)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["feature_caches"]["s0_k1"]["anomalydino_visual"][0]["size_bytes"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert root == tmp_path

    report = freeze.verify_manifest(manifest_path)
    assert report["all_ok"] is False
    assert len(report["size_mismatch"]) == 1
    assert report["size_mismatch"][0]["relative_path"] == "cache/s0_k1/anomalydino_visual/a.npz"


def test_verify_detects_extra_undeclared_npz(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    _, manifest_path = build_tree(tmp_path)

    (tmp_path / "cache" / "s0_k1" / "anomalydino_visual" / "extra.npz").write_bytes(b"X")
    report = freeze.verify_manifest(manifest_path)

    assert report["all_ok"] is False
    assert "cache/s0_k1/anomalydino_visual/extra.npz" in report["extra_undeclared_npz"]


def test_verify_reports_broken_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    manifest_path = tmp_path / "freeze_manifest.json"
    manifest_path.write_text("{not valid json", encoding="utf-8")

    report = freeze.verify_manifest(manifest_path)
    assert report["all_ok"] is False
    assert "error" in report


def test_create_and_verify_are_mutually_exclusive(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    with pytest.raises(SystemExit):
        freeze.main(["--create", "--verify", "--output", str(tmp_path / "m.json")])


def test_verify_missing_manifest_fails(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(freeze, "ROOT", tmp_path)
    # missing manifest -> non-zero exit code from main (no exception)
    assert freeze.main(["--verify", "--output", str(tmp_path / "nonexistent.json")]) == 1
