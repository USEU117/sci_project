"""Build a SHA256 provenance manifest for every frozen fusion input cache."""
from __future__ import annotations

import csv
import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "outputs/dynamic_fusion/final_validation"
OUTPUT = ROOT / "experiments/summaries/dynamic_fusion_scientific_analysis_20260809/input_provenance_sha256.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_run(name: str) -> tuple[str, int, int]:
    dataset = "mvtec" if "mvtec" in name else "visa"
    seed = int(re.search(r"_s(\d+)_", name).group(1))
    shot = int(re.search(r"_k(\d+)$", name).group(1))
    return dataset, seed, shot


def main() -> int:
    rows: list[dict] = []
    digest_cache: dict[Path, str] = {}
    directories = [
        path for path in sorted(FINAL.iterdir())
        if path.is_dir() and ("mvtec_final" in path.name or "visa_final" in path.name)
    ]
    for run_index, directory in enumerate(directories, 1):
        dataset, seed, shot = parse_run(directory.name)
        print(f"[{run_index}/{len(directories)}] {directory.name}", flush=True)
        for fused in sorted(directory.glob("*.npz")):
            category = fused.stem
            visual = ROOT / f"outputs/anomalydino/unified_matrix/seed_{seed}_shot_{shot}/predictions/{category}.npz"
            if dataset == "mvtec":
                text = ROOT / f"outputs/anomalyclip/mvtec_npz/{category}.npz"
                sidecar = ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_mvtec_518/{category}.sample_ids.npz"
            else:
                text = ROOT / f"outputs/anomalyclip/visa_all_518_cached/{category}.npz"
                sidecar = ROOT / f"outputs/dynamic_fusion/sidecars/anomalyclip_visa_518_verified/{category}.sample_ids.npz"
            for role, path in (("visual_prediction", visual), ("text_prediction", text), ("text_sample_id_sidecar", sidecar)):
                resolved = path.resolve()
                if not resolved.exists():
                    raise FileNotFoundError(resolved)
                if resolved not in digest_cache:
                    digest_cache[resolved] = sha256(resolved)
                stat = resolved.stat()
                rows.append({
                    "run_id": directory.name, "dataset": dataset, "seed": seed,
                    "shot": shot, "category": category, "role": role,
                    "path": str(resolved), "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns, "sha256": digest_cache[resolved],
                })
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} unique_files={len(digest_cache)} output={OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
