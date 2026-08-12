"""CPU-only MVTec-only re-evaluation of combined AnomalyDINO cache directories."""
from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ("bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper")


def valid(path: Path) -> bool:
    report = path / "evaluation_report.json"
    if not report.exists():
        return False
    data = json.loads(report.read_text(encoding="utf-8"))
    return data.get("category_count") == 15 and data.get("sample_count") == 1725 and data.get("validation_errors") == 0


def main() -> None:
    logs = ROOT / "outputs/logs/anomalydino_mvtec_subset_evaluation_20260808.log"
    logs.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    for seed in (0, 1, 2):
        for shot in (1, 2, 4):
            cache = ROOT / f"outputs/anomalydino/unified_matrix/seed_{seed}_shot_{shot}/predictions"
            out = ROOT / f"outputs/unified/anomalydino_mvtec_seed_{seed}_shot_{shot}"
            if valid(out):
                entries.append({"seed": seed, "shot": shot, "status": "skipped_valid"})
                continue
            command = [sys.executable, str(ROOT / "scripts/evaluate_unified.py"), "--cache-dir", str(cache), "--output-dir", str(out), "--workers", "1", "--include-categories", *CATEGORIES]
            with logs.open("a", encoding="utf-8") as stream:
                stream.write(subprocess.list2cmdline(command) + "\n")
                result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, text=True)
            entries.append({"seed": seed, "shot": shot, "status": "passed" if result.returncode == 0 and valid(out) else "failed", "exit_code": result.returncode})
            if entries[-1]["status"] == "failed":
                raise RuntimeError(f"MVTec subset evaluation failed for s{seed}_k{shot}")
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "dataset": "mvtec", "categories": list(CATEGORIES), "entries": entries, "gpu_used": False}
    target = ROOT / "experiments/summaries/anomalydino_mvtec_subset_evaluation_20260808.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
