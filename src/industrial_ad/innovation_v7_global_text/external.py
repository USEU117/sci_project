"""Task book 17 Phase 0 gate: external-set execution is disabled by default and
only allowed once a freeze manifest exists for the given phase output.
"""

from __future__ import annotations

from pathlib import Path

EXTERNAL_ALLOWED = False  # hard default off


def require_freeze_manifest(manifest_path: Path) -> None:
    """Raise unless a freeze manifest exists and the switch is on."""
    if not EXTERNAL_ALLOWED:
        raise RuntimeError("external validation disabled (EXTERNAL_ALLOWED=False)")
    if not Path(manifest_path).is_file():
        raise RuntimeError(f"freeze manifest missing: {manifest_path}")
