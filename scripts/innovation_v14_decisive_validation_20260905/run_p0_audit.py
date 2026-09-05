"""P0 data-role audit -> DATA_ROLE_AUDIT.json (doc28 s4.2). CPU, read-only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "innovation_v14_decisive_validation_20260905"))

from v14_common import CATEGORIES, load_manifest, support_paths, assert_fit_ids_are_support  # noqa: E402

OUT = ROOT / "experiments/dynamic_fusion/innovation_v14_decisive_validation_20260905/DATA_ROLE_AUDIT.json"


def main() -> int:
    m = load_manifest()
    checks = []
    all_ok = True
    for cat in CATEGORIES:
        for shot in (1, 2, 4):
            rels = support_paths(cat, shot, "0", m)
            leak = any("/test/" in r for r in rels)
            ok = (not leak) and all("/train/good/" in r for r in rels) and len(rels) == shot
            all_ok &= ok
            checks.append({"cat": cat, "seed": "0", "shot": shot, "n_support": len(rels),
                           "all_train_good": all("/train/good/" in r for r in rels),
                           "no_test": not leak, "ok": ok})
            try:
                assert_fit_ids_are_support(rels, cat, shot, "0")
            except ValueError as e:
                all_ok = False
                print("FAIL", cat, shot, e)
    payload = {"date": "2026-09-05", "all_ok": bool(all_ok), "checks": checks,
               "rule": "fit/select IDs must be K-shot support (train/good); /test/ -> fail",
               "v13_notes": {"N3_used_test_good_syn_cache": True,
                             "downgraded_to_offline_mechanism_exploration": True,
                             "N2_was_balanced_OT_not_semi_relaxed": True}}
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"all_ok": payload["all_ok"], "n_checks": len(checks)}))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
