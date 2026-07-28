from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_unified.py"
SPEC = importlib.util.spec_from_file_location("evaluate_unified", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnifiedMetricTests(unittest.TestCase):
    def test_f1_max_perfect(self) -> None:
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(MODULE.f1_max(labels, scores), 1.0)

    def test_single_class_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires both"):
            MODULE.validate_binary_classes(np.zeros(4), "labels")

    def test_invalid_labels_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 0/1"):
            MODULE.validate_binary_classes(np.asarray([0, 2]), "labels")

    def test_map_shape_is_strict(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            MODULE.squeeze_maps(np.zeros((2, 2)), "maps")

    def test_aupro_perfect_map(self) -> None:
        masks = np.zeros((2, 1, 4, 4), dtype=np.uint8)
        masks[1, 0, 1:3, 1:3] = 1
        # Use continuous normal scores so the official thresholded AUPRO
        # algorithm has a non-zero FPR span. Every anomalous pixel still ranks
        # above every normal pixel, so the expected score is perfect.
        maps = np.linspace(0.0, 0.4, 32, dtype=np.float32).reshape(2, 4, 4)
        maps[1, 1:3, 1:3] = np.asarray(
            [[0.8, 0.85], [0.9, 1.0]], dtype=np.float32
        )
        self.assertAlmostEqual(MODULE.aupro_fast(masks, maps, 200), 1.0)


if __name__ == "__main__":
    unittest.main()
