"""Unit tests for calibration math. No model download required."""
from __future__ import annotations

import unittest

from metrics import (
    apply_knots,
    ece,
    lowest_threshold_for_precision,
    pav,
    undo_temperature,
    wilson_interval,
)


class EceTests(unittest.TestCase):
    def test_perfectly_calibrated_constant(self):
        conf = [0.9] * 100
        correct = [True] * 90 + [False] * 10
        self.assertAlmostEqual(ece(conf, correct), 0.0, places=12)

    def test_all_wrong_at_one(self):
        conf = [1.0] * 10
        correct = [False] * 10
        self.assertAlmostEqual(ece(conf, correct), 1.0, places=12)

    def test_two_bins(self):
        conf = [0.25] * 4 + [0.75] * 4
        correct = [True, False, False, False, True, True, True, False]
        # bin 0.2-0.3: n=4, acc=0.25, mean_c=0.25, contrib 0
        # bin 0.7-0.8: n=4, acc=0.75, mean_c=0.75, contrib 0
        self.assertAlmostEqual(ece(conf, correct), 0.0, places=12)


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        lo, hi = wilson_interval(90, 100)
        self.assertGreater(lo, 0.82)
        self.assertLess(lo, 0.85)
        self.assertGreater(hi, 0.94)
        self.assertLess(hi, 0.96)

    def test_empty(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 0.0))


class ThresholdTests(unittest.TestCase):
    def test_lowest_that_hits_target(self):
        conf = [0.5, 0.6, 0.9, 0.95, 0.99]
        correct = [False, True, True, True, True]
        hit = lowest_threshold_for_precision(conf, correct, 0.95)
        self.assertIsNotNone(hit)
        self.assertGreaterEqual(hit["accuracy"], 0.95)
        self.assertLessEqual(hit["threshold"], 0.9)

    def test_impossible_target(self):
        conf = [0.9, 0.9]
        correct = [True, False]
        self.assertIsNone(lowest_threshold_for_precision(conf, correct, 0.99))


class IsotonicTests(unittest.TestCase):
    def test_already_isotonic(self):
        knots = pav([0.2, 0.5, 0.8], [0.1, 0.4, 0.9])
        self.assertEqual(len(knots), 3)
        self.assertAlmostEqual(apply_knots(knots, 0.5), 0.4)

    def test_pools_violation(self):
        knots = pav([0.1, 0.2, 0.3], [0.9, 0.1, 0.8])
        ys = [apply_knots(knots, x) for x in (0.1, 0.2, 0.3)]
        self.assertEqual(ys, sorted(ys))


class TemperatureTests(unittest.TestCase):
    def test_undo_makes_peak_sharper(self):
        # A flat-ish 3-way softmax; T>1 flattened it, undoing should raise the max.
        probs = {"a": 0.50, "b": 0.30, "c": 0.20}
        raw_max = undo_temperature(probs, t=1.337398453966105)
        self.assertGreater(raw_max, 0.50)
        self.assertLess(raw_max, 1.0)


if __name__ == "__main__":
    unittest.main()
