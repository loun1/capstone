import unittest

import numpy as np

from .habit_life import _charge_knots, build_daily_duty
from .habit_stress import FACTORIAL_SEEDS


class HabitLifeTests(unittest.TestCase):
    def test_zero_crossing_preserves_discharge_and_regeneration(self):
        time, charge = _charge_knots(np.array([0., 2.]), np.array([1., -1.]))
        np.testing.assert_array_equal(time, [0., 1., 2.])
        np.testing.assert_allclose(charge, [0., .5 / 3600, 0.], rtol=0, atol=1e-16)

    def test_closed_daily_duty_keeps_traction_ah_and_matched_recharge(self):
        time = np.arange(251, dtype=float)
        current = np.full((len(FACTORIAL_SEEDS), len(time)), .12)
        grid, charge, policy = build_daily_duty(time, current)
        self.assertEqual(policy["trips_per_day"], 96)
        self.assertEqual(policy["driving_duration_s"], 24000.)
        self.assertAlmostEqual(policy["drive_discharge_Ah"], .8)
        self.assertAlmostEqual(policy["charge_added_Ah"], .8)
        self.assertAlmostEqual(policy["absolute_throughput_including_recharge_Ah"], 1.6)
        self.assertAlmostEqual(policy["recharge_end_s"], 30720.)
        self.assertEqual((grid[0], grid[-1], charge[0], charge[-1]), (0., 86400., 0., 0.))


if __name__ == "__main__":
    unittest.main()
