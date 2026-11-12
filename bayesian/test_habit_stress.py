"""Check load integrals and independence of the habit prediction benchmark."""
import unittest
from unittest.mock import patch

import numpy as np

from . import habit_stress


class HabitStressTests(unittest.TestCase):
    def test_current_integrals_split_a_sign_crossing_exactly(self):
        metrics = habit_stress.current_burdens(np.array([[0., 2.], [2., -1.], [5., -1.]]))
        self.assertEqual(metrics['peak_discharge_A'], 2.)
        self.assertEqual(metrics['peak_regen_A'], 1.)
        self.assertAlmostEqual(metrics['i_squared_A2s'], 5.)
        self.assertAlmostEqual(metrics['absolute_throughput_Ah'], (5 / 3 + 3) / 3600)

    def test_habit_arms_travel_the_same_distance_in_each_section(self):
        schedule = habit_stress._environment(habit_stress.DEMO_SEED)
        lengths, _ = habit_stress.scenario.ROUTES[habit_stress.ROUTE]
        for ramps in habit_stress.FACTORIAL_ARMS.values():
            _, speed = habit_stress._driving_curve(schedule, ramps)
            for start, duration, length in zip(schedule['start_s'], schedule['travel_s'], lengths):
                samples = speed[start:start + duration + 1]
                self.assertEqual(samples[0], 0.)
                self.assertEqual(samples[-1], 0.)
                self.assertAlmostEqual(float(np.trapezoid(samples)), length)

    def test_heldout_burdens_do_not_refit_any_prediction(self):
        baseline = habit_stress.evaluate_habit_stress()['gp']
        original = habit_stress._delta_samples

        def changed_test(seeds, ramps):
            samples = original(seeds, ramps)
            return samples + .1 if seeds == habit_stress.TEST_SEEDS else samples

        with patch.object(habit_stress, '_delta_samples', side_effect=changed_test):
            changed = habit_stress.evaluate_habit_stress()['gp']
        self.assertEqual(baseline['fit_evidence'], changed['fit_evidence'])
        self.assertEqual(baseline['test_prediction'], changed['test_prediction'])
        self.assertNotEqual(baseline['test_scores'], changed['test_scores'])


if __name__ == '__main__':
    unittest.main()
