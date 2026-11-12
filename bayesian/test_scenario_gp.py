"""Test same-route generation and isolation of the synthetic test split."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from . import scenario_gp


class ScenarioGPTests(unittest.TestCase):
    def test_paired_speed_covers_each_declared_section(self):
        route, traffic, seed = scenario_gp.DEMO_EPISODE
        episode = scenario_gp.generate_episode(route, traffic, seed)
        lengths, _ = scenario_gp.ROUTES[route]
        for start, duration, distance in zip(episode.schedule['start_s'], episode.schedule['travel_s'], lengths):
            for speeds in (episode.reference_speed_m_s, episode.target_speed_m_s):
                section = speeds[start:start + duration + 1]
                self.assertEqual(section[0], 0.)
                self.assertEqual(section[-1], 0.)
                self.assertAlmostEqual(float(np.trapezoid(section)), distance, places=8)
        self.assertFalse(np.array_equal(episode.reference_A, episode.target_A))

    def test_holdout_outcomes_do_not_change_any_forecast(self):
        original_generator = scenario_gp.generate_episode
        first = scenario_gp.evaluate_scenarios()

        def changed_holdout(route, traffic, seed):
            episode = original_generator(route, traffic, seed)
            if seed in scenario_gp.TEST_SEEDS:
                return replace(episode, target_A=episode.target_A + .1)
            return episode

        with patch.object(scenario_gp, 'generate_episode', side_effect=changed_holdout):
            second = scenario_gp.evaluate_scenarios()

        for key, row in first['conditions'].items():
            self.assertEqual(row['forecasts_A'], second['conditions'][key]['forecasts_A'])
            self.assertEqual(row['gp_interval_90_half_width_A'], second['conditions'][key]['gp_interval_90_half_width_A'])
        self.assertNotEqual(first['metrics'], second['metrics'])


if __name__ == '__main__':
    unittest.main()
