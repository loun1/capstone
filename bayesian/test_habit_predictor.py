"""Ensure the chosen predictor reproduces its frozen benchmark and domain."""
import unittest
from unittest.mock import patch

import numpy as np

from . import habit_stress
from .habit_predictor import predict_habit_burden


class HabitPredictorTests(unittest.TestCase):
    def test_reproduces_the_original_quadratic_on_an_independent_ramp_pair(self):
        train = habit_stress._delta_samples(habit_stress.TRAIN_SEEDS, habit_stress.TRAIN_RAMPS)
        benchmark, _ = habit_stress._predict(train)
        with patch.object(habit_stress, '_delta_samples', return_value=train) as samples:
            report = predict_habit_burden(*habit_stress.TEST_RAMPS[0])
        samples.assert_called_once_with(habit_stress.TRAIN_SEEDS, habit_stress.TRAIN_RAMPS)
        self.assertEqual(report['method'], 'quadratic')
        values = [report['predicted_delta_vs_steady'][name] for name in habit_stress.METRICS]
        np.testing.assert_allclose(values, benchmark['quadratic'][0], rtol=1e-12, atol=1e-14)

    def test_steady_reference_is_exactly_zero(self):
        result = predict_habit_burden(*habit_stress.STEADY)['predicted_delta_vs_steady']
        self.assertEqual(result, dict.fromkeys(habit_stress.METRICS, 0.))

    def test_rejects_invalid_or_outside_ramps_before_fitting(self):
        with patch.object(habit_stress, '_delta_samples') as samples:
            for inputs in ((7.99, 11.), (12., 15.01), (float('nan'), 8.), (12., float('inf')), (True, 9.)):
                with self.subTest(inputs=inputs), self.assertRaises(ValueError):
                    predict_habit_burden(*inputs)
        samples.assert_not_called()


if __name__ == '__main__':
    unittest.main()
