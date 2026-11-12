"""Guard the matched waveform benchmark and its excess-charge definition."""
from dataclasses import replace
import unittest

import numpy as np

from . import habit_waveform as waveform


class HabitWaveformTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits = waveform.build_waveform_data()

    def test_both_forecasts_share_the_exact_training_reference(self):
        train = self.splits['train']
        reference = train.reference_current_A.mean(axis=0)
        for method in waveform.MODEL_BOUNDS:
            with self.subTest(method=method):
                model = waveform.fit_waveform(train, method)
                np.testing.assert_array_equal(model.predict(16., 15.)[:, 1], reference)
                np.testing.assert_array_equal(model.predict_residual(16., 15.)[:, 1], 0.)
                np.testing.assert_allclose(model.reference_profile()[:, 1] + model.predict_residual(9.5, 10.5)[:, 1],
                                           model.predict(9.5, 10.5)[:, 1])

    def test_scoring_changed_test_curves_cannot_change_forecast(self):
        train, validation, test = (self.splits[name] for name in ('train', 'validation', 'test'))
        for left, right in ((train, validation), (train, test), (validation, test)):
            self.assertFalse(set(left.seeds) & set(right.seeds))
            self.assertFalse(set(left.ramps_s) & set(right.ramps_s))
        model = waveform.fit_waveform(train, 'gp')
        first = waveform.score_waveform(model, test, include_burdens=False)
        changed = waveform.score_waveform(model, replace(test, target_current_A=test.target_current_A + .1),
                                          include_burdens=False)
        self.assertEqual(first['predicted_current_A'], changed['predicted_current_A'])
        self.assertNotEqual(first['episode_rmse_A'], changed['episode_rmse_A'])
        with self.assertRaises(ValueError):
            waveform.fit_waveform(test, 'quadratic')

    def test_positive_discharge_excess_handles_crossings_and_regen(self):
        reference = np.zeros((2, len(waveform.ELAPSED_S)))
        target = np.zeros_like(reference)
        reference[0, :2], target[0, :2] = [-1., 1.], [-.5, 2.]
        reference[1], target[1] = -2., -1.
        metrics = waveform.paired_charge_metrics(target, reference)
        np.testing.assert_allclose(metrics['signed_net_delta_Ah'], [1.25 / 3600, 250 / 3600])
        np.testing.assert_allclose(metrics['positive_discharge_excess_Ah'], [1.05 / 3600, 0.])


if __name__ == '__main__':
    unittest.main()
