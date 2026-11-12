import unittest
from dataclasses import replace

import numpy as np

from simulation.gp_internal_state import Z95, Intervals, fit_residual_gp, predict_intervals, score


class GPInternalStateTests(unittest.TestCase):
    def setUp(self):
        features = np.column_stack((np.arange(5) * 80., np.full(5, 80.),
                                    np.full(5, 25.), np.full(5, 25.)))
        physics = np.array([1.8, 1.2, .9, .8, .7])
        observed = physics + np.array([.32, .41, .48, .53, .56])
        self.train = Intervals("C", 25., features, observed, physics)
        self.fitted = fit_residual_gp(self.train)
        self.target = replace(self.train, cell="D")

    def test_prediction_does_not_use_heldout_observations(self):
        prediction = predict_intervals(self.fitted, self.target.features, self.target.nominal_temperature_C)
        expected = score(self.target, self.fitted, prediction)["intervals"]
        altered = replace(self.target, observed_delta_pp=np.full(5, 1234.))
        actual = score(altered, self.fitted, prediction)["intervals"]
        for left, right in zip(expected, actual):
            self.assertEqual(left["predicted_delta_pp"], right["predicted_delta_pp"])
            self.assertNotEqual(left["observed_DMA_LLI_delta_pp"], right["observed_DMA_LLI_delta_pp"])
            self.assertEqual(right["review_status"], "review_dma_disagreement")

    def test_covariance_propagates_to_cumulative_output(self):
        prediction = predict_intervals(self.fitted, self.target.features, self.target.nominal_temperature_C)
        _, latent, observed, _, _ = prediction
        self.assertGreaterEqual(np.linalg.eigvalsh(latent).min(), -1e-12)
        self.assertGreater(np.linalg.eigvalsh(observed).min(), 0)
        result = score(self.target, self.fitted, prediction)
        width = result["intervals"][-1]["gp_cumulative_predictive_half_width_pp"]
        self.assertAlmostEqual(width, Z95 * np.sqrt(observed.sum()))

    def test_cold_gate_leaves_frozen_proxy_and_avoids_fault_flag(self):
        cold = replace(self.target, cell="A", nominal_temperature_C=10.)
        prediction = predict_intervals(self.fitted, cold.features, cold.nominal_temperature_C)
        result = score(cold, self.fitted, prediction)
        for row in result["intervals"]:
            predicted = row["predicted_delta_pp"]
            self.assertEqual(row["review_status"], "insufficient_training_support")
            self.assertEqual(predicted["frozen_calendar_sei_proxy"],
                             predicted["gp_when_supported_else_physics"])


if __name__ == "__main__":
    unittest.main()
