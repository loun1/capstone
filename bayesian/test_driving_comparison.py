import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bayesian.driving_comparison import compare_profiles, summarize_meshes, validate_comparison


CONTEXT = {"route_id": "declared-test-route", "direction": "outbound", "traffic": "moderate",
           "distance_m": 3000.0, "cell_basis_id": "okane2022-spme-single-cell",
           "source_kind": "synthetic_unit_fixture"}


class DrivingComparisonTests(unittest.TestCase):
    def setUp(self):
        self.profiles = {"steady": np.array([[0., 1.], [150., 1.], [300., 1.]]),
                         "varying": np.array([[0., 0.], [150., 2.], [300., 0.]])}

    def test_equal_task_checks_integral_without_rescaling_profiles(self):
        original = copy.deepcopy(self.profiles)
        clean, duty = validate_comparison(self.profiles, CONTEXT)
        self.assertTrue(duty["net_charge_matched"])
        self.assertAlmostEqual(duty["signed_drive_Ah"]["steady"], 1/12)
        for name in clean:
            np.testing.assert_array_equal(clean[name], original[name])
            self.assertIsNot(clean[name], self.profiles[name])

    def test_unequal_net_charge_is_recorded_not_silently_normalized(self):
        self.profiles["varying"][:, 1] *= 1.2
        clean, duty = validate_comparison(self.profiles, CONTEXT)
        self.assertFalse(duty["net_charge_matched"])
        self.assertGreater(duty["net_charge_spread_Ah"], .01)
        np.testing.assert_array_equal(clean["varying"], self.profiles["varying"])

    def test_invalid_profiles_rejected_before_simulation(self):
        bad = [np.array([[1., 0.], [2., 1.], [3., 0.]]),
               np.array([[0., 1.], [0., 2.], [3., 1.]]),
               np.array([[0., 1.], [150., float("nan")], [300., 0.]]),
               np.array([[0., 0.], [150., 6.], [300., 0.]]),
               np.array([[0., 0.], [500., 1.], [1000., 0.]]),
               np.array([[0., 0.], [150., 1.], [301., 0.]])]
        for curve in bad:
            with self.subTest(curve=curve), self.assertRaises(ValueError):
                validate_comparison({"steady": self.profiles["steady"], "other": curve}, CONTEXT)

    def test_declared_route_and_cell_basis_required(self):
        for key in CONTEXT:
            with self.subTest(missing=key):
                context = dict(CONTEXT)
                del context[key]
                with self.assertRaises(ValueError):
                    validate_comparison(self.profiles, context)
        context = {**CONTEXT, "cell_basis_id": "real_model_y_pack"}
        with self.assertRaises(ValueError):
            validate_comparison(self.profiles, context)

    def test_bounded_horizon_temperature_and_duty_tolerance(self):
        for settings in ({"horizon_s": 300}, {"horizon_s": 20000}, {"temperature_C": 0},
                         {"temperature_C": 50}, {"duty_tolerance_Ah": .1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                validate_comparison(self.profiles, CONTEXT, **settings)

    @staticmethod
    def mesh_rows(delta=0., valid=True):
        def arm(q):
            return {"rpt_capacity_Ah": q, "physical": {"physical_constraints_valid": valid}}
        return [{"arms": {"no_drive_control": arm(5.), "steady": arm(4.9999), "varying": arm(4.9995)}},
                {"arms": {"no_drive_control": arm(5.), "steady": arm(4.9999 + delta), "varying": arm(4.9995 - delta)}}]

    def test_ranking_needs_matching_duty_and_resolved_two_meshes(self):
        _, duty = validate_comparison(self.profiles, CONTEXT)
        result = summarize_meshes(self.mesh_rows(), duty)
        self.assertEqual(result["duty_matched_ranking"], ["steady", "varying"])
        self.assertIsNone(result["recommendation"])
        self.assertIsNone(result["lifetime_extension_days"])
        for rows in (self.mesh_rows(valid=False), self.mesh_rows(delta=.001), self.mesh_rows()[:1]):
            self.assertIsNone(summarize_meshes(rows, duty)["duty_matched_ranking"])
        self.assertIsNone(summarize_meshes(self.mesh_rows(), {**duty, "net_charge_matched": False})["duty_matched_ranking"])

    def test_control_subtracts_common_horizon_capacity_without_initial_rpt_mismatch(self):
        _, duty = validate_comparison(self.profiles, CONTEXT)
        result = summarize_meshes(self.mesh_rows(), duty)
        self.assertAlmostEqual(result["cases"]["steady"]["capacity_difference_from_no_drive_control_pp"], .002)
        self.assertAlmostEqual(result["cases"]["varying"]["capacity_difference_from_no_drive_control_pp"], .01)

    def test_failures_are_saved_and_exclusive_output_preserves_prior_result(self):
        # Keep atomic-write fixtures in the writable checkout on Windows.
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as tmp:
            output = Path(tmp) / "comparison"
            with patch("bayesian.driving_comparison.Engine", side_effect=RuntimeError("controlled solver failure")):
                with self.assertRaisesRegex(RuntimeError, "controlled solver failure"):
                    compare_profiles(self.profiles, CONTEXT, output)
            path = output / "driving_comparison.json"
            saved = path.read_bytes()
            result = json.loads(saved)
            self.assertEqual(result["status"], "failed")
            self.assertIn("controlled solver failure", result["error"])
            self.assertFalse(result["physical_model_compatibility_verified"])
            with self.assertRaises(FileExistsError):
                compare_profiles(self.profiles, CONTEXT, output)
            self.assertEqual(path.read_bytes(), saved)


if __name__ == "__main__":
    unittest.main()
