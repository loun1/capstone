import csv
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np

from bayesian.driving import demo_trips, predict_next, read_trips


DEMO_CONTEXT = {
    "reference_vehicle": "reference", "target_vehicle": "personal",
    "route_id": "demo-city-loop", "direction": "clockwise", "traffic": "moderate",
    "distance_m": 3000.0, "duration_s": 320.0,
}


class DrivingTests(unittest.TestCase):
    def test_whole_trip_holdout_excludes_later_reference_and_personal_outcomes(self):
        trips = demo_trips()
        clean = predict_next(trips, **DEMO_CONTEXT)
        # This reference is still being recorded at the first held-out origin.
        first_test = next(trip for trip in trips if trip.trip_id == "personal-05")
        later = next(trip for trip in trips if trip.trip_id == "reference-07")
        corrupted = replace(later, cell_current_A=np.full_like(later.cell_current_A, 20.0))
        changed = predict_next([corrupted if trip is later else trip for trip in trips],
                               **DEMO_CONTEXT)
        self.assertEqual(later.starts_at, first_test.starts_at)
        self.assertEqual(clean.validation["test_folds"][0], changed.validation["test_folds"][0])
        self.assertNotEqual(clean.validation["test_folds"][1], changed.validation["test_folds"][1])

    def test_context_filter_and_as_of_exclude_unmatched_and_incomplete_trips(self):
        trips = demo_trips()
        last_reference = next(trip for trip in trips if trip.trip_id == "reference-08")
        final_personal = next(trip for trip in trips if trip.trip_id == "personal-06")
        cutoff = final_personal.completes_at
        # Include only trips completed by the cutoff, even when these rows exist.
        later_reference = replace(last_reference, starts_at=final_personal.starts_at,
                                  elapsed_s=last_reference.elapsed_s + np.linspace(0, 10, 121))
        unmatched = replace(final_personal, trip_id="unmatched", distance_m=3200)
        predicted = predict_next([later_reference if trip is last_reference else trip for trip in trips]
                                 + [unmatched], as_of=cutoff, **DEMO_CONTEXT)
        self.assertNotIn("reference-08", predicted.context["reference_training_trip_ids"])
        self.assertNotIn("unmatched", predicted.context["personal_training_trip_ids"])
        self.assertIn("personal-06", predicted.context["personal_training_trip_ids"])

    def test_deterministic_joint_paths_with_common_physical_time_grid(self):
        prediction = predict_next(demo_trips(), **DEMO_CONTEXT, samples=100)
        repeated = predict_next(demo_trips(), **DEMO_CONTEXT, samples=100)
        np.testing.assert_array_equal(prediction.sample_profiles, repeated.sample_profiles)
        times = prediction.expected_profile[:, 0]
        self.assertEqual((times[0], times[-1]), (0.0, 320.0))
        np.testing.assert_array_equal(prediction.sample_profiles[:, :, 0],
                                      np.broadcast_to(times, (100, len(times))))
        draws = prediction.sample_profiles[:, :, 1]
        self.assertGreater(np.corrcoef(draws[:, 40], draws[:, 41])[0, 1], 0.5)

    def test_csv_keeps_cell_current_sign_and_rejects_mixed_trip_metadata(self):
        header = ["vehicle_id", "driver_id", "trip_id", "starts_at", "route_id", "direction",
                  "traffic", "distance_m", "cell_basis_id", "elapsed_s", "cell_current_A"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trips.csv"
            prefix = ["vehicle-a", "driver-a", "trip-1", "2026-11-06T08:00:00+09:00",
                      "route-a", "west", "light", 3000, "synthetic-m50t-single-cell"]
            rows = [prefix + [time, current]
                    for time, current in [(0, 0), (5, -1.5), (10, 0.3)]]
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows([header, *rows])
            trip, = read_trips(path)
            np.testing.assert_array_equal(trip.cell_current_A, [0, -1.5, 0.3])
            with self.assertRaisesRegex(ValueError, "Sample gap"):
                read_trips(path, max_sample_gap_s=4)
            rows[1][6] = "heavy"
            with path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows([header, *rows])
            with self.assertRaisesRegex(ValueError, "Inconsistent metadata"):
                read_trips(path)

    def test_shared_vehicle_and_different_cell_basis_are_not_pooled(self):
        trips = demo_trips()
        personal = next(trip for trip in trips if trip.trip_id == "personal-01")
        another_driver = replace(personal, driver_id="someone-else", trip_id="another-driver")
        with self.assertRaisesRegex(ValueError, "Select one driver"):
            predict_next(trips + [another_driver], **DEMO_CONTEXT)
        predicted = predict_next(trips + [another_driver], target_driver="personal-driver",
                                 **DEMO_CONTEXT)
        self.assertEqual(predicted.context["target_driver"], "personal-driver")
        mismatched = replace(personal, cell_basis_id="other-cell")
        with self.assertRaisesRegex(ValueError, "same model cell basis"):
            predict_next([mismatched if trip is personal else trip for trip in trips],
                         **DEMO_CONTEXT)


if __name__ == "__main__":
    unittest.main()
