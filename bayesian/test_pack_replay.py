"""Replay selection, provenance, and failure reporting without extra PDE runs."""
import contextlib
import copy
import io
import json
from unittest import mock
import unittest

import numpy as np

from . import test_pack_input as fixtures
from .pack_input import load_pack_input
from .pack_replay import run_replay, select_segment


class PackReplayTests(unittest.TestCase):
    write_inputs = fixtures.PackInputTests.write_inputs

    def setUp(self):
        fixtures.PackInputTests.setUp(self)
        self.meta["completed_cycle_metrics"][0]["initial_soc_bms_pct"] = 80
        self.meta["completed_cycle_metrics"][1]["initial_soc_bms_pct"] = 73.5
        self.write_inputs()

    def loaded(self):
        return load_pack_input(self.csv_path, self.meta_path, self.config)

    def run_replay(self, output=None, **kwargs):
        return run_replay(self.csv_path, self.meta_path, self.config_path,
                          output or self.root / "replay", **kwargs)

    @staticmethod
    def simulated(segment, _):
        trace = []
        capacity = 0.0
        for start, end, current in segment["intervals"]:
            before = capacity
            capacity += current * (end - start) / 3600
            trace.append({"elapsed_start_s": float(start), "elapsed_end_s": float(end),
                          "cell_current_a": float(current), "model_current_end_a": float(current),
                          "model_voltage_start_v": 3.9, "model_voltage_end_v": 3.8,
                          "model_discharge_capacity_start_ah": float(before),
                          "model_discharge_capacity_end_ah": float(capacity),
                          "charge_balance_residual_ah": 0.0})
        return trace, {"test_stub": True}

    def test_selection_keeps_only_complete_source_intervals_without_resampling(self):
        cycles, meta = self.loaded()
        segment = select_segment(cycles, meta, cycle_id=1, duration_s=300)
        np.testing.assert_array_equal(segment["values"], cycles[0].values[:2])
        np.testing.assert_array_equal(segment["intervals"], [[0, 1, 1]])
        self.assertEqual(segment["initial_soc"], .8)
        self.assertEqual(segment["temperature_c"], 20)
        exact = select_segment(cycles, meta, cycle_id=1, duration_s=301)
        self.assertEqual(len(exact["intervals"]), 2)
        self.assertEqual(exact["intervals"][-1, 1], 301)
        second = select_segment(cycles, meta, cycle_id=2, duration_s=2)
        self.assertEqual(second["initial_soc"], .735)
        self.assertEqual(second["temperature_c"], 22)
        np.testing.assert_array_equal(second["intervals"], [[0, .125, 2], [.125, 2, 0]])

    def test_invalid_segment_selection_is_rejected(self):
        cycles, meta = self.loaded()
        for cycle_id, duration in ((99, 1), (-1, 1), (True, 1), (1, 0),
                                   (1, .5), (1, 302), (1, 3601),
                                   (1, float("nan")), (1, float("inf"))):
            with self.subTest(cycle_id=cycle_id, duration=duration), self.assertRaises(ValueError):
                select_segment(cycles, meta, cycle_id=cycle_id, duration_s=duration)

    def test_missing_or_invalid_bms_soc_is_not_replaced_with_model_default(self):
        cycles, meta = self.loaded()
        for value in (None, True, "80", -1, 0, 100, float("nan")):
            changed = copy.deepcopy(meta)
            changed["completed_cycle_metrics"][0]["initial_soc_bms_pct"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_segment(cycles, changed, cycle_id=1, duration_s=1)

    def test_selected_current_and_sensor_temperature_bounds_are_enforced(self):
        for column, value in ((4, 161), (4, -161), (5, 46), (5, -1)):
            rows = copy.deepcopy(self.rows)
            rows[1][column] = value
            self.write_inputs(rows=rows)
            cycles, meta = self.loaded()
            with self.subTest(column=column, value=value), self.assertRaises(ValueError):
                select_segment(cycles, meta, cycle_id=1, duration_s=1)

    def test_success_report_keeps_source_window_assumptions_and_actual_duration(self):
        with mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=self.simulated):
            report = self.run_replay(duration_s=300)
        self.assertEqual(report["status"], "complete")
        self.assertEqual(report["requested_duration_s"], 300)
        self.assertEqual(report["selected_duration_s"], 1)
        self.assertEqual(report["source_start_timestamp_s"], 100)
        self.assertEqual(report["source_end_timestamp_s"], 101)
        self.assertEqual(report["source_rows"], 2)
        self.assertEqual(report["trace"][0]["source_timestamp_s"], 101)
        self.assertAlmostEqual(report["trace"][0]["observed_cell_voltage_v"], 420 / 108)
        self.assertEqual(report["initialization"]["model_soc"], .8)
        self.assertFalse(report["physical_model_compatibility_verified"])
        self.assertFalse(report["validation"]["voltage_accuracy_accepted"])
        self.assertIsNone(report["remaining_life_days"])
        self.assertIsNone(report["soh_percent"])
        saved = json.loads((self.root / "replay/result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved, report)
        before = (self.root / "replay/result.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.run_replay(duration_s=300)
        self.assertEqual((self.root / "replay/result.json").read_bytes(), before)

    def test_simulation_failure_is_persisted_and_propagated(self):
        with mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=ValueError("cutoff reached")):
            with self.assertRaisesRegex(ValueError, "cutoff reached"):
                self.run_replay(duration_s=1)
        saved = json.loads((self.root / "replay/result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertEqual(saved["error_type"], "ValueError")
        self.assertIn("cutoff reached", saved["error"])
        self.assertNotIn("validation", saved)
        self.assertNotIn("summary", saved)

    def test_incomplete_or_charge_inconsistent_simulation_cannot_report_complete(self):
        def corrupt(segment, config):
            trace, numerics = self.simulated(segment, config)
            trace[-1]["model_discharge_capacity_end_ah"] += .001
            return trace, numerics
        for label, returned in (("incomplete", lambda segment, config: ([], {})), ("bad_charge", corrupt)):
            output = self.root / label
            with self.subTest(label=label), mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=returned):
                with self.assertRaises(ValueError):
                    self.run_replay(output, duration_s=1)
            saved = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "failed")

    def test_cli_replay_routes_arguments_and_failure_exit_code(self):
        from .__main__ import main
        argv = ["replay-input", "--csv", str(self.csv_path), "--metadata", str(self.meta_path),
                "--pack-config", str(self.config_path), "--output", str(self.root / "cli"),
                "--cycle-id", "2", "--duration-s", "2"]
        with mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=self.simulated), contextlib.redirect_stdout(io.StringIO()):
            main(argv)
        saved = json.loads((self.root / "cli/result.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["cycle_id"], 2)
        self.assertEqual(saved["selected_duration_s"], 2)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(argv)
        self.assertEqual(error.exception.code, 2)

    def test_calibration_cli_preserves_bms_soc_and_records_material_source(self):
        from .__main__ import main
        argv = ["replay-input", "--csv", str(self.csv_path), "--metadata", str(self.meta_path),
                "--pack-config", str(self.config_path), "--output", str(self.root / "calibrated"),
                "--cell-calibration", "m50-c-joint-v3"]
        with mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=self.simulated) as simulate:
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
        segment = simulate.call_args.args[0]
        report = json.loads((self.root / "calibrated/result.json").read_text(encoding="utf-8"))
        source = report["model"]["cell_calibration"]
        self.assertEqual(source["id"], "m50-c-joint-v3")
        self.assertIn("cell_calibration", report["source_files"])
        self.assertEqual(segment["initial_soc"], .8)
        self.assertEqual(segment["material_overrides"], source["material_overrides"])
        self.assertEqual(segment["material_overrides"]["Initial concentration in negative electrode [mol.m-3]"],
                         30671.0499722885)
        self.assertEqual(segment["initial_particle_lithium_Ah"], 7.094611861277718)
        self.assertFalse(report["physical_model_compatibility_verified"])

    def test_lumped_replay_reports_ambient_initial_and_observed_temperatures_separately(self):
        from .__main__ import main

        def simulated_lumped(segment, model_config):
            self.assertEqual(segment["ambient_temperature_c"], 18)
            self.assertEqual(segment["temperature_c"], 20)
            rows, numerics = self.simulated(segment, model_config)
            for row in rows:
                row.update(model_cell_temperature_start_c=20.0, model_cell_temperature_end_c=19.99)
            return rows, numerics

        argv = ["replay-input", "--csv", str(self.csv_path), "--metadata", str(self.meta_path),
                "--pack-config", str(self.config_path), "--output", str(self.root / "lumped"),
                "--thermal-mode", "lumped", "--ambient-temperature-c", "18"]
        with mock.patch("bayesian.pack_replay._simulate_intervals", side_effect=simulated_lumped):
            with contextlib.redirect_stdout(io.StringIO()):
                main(argv)
        report = json.loads((self.root / "lumped/result.json").read_text(encoding="utf-8"))
        self.assertEqual(report["model"]["options"]["thermal"], "lumped")
        self.assertEqual(report["temperature"]["ambient_temperature_c"], 18)
        self.assertEqual(report["temperature"]["initial_cell_temperature_c"], 20)
        self.assertEqual(report["temperature"]["modeled_cell_end_c"], 19.99)
        self.assertEqual(report["trace"][0]["observed_module_temperature_c"], 21)

    def test_lumped_replay_requires_valid_ambient_boundary_before_creating_output(self):
        for ambient in (None, True, float("nan"), -273.15):
            with self.subTest(ambient=ambient), self.assertRaisesRegex(ValueError, "ambient"):
                self.run_replay(thermal_mode="lumped", ambient_temperature_c=ambient)
        with self.assertRaisesRegex(ValueError, "Ambient"):
            self.run_replay(ambient_temperature_c=20)
        self.assertFalse((self.root / "replay").exists())


if __name__ == "__main__":
    unittest.main()
