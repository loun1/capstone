"""Independent input-contract checks using small known physical traces."""
import contextlib
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .pack_input import CELL_COLUMNS, RAW_COLUMNS, PackConfig, load_pack_input, prepare_input


class PackInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.csv_path = self.root / "raw.csv"
        self.meta_path = self.root / "metadata.json"
        self.config_path = self.root / "pack.json"
        self.config = PackConfig(108, 32, "m50t-equivalent-cell", "test explicit pack design")
        # Duplicated boundary time belongs to distinct complete cycles.
        # Nonuniform intervals deliberately exceed the GP trip gap limit.
        self.rows = [
            ["taxi", 1, 100, 432, 0, 20],
            ["taxi", 1, 101, 420, 32, 21],
            ["taxi", 1, 401, 410, -64, 22],
            ["taxi", 2, 401, 410, -64, 22],
            ["taxi", 2, 401.125, 416, 64, 23],
            ["taxi", 2, 403, 420, 0, 21],
        ]
        self.meta = {
            "columns": 6, "observation_rows": 6, "status": "completed",
            "actual_preprocessor_execution_verified": False,
            "public_export": {
                "schema": "pack_raw_six_v1", "revision": 2, "basis": "whole_pack",
                "columns": list(RAW_COLUMNS), "timestamp_unit": "s", "voltage_unit": "V",
                "current_unit": "A", "temperature_unit": "degC",
                "current_sign": "positive_discharge_negative_charge_or_regeneration",
                "temperature": "fixed_module_sensor",
                "temperature_channel_id": "module_01_temperature_c",
                "time_basis": "Preserve battery simulation seconds, including saved history; not UTC or minutes.",
            },
            "cycle_protocol": {
                "completed_cycles": 2, "cycle_ids": [1, 2], "cycle_is_not_efc": True,
                "preparation_counted_as_cycle": False,
            },
            "completed_cycle_metrics": [
                {"observation_rows": 3, "start_timestamp_s": 100, "end_timestamp_s": 401,
                 "duration_s": 301, "charge_ah": 64 * 300 / 3600,
                 "discharge_ah": 32 / 3600},
                {"observation_rows": 3, "start_timestamp_s": 401, "end_timestamp_s": 403,
                 "duration_s": 2, "charge_ah": 0, "discharge_ah": 64 * .125 / 3600},
            ],
        }
        self.pack = {
            "schema": "pack_to_cell_config_v1", "series_cells": 108,
            "parallel_cells": 32, "cell_basis_id": "m50t-equivalent-cell",
            "source": "test explicit pack design",
            "current_sharing": "equal_parallel_current_assumption",
            "physical_model_compatibility_verified": False,
        }
        self.write_inputs()

    def write_inputs(self, rows=None, meta=None, pack=None, header=RAW_COLUMNS):
        with self.csv_path.open("w", newline="", encoding="utf-8") as out:
            writer = csv.writer(out)
            writer.writerow(header)
            writer.writerows(self.rows if rows is None else rows)
        self.meta_path.write_text(json.dumps(self.meta if meta is None else meta), encoding="utf-8")
        self.config_path.write_text(json.dumps(self.pack if pack is None else pack), encoding="utf-8")

    def load(self):
        return load_pack_input(self.csv_path, self.meta_path, self.config)

    def test_nonuniform_intervals_and_cycle_boundary_preserved(self):
        cycles, metadata = self.load()
        self.assertEqual(len(cycles), 2)
        self.assertEqual(metadata, self.meta)
        self.assertEqual(sum(len(c.values) for c in cycles), 6)
        np.testing.assert_array_equal(cycles[0].elapsed_s, [0, 1, 301])
        np.testing.assert_array_equal(cycles[1].elapsed_s, [0, .125, 2])
        np.testing.assert_array_equal(cycles[0].current_intervals(), [[0, 1, 1], [1, 301, -2]])
        np.testing.assert_array_equal(cycles[1].current_intervals(), [[0, .125, 2], [.125, 2, 0]])
        np.testing.assert_array_equal(cycles[0].values[-1], cycles[1].values[0])
        self.assertFalse(cycles[0].values.flags.writeable)

    def test_conversion_and_no_fabricated_trip_context(self):
        output = self.root / "converted"
        before = {p.name: p.read_bytes() for p in (self.csv_path, self.meta_path, self.config_path)}
        report = prepare_input(self.csv_path, self.meta_path, self.config_path, output)
        self.assertEqual(report["rows"], 6)
        self.assertEqual(report["boundary_duplicate_times"], 1)
        self.assertFalse(report["gp_trip_context_ready"])
        self.assertFalse(report["physical_model_compatibility_verified"])
        self.assertFalse(report["producer_preprocessor_execution_verified"])
        self.assertIn("driver_id", report["missing_gp_context"])
        self.assertEqual(report["temperature_channel_id"], "module_01_temperature_c")
        self.assertAlmostEqual(report["cycles"][0]["charge_or_regeneration_ah"], 64 * 300 / 3600)
        self.assertAlmostEqual(report["cycles"][0]["discharge_ah"], 32 / 3600)
        self.assertAlmostEqual(report["cycles"][0]["metadata_charge_ah_residual"], 0)
        with (output / "normalized_cell_input.csv").open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            self.assertEqual(tuple(reader.fieldnames), CELL_COLUMNS)
            normalized = list(reader)
        self.assertEqual(len(normalized), 6)
        for raw, converted in zip(self.rows, normalized):
            self.assertEqual(converted["vehicle_id"], raw[0])
            self.assertEqual(int(converted["cycle_id"]), raw[1])
            for index, key in enumerate(RAW_COLUMNS[2:], start=2):
                self.assertEqual(float(converted[key]), raw[index])
            self.assertEqual(float(converted["cell_voltage_v"]), raw[3] / 108)
            self.assertEqual(float(converted["cell_current_a"]), raw[4] / 32)
        for p in (self.csv_path, self.meta_path, self.config_path):
            self.assertEqual(p.read_bytes(), before[p.name])
        self.assertEqual(report["normalized_csv_sha256"], hashlib.sha256((output / "normalized_cell_input.csv").read_bytes()).hexdigest())
        self.assertEqual(json.loads((output / "input_report.json").read_text(encoding="utf-8")), report)
        saved = (output / "normalized_cell_input.csv").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_input(self.csv_path, self.meta_path, self.config_path, output)
        self.assertEqual((output / "normalized_cell_input.csv").read_bytes(), saved)

    def test_rejects_units_sign_and_basis_mismatch(self):
        for key, bad in (("timestamp_unit", "min"), ("current_sign", "positive_charge"),
                         ("basis", "single_cell"), ("voltage_unit", "mV"),
                         ("current_unit", "mA"), ("temperature_unit", "K"),
                         ("temperature", "ambient"), ("schema", "unknown"),
                         ("time_basis", "UTC timestamps in minutes")):
            with self.subTest(key=key):
                meta = copy.deepcopy(self.meta)
                meta["public_export"][key] = bad
                self.write_inputs(meta=meta)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_nonfinite_or_invalid_numeric_samples(self):
        for column, bad in ((2, "nan"), (3, "inf"), (4, "-inf"), (5, "nan"),
                            (2, -1), (3, 0), (5, -274), (1, "1.0")):
            with self.subTest(column=column, bad=bad):
                rows = copy.deepcopy(self.rows)
                rows[1][column] = bad
                self.write_inputs(rows=rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_bad_time_order_or_overlap(self):
        for index, timestamp in ((1, 100), (1, 99), (3, 400)):
            with self.subTest(index=index, timestamp=timestamp):
                rows = copy.deepcopy(self.rows)
                rows[index][2] = timestamp
                self.write_inputs(rows=rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_counts_and_partial_exports(self):
        changes = [
            ("observation_rows", 7), ("status", "partial"), ("columns", 5),
            ("cycle_protocol", {"cycle_ids": [1, 2], "completed_cycles": 1, "cycle_is_not_efc": True}),
            ("cycle_protocol", {"cycle_ids": [1, 1], "completed_cycles": 2, "cycle_is_not_efc": True}),
            ("cycle_protocol", {"cycle_ids": [1, 2], "completed_cycles": 2, "cycle_is_not_efc": False}),
            ("completed_cycle_metrics", self.meta["completed_cycle_metrics"][:1]),
        ]
        for key, bad in changes:
            with self.subTest(key=key, bad=bad):
                meta = copy.deepcopy(self.meta)
                meta[key] = bad
                self.write_inputs(meta=meta)
                with self.assertRaises(ValueError):
                    self.load()
        for key, bad in (("observation_rows", 4), ("start_timestamp_s", 99),
                         ("end_timestamp_s", 400), ("duration_s", 300)):
            with self.subTest(metric=key):
                meta = copy.deepcopy(self.meta)
                meta["completed_cycle_metrics"][0][key] = bad
                self.write_inputs(meta=meta)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_malformed_metadata_shapes_with_value_error(self):
        for key, bad in (("public_export", []), ("cycle_protocol", None),
                         ("completed_cycle_metrics", ["invalid", {}])):
            with self.subTest(key=key):
                meta = copy.deepcopy(self.meta)
                meta[key] = bad
                self.write_inputs(meta=meta)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_missing_extra_or_wrong_order_csv_columns(self):
        for header in (RAW_COLUMNS[:-1], (*RAW_COLUMNS, "fake"), tuple(reversed(RAW_COLUMNS))):
            with self.subTest(header=header):
                self.write_inputs(header=header)
                with self.assertRaises(ValueError):
                    self.load()
        for badrow in (self.rows[1][:-1], [*self.rows[1], "extra"], ["", *self.rows[1][1:]]):
            with self.subTest(badrow=badrow):
                rows = copy.deepcopy(self.rows)
                rows[1] = badrow
                self.write_inputs(rows=rows)
                with self.assertRaises(ValueError):
                    self.load()

    def test_rejects_reordered_cycles_or_multiple_vehicles(self):
        for rows in (self.rows[3:] + self.rows[:3], [self.rows[0], self.rows[3], *self.rows[1:3], *self.rows[4:]]):
            with self.subTest(rows=rows):
                self.write_inputs(rows=rows)
                with self.assertRaises(ValueError):
                    self.load()
        rows = copy.deepcopy(self.rows)
        for row in rows[3:]:
            row[0] = "other-taxi"
        self.write_inputs(rows=rows)
        with self.assertRaises(ValueError):
            self.load()

    def test_topology_is_explicit_positive_integer_with_provenance(self):
        for key, bad in (("series_cells", 0), ("parallel_cells", -2),
                         ("parallel_cells", 32.0), ("series_cells", True),
                         ("source", " "), ("cell_basis_id", ""),
                         ("current_sharing", "unknown"),
                         ("physical_model_compatibility_verified", True),
                         ("physical_model_compatibility_verified", 0)):
            with self.subTest(key=key, bad=bad):
                pack = copy.deepcopy(self.pack)
                pack[key] = bad
                self.write_inputs(pack=pack)
                with self.assertRaises(ValueError):
                    PackConfig.read(self.config_path)
        pack = copy.deepcopy(self.pack)
        del pack["parallel_cells"]
        self.write_inputs(pack=pack)
        with self.assertRaises(ValueError):
            PackConfig.read(self.config_path)

    def test_invalid_input_never_creates_output(self):
        meta = copy.deepcopy(self.meta)
        meta["observation_rows"] = 99
        self.write_inputs(meta=meta)
        output = self.root / "must-not-exist"
        with self.assertRaises(ValueError):
            prepare_input(self.csv_path, self.meta_path, self.config_path, output)
        self.assertFalse(output.exists())

    def test_cli_runs_preparation_without_running_forecast(self):
        from .__main__ import main
        output = self.root / "cli"
        with contextlib.redirect_stdout(io.StringIO()):
            main(["prepare-input", "--csv", str(self.csv_path), "--metadata", str(self.meta_path),
                  "--pack-config", str(self.config_path), "--output", str(output)])
        report = json.loads((output / "input_report.json").read_text(encoding="utf-8"))
        self.assertFalse(report["gp_trip_context_ready"])
        self.assertNotIn("forecast", report)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["prepare-input", "--csv", str(self.csv_path), "--metadata", str(self.meta_path),
                  "--pack-config", str(self.config_path), "--output", str(output)])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
