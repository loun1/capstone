"""Read pack_raw_six_v1 telemetry without inventing GP trip context.

Simulation seconds and per-cycle endpoints are preserved. Cell conversion is
an explicitly configured equivalent-cell mapping, not a model calibration.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np


RAW_COLUMNS = ("vehicle_id", "cycle_id", "timestamp_s", "voltage_v", "current_a", "temperature_c")
CELL_COLUMNS = (*RAW_COLUMNS, "elapsed_s", "cell_voltage_v", "cell_current_a")
SOURCE_TIME_BASIS = "Preserve battery simulation seconds, including saved history; not UTC or minutes."


def _json(path):
    with Path(path).open(encoding="utf-8-sig") as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _integer(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class PackConfig:
    series_cells: int
    parallel_cells: int
    cell_basis_id: str
    source: str

    def __post_init__(self):
        _integer(self.series_cells, "series_cells")
        _integer(self.parallel_cells, "parallel_cells")
        for key in ("cell_basis_id", "source"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key).strip():
                raise ValueError(f"Explicit {key} required")

    @classmethod
    def read(cls, path):
        value = _json(path)
        expected = {"schema": "pack_to_cell_config_v1",
                    "current_sharing": "equal_parallel_current_assumption",
                    "physical_model_compatibility_verified": False}
        for key, required in expected.items():
            if key not in value or type(value[key]) is not type(required) or value[key] != required:
                raise ValueError(f"Pack config requires {key}={required!r}")
        try:
            return cls(**{key: value[key] for key in
                          ("series_cells", "parallel_cells", "cell_basis_id", "source")})
        except KeyError as error:
            raise ValueError(f"Missing pack configuration: {error}") from error


@dataclass(frozen=True)
class PackCycle:
    vehicle_id: str
    cycle_id: int
    # Source columns: simulation time, pack voltage, pack current, sensor temperature.
    values: np.ndarray
    config: PackConfig

    @property
    def elapsed_s(self):
        return self.values[:, 0] - self.values[0, 0]

    @property
    def cell_voltage_v(self):
        return self.values[:, 1] / self.config.series_cells

    @property
    def cell_current_a(self):
        return self.values[:, 2] / self.config.parallel_cells

    def current_intervals(self):
        """[start_s, end_s, cell_current_a] with right-endpoint hold.

        This explicitly declares the reconstruction that matches the supplied
        export's Ah counters. It does not imply measured intersample dynamics.
        No interval is created across cycle boundaries.
        """
        return np.column_stack((self.elapsed_s[:-1], self.elapsed_s[1:], self.cell_current_a[1:]))


def _validate_metadata(meta):
    export = meta.get("public_export", {})
    if not isinstance(export, dict):
        raise ValueError("metadata.public_export must be an object")
    expected = {
        "schema": "pack_raw_six_v1", "revision": 2, "basis": "whole_pack",
        "columns": list(RAW_COLUMNS), "timestamp_unit": "s", "voltage_unit": "V",
        "current_unit": "A", "temperature_unit": "degC",
        "current_sign": "positive_discharge_negative_charge_or_regeneration",
        "temperature": "fixed_module_sensor",
        "time_basis": SOURCE_TIME_BASIS,
    }
    for key, required in expected.items():
        if export.get(key) != required:
            raise ValueError(f"metadata.public_export requires {key}={required!r}")
    channel = export.get("temperature_channel_id")
    if not isinstance(channel, str) or not channel.strip():
        raise ValueError("Metadata must identify temperature channel and time basis")
    if meta.get("columns") != len(RAW_COLUMNS):
        raise ValueError("Metadata column count mismatch")
    _integer(meta.get("observation_rows"), "observation_rows")
    protocol = meta.get("cycle_protocol", {})
    if not isinstance(protocol, dict):
        raise ValueError("metadata.cycle_protocol must be an object")
    if meta.get("status") != "completed" or protocol.get("cycle_is_not_efc") is not True:
        raise ValueError("This importer requires a completed export with cycle_is_not_efc=true")
    ids = protocol.get("cycle_ids")
    if not isinstance(ids, list) or not ids:
        raise ValueError("Explicit completed cycle_ids required")
    for value in ids:
        _integer(value, "cycle_id", minimum=0)
    if len(set(ids)) != len(ids) or protocol.get("completed_cycles") != len(ids):
        raise ValueError("Completed cycle count/IDs mismatch")
    metrics = meta.get("completed_cycle_metrics")
    if not isinstance(metrics, list) or len(metrics) != len(ids):
        raise ValueError("One completed_cycle_metrics entry per cycle required")
    if not all(isinstance(metric, dict) for metric in metrics):
        raise ValueError("Every completed_cycle_metrics entry must be an object")
    return ids, metrics


def load_pack_input(csv_path, metadata_path, config: PackConfig):
    """Validate complete raw export and retain source order and every sample."""
    meta = _json(metadata_path)
    ids, metrics = _validate_metadata(meta)
    groups = {}
    previous_key = None
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if tuple(reader.fieldnames or ()) != RAW_COLUMNS:
            raise ValueError(f"Raw CSV requires exactly these ordered columns: {RAW_COLUMNS}")
        for row in reader:
            line = reader.line_num
            if None in row or any(row[key] is None or not row[key].strip() for key in RAW_COLUMNS):
                raise ValueError(f"Incomplete/extra CSV fields at line {line}")
            try:
                cycle = int(row["cycle_id"])
                if str(cycle) != row["cycle_id"] or cycle < 0:
                    raise ValueError("noncanonical cycle ID")
                values = [float(row[key]) for key in RAW_COLUMNS[2:]]
            except ValueError as error:
                raise ValueError(f"Invalid numeric sample at CSV line {line}") from error
            if not np.isfinite(values).all() or values[0] < 0 or values[1] <= 0 or values[3] < -273.15:
                raise ValueError(f"Nonfinite or invalid physical sample at CSV line {line}")
            key = (row["vehicle_id"], cycle)
            if key != previous_key and key in groups:
                raise ValueError(f"Noncontiguous cycle at CSV line {line}")
            groups.setdefault(key, []).append(values)
            previous_key = key
    if len({key[0] for key in groups}) != 1:
        raise ValueError("One vehicle per metadata export required")
    if [key[1] for key in groups] != ids:
        raise ValueError("CSV cycles/order do not match completed metadata cycle_ids")
    if sum(map(len, groups.values())) != meta["observation_rows"]:
        raise ValueError("CSV observation_rows mismatch")
    cycles = []
    for ((vehicle, cycle_id), rows), metric in zip(groups.items(), metrics):
        data = np.asarray(rows, dtype=float)
        if len(data) < 2 or not np.all(np.diff(data[:, 0]) > 0):
            raise ValueError(f"Cycle {cycle_id} timestamps must be strictly increasing")
        if cycles and data[0, 0] < cycles[-1].values[-1, 0]:
            raise ValueError("Cycle times overlap; boundary equality alone is allowed")
        expected = {"observation_rows": len(data), "start_timestamp_s": data[0, 0],
                    "end_timestamp_s": data[-1, 0], "duration_s": data[-1, 0] - data[0, 0]}
        for key, actual in expected.items():
            recorded = metric.get(key)
            if (isinstance(recorded, bool) or not isinstance(recorded, (int, float))
                    or not np.isfinite(recorded) or abs(recorded - actual) > 1e-8):
                raise ValueError(f"Cycle {cycle_id} metadata {key} mismatch")
        data.setflags(write=False)
        cycles.append(PackCycle(vehicle, cycle_id, data, config))
    return cycles, meta


def _summary(cycle, metric):
    values = cycle.values
    dt = np.diff(values[:, 0])
    current = values[1:, 2]
    charge = float(np.sum(np.maximum(-current, 0) * dt) / 3600)
    discharge = float(np.sum(np.maximum(current, 0) * dt) / 3600)
    result = {
        "vehicle_id": cycle.vehicle_id, "cycle_id": cycle.cycle_id, "rows": len(values),
        "start_timestamp_s": float(values[0, 0]), "end_timestamp_s": float(values[-1, 0]),
        "sample_interval_s": [float(dt.min()), float(dt.max())],
        "pack_voltage_v": [float(values[:, 1].min()), float(values[:, 1].max())],
        "pack_current_a": [float(values[:, 2].min()), float(values[:, 2].max())],
        "cell_voltage_v": [float(cycle.cell_voltage_v.min()), float(cycle.cell_voltage_v.max())],
        "cell_current_a": [float(cycle.cell_current_a.min()), float(cycle.cell_current_a.max())],
        "module_temperature_c": [float(values[:, 3].min()), float(values[:, 3].max())],
        "charge_or_regeneration_ah": charge, "discharge_ah": discharge,
        "integration": "right_endpoint_hold_on_each_source_interval",
        "cycle_is_not_efc": True,
    }
    # Compare, never silently replace producer counters or infer charge phases.
    for key, actual in (("charge_ah", charge), ("discharge_ah", discharge)):
        recorded = metric.get(key)
        if isinstance(recorded, (int, float)) and not isinstance(recorded, bool) and np.isfinite(recorded):
            result[f"metadata_{key}"] = recorded
            result[f"metadata_{key}_residual"] = actual - recorded
    return result


def prepare_input(csv_path, metadata_path, config_path, output):
    """Write an exclusive new normalized export and its auditable input report."""
    config = PackConfig.read(config_path)
    paths = {"raw_csv": Path(csv_path), "metadata": Path(metadata_path), "pack_config": Path(config_path)}
    hashes = {key: _sha(path) for key, path in paths.items()}
    cycles, meta = load_pack_input(csv_path, metadata_path, config)
    if hashes != {key: _sha(path) for key, path in paths.items()}:
        raise ValueError("Input changed while it was being read")
    report = {
        "schema": "normalized_pack_input_report_v1", "status": "input_conversion_verified",
        "source_files": {key: {"path": str(path.resolve()), "sha256": hashes[key]} for key, path in paths.items()},
        "source_schema": meta["public_export"]["schema"], "rows": sum(len(c.values) for c in cycles),
        "series_cells": config.series_cells, "parallel_cells": config.parallel_cells,
        "cell_basis_id": config.cell_basis_id, "topology_source": config.source,
        "voltage_mapping": "pack_voltage_v / series_cells",
        "current_mapping": "pack_current_a / parallel_cells; sign unchanged",
        "current_sharing": "equal_parallel_current_assumption",
        "time_basis": "original simulation seconds; elapsed_s resets within each cycle; no UTC inferred",
        "temperature_channel_id": meta["public_export"]["temperature_channel_id"],
        "temperature_interpretation": "module sensor preserved; not a cell-core or ambient measurement",
        "boundary_policy": "preserve both samples across cycles; never integrate across cycle boundary",
        "boundary_duplicate_times": sum(a.values[-1, 0] == b.values[0, 0] for a, b in zip(cycles, cycles[1:])),
        "producer_preprocessor_execution_verified": meta.get("actual_preprocessor_execution_verified"),
        "physical_model_compatibility_verified": False, "gp_trip_context_ready": False,
        "missing_gp_context": ["driver_id", "trip segmentation", "UTC start/clock mapping", "route_id", "direction", "traffic", "matched reference vehicle history"],
        "cycles": [_summary(c, m) for c, m in zip(cycles, meta["completed_cycle_metrics"])],
    }
    report["boundary_duplicate_times"] = int(report["boundary_duplicate_times"])
    output = Path(output)
    # No overwrite of raw sources or earlier conversion runs.
    output.mkdir(parents=True, exist_ok=False)
    normalized = output / "normalized_cell_input.csv"
    with normalized.open("x", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(CELL_COLUMNS)
        for cycle in cycles:
            for row, elapsed, voltage, current in zip(cycle.values, cycle.elapsed_s, cycle.cell_voltage_v, cycle.cell_current_a):
                writer.writerow((cycle.vehicle_id, cycle.cycle_id, *row, elapsed, voltage, current))
    report["normalized_csv_sha256"] = _sha(normalized)
    with (output / "input_report.json").open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, allow_nan=False, indent=2)
        target.write("\n")
    return report
