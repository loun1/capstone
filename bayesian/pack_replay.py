"""Replay an explicitly bounded pack segment through SPMe with optional thermal dynamics."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from .pack_input import PackConfig, _json, _sha, load_pack_input

ROOT = Path(__file__).resolve().parents[1]
CHARGE_TOLERANCE_AH = 1e-7


def select_segment(cycles, metadata, *, cycle_id, duration_s):
    if type(cycle_id) is not int or cycle_id < 0:
        raise ValueError("cycle_id must be a nonnegative integer")
    if not np.isfinite(duration_s) or not 0 < duration_s <= 3600:
        raise ValueError("Replay duration must be within (0, 3600] seconds")
    found = [c for c in cycles if c.cycle_id == cycle_id]
    if len(found) != 1:
        raise ValueError("Select exactly one existing cycle")
    cycle = found[0]
    if duration_s > cycle.elapsed_s[-1]:
        raise ValueError("Requested duration exceeds recorded cycle")
    end = int(np.searchsorted(cycle.elapsed_s, duration_s, side="right") - 1)
    if end < 1:
        raise ValueError("Requested duration contains no complete source interval")
    if end > 10000:
        raise ValueError("Short replay is limited to 10000 source intervals")
    metric = metadata["completed_cycle_metrics"][metadata["cycle_protocol"]["cycle_ids"].index(cycle_id)]
    soc = metric.get("initial_soc_bms_pct")
    if isinstance(soc, bool) or not isinstance(soc, (int, float)) or not np.isfinite(soc) or not 0 < soc < 100:
        raise ValueError("Finite initial_soc_bms_pct within (0, 100) is required")
    values = cycle.values[:end + 1]
    if np.max(abs(cycle.cell_current_a[:end + 1])) > 5:
        raise ValueError("Selected segment exceeds exploratory 5 A cell-current limit")
    if not np.all((values[:, 3] >= 0) & (values[:, 3] <= 45)):
        raise ValueError("Selected module sensor temperatures must be within 0 to 45 C")
    return {
        "cycle": cycle, "values": values,
        "intervals": cycle.current_intervals()[:end],
        "initial_soc": soc / 100,
        "temperature_c": float(values[0, 3]),
    }


def _simulate_intervals(segment, model_config):
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(key, "1")
    os.environ.setdefault("PYBAMM_DISABLE_TELEMETRY", "true")
    import pybamm

    if model_config.get("parameter_set") != "OKane2022" or model_config["options"].get("thermal") != "isothermal":
        raise ValueError("Replay uses the existing OKane2022 isothermal SPMe model")
    params = pybamm.ParameterValues(model_config["parameter_set"])
    params.update(segment.get("material_overrides", {}))
    temperature_k = segment["temperature_c"] + 273.15
    options = model_config["options"].copy()
    ambient_c = segment.get("ambient_temperature_c")
    if ambient_c is not None:
        options["thermal"] = "lumped"
    params.update({"Current function [A]": pybamm.InputParameter("cell_current_a"),
                   "Initial temperature [K]": temperature_k,
                   "Ambient temperature [K]": temperature_k if ambient_c is None else ambient_c + 273.15})
    limits = [float(params["Lower voltage cut-off [V]"]), float(params["Upper voltage cut-off [V]"])]
    if limits != [model_config["voltage_min_V"], model_config["voltage_max_V"]]:
        raise ValueError("Baseline voltage limits disagree with model configuration")
    model = pybamm.lithium_ion.SPMe(options=options)
    mesh = dict(model.default_var_pts)
    sim = pybamm.Simulation(model, parameter_values=params, var_pts=mesh,
                           solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10))
    sim.build(initial_soc=segment["initial_soc"], inputs={"cell_current_a": float(segment["intervals"][0, 2])})
    differential_count = sim.built_model.len_rhs
    trace = []
    previous_state = None
    previous_capacity = None
    maximum_state_jump = 0.0
    calibration_inventory = segment.get("initial_particle_lithium_Ah")
    extra_numerics = {}
    for index, (start, end, current) in enumerate(segment["intervals"], start=1):
        solution = sim.step(float(end - start), inputs={"cell_current_a": float(current)},
                            save=False, t_eval=np.array([0.0, end - start]))
        times = np.asarray(solution["Time [s]"].entries, dtype=float)
        volts = np.asarray(solution["Voltage [V]"].entries, dtype=float)
        amps = np.asarray(solution["Current [A]"].entries, dtype=float)
        capacities = np.asarray(solution["Discharge capacity [A.h]"].entries, dtype=float)
        state = np.asarray(solution.all_ys[0], dtype=float)
        if (solution.termination != "final time" or len(times) < 2
                or abs(times[0] - start) > 1e-6 or abs(times[-1] - end) > 1e-6):
            raise ValueError(f"Interval {index} stopped early: {solution.termination}")
        if not all(np.isfinite(a).all() for a in (times, volts, amps, capacities, state)):
            raise ValueError(f"Nonfinite solution at interval {index}")
        if np.max(abs(amps - current)) > 1e-9:
            raise ValueError(f"Imposed current mismatch at interval {index}")
        if index == 1 and calibration_inventory is not None:
            lithium_mol = float(solution["Total lithium in particles [mol]"].entries[0])
            actual_ah = lithium_mol * pybamm.constants.F.value / 3600
            if not np.isfinite(actual_ah) or abs(actual_ah - calibration_inventory) > CHARGE_TOLERANCE_AH:
                raise ValueError("Calibrated lithium inventory changed during BMS SOC initialization")
            extra_numerics = {"initial_particle_lithium_Ah": actual_ah,
                               "calibrated_inventory_residual_Ah": actual_ah - calibration_inventory}
        if previous_state is not None:
            # Algebraic state may jump with load; differential state must persist.
            jump = float(np.max(abs(state[:differential_count, 0] - previous_state)))
            maximum_state_jump = max(maximum_state_jump, jump)
            if jump > 1e-8:
                raise ValueError(f"Differential state was reset at interval {index}")
            if abs(capacities[0] - previous_capacity) > CHARGE_TOLERANCE_AH:
                raise ValueError(f"Capacity state was reset at interval {index}")
        previous_state = state[:differential_count, -1].copy()
        previous_capacity = float(capacities[-1])
        expected_ah = float(current * (end - start) / 3600)
        residual = float(capacities[-1] - capacities[0] - expected_ah)
        if abs(residual) > CHARGE_TOLERANCE_AH:
            raise ValueError(f"Charge balance failed at interval {index}")
        trace.append({"elapsed_start_s": float(start), "elapsed_end_s": float(end),
                      "cell_current_a": float(current), "model_current_end_a": float(amps[-1]),
                      "model_voltage_start_v": float(volts[0]), "model_voltage_end_v": float(volts[-1]),
                      "model_discharge_capacity_start_ah": float(capacities[0]),
                      "model_discharge_capacity_end_ah": float(capacities[-1]),
                      "charge_balance_residual_ah": residual})
        if ambient_c is not None:
            temperature = np.asarray(solution["Volume-averaged cell temperature [K]"].entries, dtype=float)
            if not np.all(np.isfinite(temperature) & (temperature > 0)):
                raise ValueError(f"Invalid modeled cell temperature at interval {index}")
            trace[-1].update(model_cell_temperature_start_c=float(temperature[0] - 273.15),
                             model_cell_temperature_end_c=float(temperature[-1] - 273.15))
    if ambient_c is not None:
        extra_numerics["thermal_parameters"] = {
            name: float(params[name]) for name in ("Cell volume [m3]", "Cell cooling surface area [m2]",
                                                 "Total heat transfer coefficient [W.m-2.K-1]")
        }
    return trace, {"pybamm_version": pybamm.__version__, "python_version": platform.python_version(),
                   "solver": "IDAKLU", "rtol": 1e-8, "atol": 1e-10,
                   "mesh_points": {str(k): int(v) for k, v in mesh.items()},
                   "voltage_limits_v": limits, "maximum_differential_state_jump": maximum_state_jump,
                   **extra_numerics}


def run_replay(csv_path, metadata_path, config_path, output, *, cycle_id=1, duration_s=300.0,
               cell_calibration=None, thermal_mode="isothermal", ambient_temperature_c=None):
    if thermal_mode not in ("isothermal", "lumped"):
        raise ValueError("Thermal mode must be isothermal or lumped")
    if thermal_mode == "lumped":
        if (isinstance(ambient_temperature_c, bool) or not isinstance(ambient_temperature_c, (int, float))
                or not np.isfinite(ambient_temperature_c) or ambient_temperature_c <= -273.15):
            raise ValueError("Lumped thermal replay requires explicit finite ambient temperature in Celsius")
    elif ambient_temperature_c is not None:
        raise ValueError("Ambient temperature is used with lumped thermal replay")
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Replay output already exists: {output}")
    model_path = ROOT / "configs/model.json"
    paths = {"raw_csv": Path(csv_path), "metadata": Path(metadata_path), "pack_config": Path(config_path),
             "model_config": model_path, "replay_code": Path(__file__),
             "input_code": Path(__file__).with_name("pack_input.py")}
    if cell_calibration == "m50-c-joint-v3":
        paths["cell_calibration"] = ROOT / "configs/m50_c_joint_v3.json"
    elif cell_calibration is not None:
        raise ValueError(f"Unsupported cell calibration: {cell_calibration}")
    sources = {key: {"path": str(p.resolve()), "sha256": _sha(p)} for key, p in paths.items()}
    config = PackConfig.read(config_path)
    cycles, meta = load_pack_input(csv_path, metadata_path, config)
    segment = select_segment(cycles, meta, cycle_id=cycle_id, duration_s=duration_s)
    cycle, values, intervals = segment["cycle"], segment["values"], segment["intervals"]
    model_config = _json(model_path)
    calibration = None
    if cell_calibration is not None:
        calibration = _json(paths["cell_calibration"])
        if (calibration["id"] != cell_calibration or calibration["family"] != "SPMe"
                or calibration["parameter_set"] != model_config["parameter_set"]):
            raise ValueError("Cell calibration and replay model are incompatible")
        segment["material_overrides"] = calibration["material_overrides"]
        segment["initial_particle_lithium_Ah"] = calibration["initial_particle_lithium_Ah"]
    if thermal_mode == "lumped":
        segment["ambient_temperature_c"] = float(ambient_temperature_c)
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "schema": "pack_spme_replay_v1", "status": "running", "source_files": sources,
        "scope": "short input-to-SPMe integration diagnostic; supplied ROM data, not independent real-vehicle validation",
        "vehicle_id": cycle.vehicle_id, "cycle_id": cycle_id, "source_rows": len(values),
        "source_start_timestamp_s": float(values[0, 0]), "source_end_timestamp_s": float(values[-1, 0]),
        "requested_duration_s": float(duration_s), "selected_duration_s": float(intervals[-1, 1]),
        "selection_policy": "cycle start to last recorded sample at or before duration; no synthetic endpoint",
        "interval_count": len(intervals),
        "model": {"family": "SPMe", "parameter_set": model_config["parameter_set"], "options": model_config["options"]},
        "mapping": {"series_cells": config.series_cells, "parallel_cells": config.parallel_cells,
                    "topology_source": config.source, "cell_basis_id": config.cell_basis_id,
                    "assumption": "average series voltage and equal parallel current sharing"},
        "initialization": {"model_soc": segment["initial_soc"], "source": "metadata completed_cycle_metrics.initial_soc_bms_pct",
                           "assumption": "BMS SOC mapped to homogeneous fresh model state; no prior history transferred or voltage fitted"},
        "temperature": {"mode": "fixed_isothermal_at_initial_module_sensor",
                        "model_temperature_c": segment["temperature_c"],
                        "sensor_channel": meta["public_export"]["temperature_channel_id"],
                        "observed_sensor_range_c": [float(values[:, 3].min()), float(values[:, 3].max())],
                        "assumption": "module sensor proxy, not verified cell-core or ambient temperature"},
        "current_policy": "right-endpoint hold independently solved on each source interval; no smoothing or cross-cycle integration",
        "voltage_comparison": "interval end, before next current switch; source row0 excluded because first applied current is row1",
        "physical_model_compatibility_verified": False, "lifetime_validated": False,
        "remaining_life_days": None, "soh_percent": None,
    }
    if calibration is not None:
        report["model"]["cell_calibration"] = {
            "id": calibration["id"],
            "source": calibration["source"],
            "material_overrides": calibration["material_overrides"],
            "initialization_policy": "Recompute homogeneous concentrations at input BMS SOC using calibrated lithium inventory",
        }
    if thermal_mode == "lumped":
        report["model"]["options"] = {**model_config["options"], "thermal": thermal_mode}
        report["model"]["variant"] = "SPMeT_lumped"
        report["temperature"].pop("model_temperature_c")
        report["temperature"].update(
            mode="lumped_cell_average",
            initial_cell_temperature_c=segment["temperature_c"],
            ambient_temperature_c=segment["ambient_temperature_c"],
            assumption="First module sensor used as initial cell-average proxy; constant ambient supplied by caller",
        )
    started = time.perf_counter()
    report_path = output / "result.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    try:
        trace, numerics = _simulate_intervals(segment, model_config)
        if len(trace) != len(intervals):
            raise ValueError("Not all input intervals were returned")
        for record, raw in zip(trace, values[1:]):
            observed = float(raw[1] / config.series_cells)
            record.update(source_timestamp_s=float(raw[0]), observed_pack_voltage_v=float(raw[1]),
                          observed_cell_voltage_v=observed, observed_module_temperature_c=float(raw[3]),
                          model_pack_voltage_estimate_v=record["model_voltage_end_v"] * config.series_cells,
                          voltage_error_v=record["model_voltage_end_v"] - observed)
        signed = float(np.sum(intervals[:, 2] * (intervals[:, 1] - intervals[:, 0])) / 3600)
        predicted = trace[-1]["model_discharge_capacity_end_ah"] - trace[0]["model_discharge_capacity_start_ah"]
        residual = predicted - signed
        if abs(residual) > CHARGE_TOLERANCE_AH:
            raise ValueError("Total charge balance failed")
        if any(_sha(p) != sources[key]["sha256"] for key, p in paths.items()):
            raise ValueError("Input or implementation changed during replay")
        error = np.array([r["voltage_error_v"] for r in trace])
        voltages = np.array([r["model_voltage_end_v"] for r in trace])
        if thermal_mode == "lumped":
            modeled = [trace[0]["model_cell_temperature_start_c"],
                       *(row["model_cell_temperature_end_c"] for row in trace)]
            differences = np.asarray(modeled[1:]) - values[1:, 3]
            report["temperature"].update(
                modeled_cell_range_c=[min(modeled), max(modeled)],
                modeled_cell_end_c=modeled[-1],
                cell_average_vs_module_sensor_rmse_c=float(np.sqrt(np.mean(differences**2))),
                comparison_scope="Simulated cell average compared to supplied ROM module sensor",
            )
        report.update(status="complete", numerics=numerics, trace=trace,
                      summary={"completed_duration_s": trace[-1]["elapsed_end_s"],
                               "signed_cell_discharge_ah": signed, "model_capacity_change_ah": predicted,
                               "total_charge_balance_residual_ah": residual,
                               "max_interval_charge_residual_ah": max(abs(r["charge_balance_residual_ah"]) for r in trace),
                               "cell_voltage_rmse_v": float(np.sqrt(np.mean(error**2))),
                               "cell_voltage_bias_v": float(np.mean(error)),
                               "cell_voltage_max_abs_error_v": float(np.max(abs(error))),
                               "model_cell_voltage_range_v": [float(voltages.min()), float(voltages.max())],
                               "charge_guard_ah": CHARGE_TOLERANCE_AH,
                               "input_hashes_unchanged": True},
                      validation={"full_interval_coverage": True, "current_application_verified": True,
                                  "differential_state_continuity_verified": True,
                                  "charge_balance_verified": True,
                                  "voltage_accuracy_accepted": False,
                                  "voltage_acceptance_reason": "No independent accuracy threshold or physical model admission supplied for this pack"})
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_wall_s"] = time.perf_counter() - started
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
