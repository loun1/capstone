"""Compare repeated virtual driving using a separate empirical LG M50 cell model."""
from __future__ import annotations

from hashlib import sha256
from importlib.metadata import version
import inspect
import json
from pathlib import Path

import numpy as np

from .habit_predictor import predict_habit_burden
from .habit_stress import FACTORIAL_ARMS, FACTORIAL_SEEDS, ROUTE, TRAFFIC, current_burdens


DAY_S = 86400.0
YEAR_DAYS = 365
MAX_DAYS = 20 * YEAR_DAYS
THRESHOLD = .8
DEPARTURE_SOC = .8
TEMPERATURE_C = 25.0
RECHARGE_START_S = 8 * 3600.0
RECHARGE_A = 1.5
SEED_SEQUENCE_REPEATS = 3
BLAST_VERSION = "1.1.0"


def _sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _charge_knots(time, current):
    """Integrate linear current, preserving turning points at zero crossings."""
    crossings = np.flatnonzero(current[:-1] * current[1:] < 0)
    left = current[crossings]
    right = current[crossings + 1]
    roots = time[crossings] - left * (time[crossings + 1] - time[crossings]) / (right - left)
    grid = np.sort(np.r_[time, roots])
    interpolated = np.interp(grid, time, current)
    increments = np.diff(grid) * (interpolated[:-1] + interpolated[1:]) / 7200
    return grid, np.r_[0.0, np.cumsum(increments)]


def build_daily_duty(time, currents):
    """Assemble all matched trips and recharge at the same rate and clock time."""
    time = np.asarray(time, dtype=float)
    currents = np.asarray(currents, dtype=float)
    if (time.ndim != 1 or len(time) < 2 or time[0] != 0 or np.any(np.diff(time) <= 0)
            or currents.shape != (len(FACTORIAL_SEEDS), len(time))
            or not np.isfinite(time).all() or not np.isfinite(currents).all()
            or np.max(np.abs(currents)) > 5):
        raise ValueError("Expected finite paired driving curves on the declared time grid")
    trips = [_charge_knots(time, current) for current in currents]
    grids, charges = [np.array([0.0])], [np.array([0.0])]
    drawn_Ah, regen_Ah, peak_discharge, peak_regen, i_squared = 0., 0., 0., 0., 0.
    for current, (_, charge) in zip(currents, trips):
        change = np.diff(charge)
        drawn_Ah += float(np.maximum(change, 0).sum())
        regen_Ah += float(np.maximum(-change, 0).sum())
        burdens = current_burdens(np.column_stack((time, current)))
        peak_discharge = max(peak_discharge, burdens["peak_discharge_A"])
        peak_regen = max(peak_regen, burdens["peak_regen_A"])
        i_squared += burdens["i_squared_A2s"]
    offset_s, offset_Ah = 0., 0.
    for grid, charge in trips * SEED_SEQUENCE_REPEATS:
        grids.append(offset_s + grid[1:])
        charges.append(offset_Ah + charge[1:])
        offset_s += grid[-1]
        offset_Ah += charge[-1]
    end_charge_s = RECHARGE_START_S + 3600 * offset_Ah / RECHARGE_A
    if not 0 < offset_s < RECHARGE_START_S < end_charge_s < DAY_S / 2:
        raise ValueError("Driving and recharge must finish in the first half of the virtual day")
    grids.append(np.array([RECHARGE_START_S, end_charge_s, DAY_S / 2, DAY_S]))
    charges.append(np.array([offset_Ah, 0., 0., 0.]))
    grid, cumulative = np.concatenate(grids), np.concatenate(charges)
    if not np.all(np.diff(grid) > 0) or cumulative[0] != cumulative[-1]:
        raise ValueError("Daily profile is not a closed SOC duty")
    expected_Ah = (drawn_Ah - regen_Ah) * SEED_SEQUENCE_REPEATS
    if not np.isclose(offset_Ah, expected_Ah, atol=1e-12, rtol=0):
        raise ValueError("Daily charge ledger mismatch")
    return grid, cumulative, {
        "trips_per_day": len(trips) * SEED_SEQUENCE_REPEATS,
        "driving_duration_s": offset_s,
        "recharge_end_s": end_charge_s,
        "charge_added_Ah": offset_Ah,
        "drive_discharge_Ah": drawn_Ah * SEED_SEQUENCE_REPEATS,
        "drive_regen_Ah": regen_Ah * SEED_SEQUENCE_REPEATS,
        "drive_peak_discharge_A": peak_discharge,
        "drive_peak_regen_A": peak_regen,
        "drive_i_squared_A2s": i_squared * SEED_SEQUENCE_REPEATS,
        "absolute_throughput_including_recharge_Ah": float(np.sum(np.abs(np.diff(cumulative)))),
    }


def _simulate(factory, grid, cumulative, steps_per_day, label, progress):
    model = factory()
    cap = float(model.cap)
    temp = np.full_like(grid, TEMPERATURE_C)
    expected_daily_efc = float(np.sum(np.abs(np.diff(cumulative))) / (2 * cap))
    split = int(np.searchsorted(grid, DAY_S / 2))
    chunks = (slice(None),) if steps_per_day == 1 else (slice(0, split + 1), slice(split, None))
    history = [{"day": 0, "soh_pct": 100., "calendar_loss_pp": 0., "cycle_loss_pp": 0., "efc": 0.}]
    crossing = None
    for day in range(1, MAX_DAYS + 1):
        # Hold the required traction charge fixed as available cell capacity declines.
        for chunk in chunks:
            q = float(model.outputs["q"][-1])
            soc = DEPARTURE_SOC - cumulative[chunk] / (cap * q)
            if not 0 <= soc.min() <= soc.max() <= 1:
                raise ValueError("Required trip exceeds available modeled SOC")
            model.update_battery_state(grid[chunk], soc, temp[chunk])
        q = float(model.outputs["q"][-1])
        calendar = float(100 * (1 - model.outputs["q_t"][-1]))
        cycle = float(100 * (1 - model.outputs["q_EFC"][-1]))
        efc = float(model.stressors["efc"][-1])
        if (not np.isfinite([q, calendar, cycle, efc]).all() or q <= 0
                or abs(model.stressors["t_days"][-1] - day) > 1e-8
                or abs(efc - day * expected_daily_efc) > 1e-7
                or abs(100 * (1 - q) - calendar - cycle) > 1e-10
                or q > history[-1]["soh_pct"] / 100 + 1e-12):
            raise ValueError("BLAST trajectory failed age, capacity or throughput checks")
        history.append({"day": day, "soh_pct": q * 100, "calendar_loss_pp": calendar,
                        "cycle_loss_pp": cycle, "efc": efc})
        if q <= THRESHOLD:
            crossing = [day - 1, day]
            break
        if progress and day % (5 * YEAR_DAYS) == 0:
            progress(f"{label}: {day // YEAR_DAYS} years, modeled SOH {q * 100:.3f}%")
    return {"daily_history": history, "soh80_crossing_day_bracket": crossing,
            "right_censored": crossing is None, "daily_efc": expected_daily_efc,
            "steps_per_day": steps_per_day,
            "end": history[-1]}


def run_habit_life(habit_report, output, *, progress=None):
    """Use the pinned empirical model and report both time-step settings."""
    try:
        from blast.models import Nmc811_GrSi_LGM50_5Ah_Battery as factory
        from blast.models.degradation_model import BatteryDegradationModel
    except ImportError as error:
        raise RuntimeError("Load BLAST-Lite 1.1.0 in an isolated environment with NumPy 2.3.x") from error
    if version("blast-lite") != BLAST_VERSION or not hasattr(np, "trapz"):
        raise ValueError("The comparison requires BLAST-Lite 1.1.0 and NumPy with np.trapz")
    habit_report, output = Path(habit_report), Path(output)
    source = json.loads(habit_report.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent
    context, factorial = source["context"], source["factorial"]
    expected_ramps = {name: list(ramps) for name, ramps in FACTORIAL_ARMS.items()}
    if (source["schema"] != "synthetic_habit_stress_v1" or source["status"] != "complete"
            or (context["route_id"], context["traffic"]) != (ROUTE, TRAFFIC)
            or factorial["seeds"] != list(FACTORIAL_SEEDS) or factorial["ramps_s"] != expected_ramps
            or set(source["source_sha256"]) != {"habit_stress.py", "scenario_gp.py"}
            or any(_sha(root / name) != digest for name, digest in source["source_sha256"].items())):
        raise ValueError("Use the preserved, matched virtual habit benchmark from this checkout")
    cases = {name: build_daily_duty(source["elapsed_s"], factorial["current_A"][name])
             for name in FACTORIAL_ARMS}
    predictions = {name: predict_habit_burden(*ramps) for name, ramps in FACTORIAL_ARMS.items()}
    output.mkdir(parents=True, exist_ok=False)
    _save(output / "quadratic_habit_predictions.json", predictions)
    target = output / "habit_life.json"
    provenance = {"habit_report_sha256": _sha(habit_report), "runner_sha256": _sha(__file__),
                  "blast_version": BLAST_VERSION, "numpy_version": np.__version__,
                  "blast_cell_source_sha256": _sha(inspect.getfile(factory)),
                  "blast_base_source_sha256": _sha(inspect.getfile(BatteryDegradationModel)),
                  "model_source": "https://github.com/NatLabRockies/BLAST-Lite/releases/tag/v1.1.0"}
    report = {
        "schema": "virtual_habit_life_blast_v1", "status": "running", "context": context,
        "scope": "Scenario-specific empirical cell projection; separate from short SPMe and road validation",
        "model": "BLAST-Lite Nmc811_GrSi_LGM50_5Ah_Battery", "provenance": provenance,
        "quadratic_prediction_scope": "Separate scalar burden prediction; lifetime consumes integrated current history",
        "policy": {"day_s": DAY_S, "year_days": YEAR_DAYS, "max_days": MAX_DAYS,
                   "end_capacity_pct": THRESHOLD * 100, "departure_soc": DEPARTURE_SOC,
                   "temperature_C": TEMPERATURE_C, "recharge_start_s": RECHARGE_START_S,
                   "recharge_current_A": RECHARGE_A, "trips_per_day": len(FACTORIAL_SEEDS) * SEED_SEQUENCE_REPEATS,
                   "daily_distance_km": context["distance_m"] * len(FACTORIAL_SEEDS) * SEED_SEQUENCE_REPEATS / 1000,
                   "nominal_cell_capacity_Ah": float(factory().cap), "matched_seed_order": list(FACTORIAL_SEEDS),
                   "fixed_charge_demand": True, "coulombic_efficiency_assumed": 1.0},
        "limits": ["Laboratory cycling fit used 100% depth of discharge; virtual partial cycling extrapolates that fit.",
                   "Model reduces cycling to EFC and average discharge C-rate; no peak-current, charge-rate or heating dependence.",
                   "Constant cell temperature and repeated route schedule. No independent taxi-pack lifetime validation."],
        "daily_duty": {name: case[2] for name, case in cases.items()}, "runs": {},
    }
    _save(target, report)
    try:
        for steps in (1, 2):
            runs = {}
            report["runs"][str(steps)] = runs
            for name, (grid, charge, _) in cases.items():
                label = f"{steps} steps/day {name}"
                if progress:
                    progress(label)
                runs[name] = _simulate(factory, grid, charge, steps, label, progress)
                _save(target, report)
        comparisons = {}
        for step, runs in report["runs"].items():
            steady = runs["steady"]["soh80_crossing_day_bracket"]
            comparisons[step] = {}
            for name, run in runs.items():
                crossing = run["soh80_crossing_day_bracket"]
                comparisons[step][name] = (None if steady is None or crossing is None else
                                           [crossing[0] - steady[1], crossing[1] - steady[0]])
        report.update(status="complete", soh80_difference_vs_steady_day_brackets=comparisons)
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _save(target, report)
    return report
