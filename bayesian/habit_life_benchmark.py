"""Validate waveform model settings against paired synthetic cell lifetime."""
from __future__ import annotations

from itertools import product
import math
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .habit_life import (DEPARTURE_SOC, RECHARGE_A, RECHARGE_START_S, TEMPERATURE_C,
                         THRESHOLD, _save, _sha, build_daily_duty)
from .habit_life_fast import fast_lifetime
from .habit_stress import ROUTE, TRAFFIC
from .habit_waveform import build_waveform_split, fit_waveform, paired_charge_metrics
from .scenario_gp import ELAPSED_S, ROUTES


BO_BUDGET = 10
METHODS = ("gp", "quadratic")
DEFAULT_PARAMETERS = {"gp": {"length_scale": .5, "log10_ridge": -4.},
                      "quadratic": {"log10_ridge": -8.}}


def _lifetime(currents):
    grid, cumulative, duty = build_daily_duty(ELAPSED_S, currents)
    result = fast_lifetime(grid, cumulative)
    if result["crossing_day"] is None:
        raise ValueError("80% capacity was not reached in the common simulation horizon")
    return {"crossing_day": float(result["crossing_day"]),
            "crossing_bracket_days": result["soh80_crossing_day_bracket"],
            "daily_efc": float(result["daily_efc"]), "daily_duty": duty, "numerics": result["numerics"]}


def _trial_reference(split, progress):
    if progress:
        progress("Simulating the original steady reference waveform sequence")
    ref = _lifetime(split.reference_current_A)
    targets = []
    for index, ramps in enumerate(split.ramps_s):
        if progress:
            progress(f"Simulating original target sequence {index + 1}/{len(split.ramps_s)}: {tuple(ramps)}")
        result = _lifetime(split.target_current_A[:, index, :])
        targets.append({"ramps_s": list(ramps), **result,
                        "delta_life_days": result["crossing_day"] - ref["crossing_day"]})
    return {"reference": ref, "targets": targets}


def _repeat_mean(profile, split):
    curve = np.asarray(profile, dtype=float)
    if curve.shape != (len(ELAPSED_S), 2) or not np.array_equal(curve[:, 0], ELAPSED_S):
        raise ValueError("The forecaster must return current on the common time grid")
    return np.broadcast_to(curve[:, 1], split.reference_current_A.shape)


def _evaluate(model, split, truth, predicted_reference, *, detail=False):
    baseline = np.asarray(model.reference_profile(), dtype=float)[:, 1]
    errors, current_errors, cases = [], [], []
    for index, true in enumerate(truth["targets"]):
        curve = np.asarray(model.predict(*true["ramps_s"]), dtype=float)
        predicted = _lifetime(_repeat_mean(curve, split))
        delta = predicted["crossing_day"] - predicted_reference["crossing_day"]
        error = delta - true["delta_life_days"]
        errors.append(error)
        actual = split.target_current_A[:, index, :]
        current_errors.append(float(np.mean((curve[:, 1] - actual)**2)))
        case = {"ramps_s": true["ramps_s"], "predicted_lifetime": predicted,
                "predicted_delta_life_days": delta, "raw_delta_life_days": true["delta_life_days"],
                "delta_life_error_days": error}
        if detail:
            forecast_charge = paired_charge_metrics(curve[:, 1], baseline)
            actual_charge = paired_charge_metrics(actual, split.reference_current_A)
            case.update(predicted_current_A=curve[:, 1].tolist(), predicted_residual_A=(curve[:, 1] - baseline).tolist(),
                        predicted_excess_discharge_Ah=float(forecast_charge["positive_discharge_excess_Ah"]),
                        raw_mean_excess_discharge_Ah=float(np.mean(actual_charge["positive_discharge_excess_Ah"])),
                        predicted_signed_net_change_Ah=float(forecast_charge["signed_net_delta_Ah"]),
                        raw_mean_signed_net_change_Ah=float(np.mean(actual_charge["signed_net_delta_Ah"])))
        cases.append(case)
    raw_delta = np.array([case["raw_delta_life_days"] for case in cases])
    predicted_delta = np.array([case["predicted_delta_life_days"] for case in cases])
    return {"delta_life_rmse_days": float(np.sqrt(np.mean(np.square(errors)))),
            "delta_life_mae_days": float(np.mean(np.abs(errors))),
            "absolute_target_life_rmse_days": float(np.sqrt(np.mean(np.square([
                case["predicted_lifetime"]["crossing_day"] - target["crossing_day"]
                for case, target in zip(cases, truth["targets"])])))),
            "current_rmse_A": float(np.sqrt(np.mean(current_errors))),
            "delta_sign_agreement_count": int(np.count_nonzero(np.sign(raw_delta) == np.sign(predicted_delta))),
            "target_count": len(cases), "cases": cases}


def _candidates(method):
    if method == "gp":
        lengths = sorted(set(np.geomspace(.12, 2., 9)) | {.5})
        ridges = sorted(set(np.linspace(-8., -.3, 13)) | {-4.})
        params = [{"length_scale": float(a), "log10_ridge": float(b)} for a, b in product(lengths, ridges)]
        unit = [[(math.log(p["length_scale"]) - math.log(.12)) / math.log(2. / .12),
                 (p["log10_ridge"] + 8.) / 7.7] for p in params]
        initial = [DEFAULT_PARAMETERS[method], params[0], params[-1],
                   {"length_scale": .12, "log10_ridge": -.3}, {"length_scale": 2., "log10_ridge": -8.}]
    else:
        params = [{"log10_ridge": float(b)} for b in np.linspace(-8., -.3, 25)]
        unit = [[(p["log10_ridge"] + 8.) / 7.7] for p in params]
        initial = [params[i] for i in (0, 24, 12, 6, 18)]
    return params, np.array(unit), [params.index(p) for p in initial]


def _tune(method, train, validation, truth, predicted_reference, progress, on_iteration):
    params, unit, initial = _candidates(method)
    history, used = [], []
    for iteration in range(BO_BUDGET):
        if iteration < len(initial):
            index = initial[iteration]
        else:
            values = np.array([row["validation_delta_life_rmse_days"] for row in history])
            surrogate = GaussianProcessRegressor(kernel=Matern(length_scale=.4, length_scale_bounds="fixed", nu=2.5),
                                                 alpha=1e-6, normalize_y=True, optimizer=None)
            surrogate.fit(unit[used], values)
            mean, std = surrogate.predict(unit, return_std=True)
            std = np.maximum(std, 1e-12)
            improvement = values.min() - mean
            z = improvement / std
            expected_improvement = improvement * ndtr(z) + std * np.exp(-z**2 / 2) / np.sqrt(2 * np.pi)
            expected_improvement[used] = -np.inf
            index = int(np.argmax(expected_improvement))
        used.append(index)
        parameters = params[index]
        model = fit_waveform(train, method, **parameters)
        score = _evaluate(model, validation, truth, predicted_reference)
        loss = score["delta_life_rmse_days"]
        history.append({"iteration": iteration + 1, "parameters": parameters,
                        "proposal": "initial_design" if iteration < len(initial) else "expected_improvement",
                        "validation_delta_life_rmse_days": loss, "validation_current_rmse_A": score["current_rmse_A"],
                        "validation_delta_predictions_days": [case["predicted_delta_life_days"] for case in score["cases"]]})
        on_iteration(method, history)
        if progress:
            progress(f"{method} BO {iteration + 1}/{BO_BUDGET}: validation lifetime-difference RMSE {loss:.3f} days")
    best = min(history, key=lambda row: row["validation_delta_life_rmse_days"])
    return best


def run_life_model_benchmark(output, *, progress=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    target = output / "habit_life_model_benchmark.json"
    root = Path(__file__).resolve().parent
    sources = {name: _sha(root / name) for name in (
        "habit_life_benchmark.py", "habit_life_fast.py", "habit_waveform.py", "habit_life.py", "habit_stress.py", "scenario_gp.py")}
    report = {"schema": "virtual_habit_lifetime_model_comparison_v1", "status": "running",
              "scope": "Virtual waveform forecast and BLAST-Lite LG M50 cell projection",
              "policy": {"route": ROUTE, "traffic": TRAFFIC, "trips_per_day": 96,
                         "distance_km_per_day": 96 * sum(ROUTES[ROUTE][0]) / 1000,
                         "temperature_C": TEMPERATURE_C, "departure_soc": DEPARTURE_SOC,
                         "recharge_start_s": RECHARGE_START_S, "recharge_A": RECHARGE_A,
                         "capacity_threshold": THRESHOLD, "steps_per_day": 1},
              "bo": {"objective": "minimize validation RMSE in target-minus-steady crossing day",
                     "surrogate": "Matérn GP over a bounded grid with expected improvement",
                     "evaluations_per_method": BO_BUDGET, "histories": {}},
              "source_sha256": sources}

    def on_iteration(method, history):
        report["bo"]["histories"][method] = history
        _save(target, report)

    try:
        train, validation = (build_waveform_split(name) for name in ("train", "validation"))
        report["splits"] = {split.name: {"environment_seeds": list(split.seeds),
                                        "ramps_s": np.asarray(split.ramps_s).tolist()}
                            for split in (train, validation)}
        initial_model = fit_waveform(train, "gp", **DEFAULT_PARAMETERS["gp"])
        shared_profile = initial_model.reference_profile()
        common_reference = _lifetime(_repeat_mean(shared_profile, validation))
        report["predicted_reference_lifetime"] = common_reference
        report["reference_current_A"] = np.asarray(shared_profile)[:, 1].tolist()
        report["elapsed_s"] = ELAPSED_S.tolist()
        report["validation_raw"] = _trial_reference(validation, progress)
        _save(target, report)
        chosen = {method: _tune(method, train, validation, report["validation_raw"], common_reference,
                                progress, on_iteration) for method in METHODS}
        report["bo"]["chosen"] = chosen
        _save(target, report)
        # The held-out raw lifetime and forecast errors are scored after both choices are fixed.
        test = build_waveform_split("test")
        report["splits"]["test"] = {"environment_seeds": list(test.seeds),
                                     "ramps_s": np.asarray(test.ramps_s).tolist()}
        report["test_raw"] = _trial_reference(test, progress)
        report["test_results"] = {}
        for method, choice in chosen.items():
            model = fit_waveform(train, method, **choice["parameters"])
            if not np.array_equal(model.reference_profile(), shared_profile):
                raise ValueError("Both forecasters must use the same training reference")
            report["test_results"][method] = _evaluate(model, test, report["test_raw"], common_reference, detail=True)
        report["lower_test_delta_rmse_method"] = min(
            METHODS, key=lambda method: report["test_results"][method]["delta_life_rmse_days"])
        np.savez_compressed(output / "heldout_waveforms.npz", time_s=ELAPSED_S,
                            reference_current_A=test.reference_current_A, target_current_A=test.target_current_A,
                            ramps_s=test.ramps_s, environment_seeds=test.seeds)
        report["heldout_waveforms_sha256"] = _sha(output / "heldout_waveforms.npz")
        if any(_sha(root / name) != digest for name, digest in sources.items()):
            raise ValueError("A benchmark source changed during execution")
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        _save(target, report)
    return report
