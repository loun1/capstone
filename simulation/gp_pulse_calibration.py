"""Fit one SPMeT parameter on measured GITT pulses using GP search."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from simulation.joint_spme_bol_fit import derive_updates, homogeneous_state

PROFILE = Path(__file__).resolve().parents[1] / "configs" / "m50_c_joint_v3.json"
FIT_PULSES = (2, 3)
VALIDATION_PULSES = (4, 5, 6)
AMBIENT_C = 25.0
RECOVERY_S = 60.0
RECOVERY_TOLERANCE_MV = 0.1
MESH = {"x_n": 20, "x_s": 20, "x_p": 20, "r_n": 30, "r_p": 30}
CONTACT = "Contact resistance [Ohm]"
EXCHANGE = "Electrode exchange-current multiplier"


@dataclass(frozen=True)
class SearchSpace:
    parameter: str
    baseline: float
    lower: float
    upper: float
    validation_pulses: tuple[int, ...]


SPACES = {
    "contact-resistance": SearchSpace(CONTACT, 0.0, 0.0, 0.020, VALIDATION_PULSES),
    "exchange-current": SearchSpace(EXCHANGE, 1.0, 0.5, 2.0, (7, 8, 9)),
}


@dataclass(frozen=True)
class Pulse:
    number: int
    time_s: np.ndarray
    current_A: np.ndarray
    voltage_V: np.ndarray
    surface_C: np.ndarray
    loaded: np.ndarray
    rest_voltage_V: float


def read_pulses(path, validation_pulses):
    with np.load(path) as data:
        time, current, voltage, surface, loaded = (
            data[name] for name in ("t", "i", "v", "temperature_C", "loaded")
        )
    if time.ndim != 1 or any(a.shape != time.shape for a in (current, voltage, surface, loaded)):
        raise ValueError("Pulse arrays must share a one-dimensional time axis")
    if not all(np.isfinite(a).all() for a in (time, current, voltage, surface, loaded)):
        raise ValueError("Pulse input contains a nonfinite value")
    if not np.all(np.diff(time) > 0) or not np.isin(loaded, (0, 1)).all():
        raise ValueError("Pulse input requires increasing time and a binary loaded mask")
    boundaries = np.flatnonzero(np.diff(loaded.astype(int), prepend=0, append=0))
    starts, stops = boundaries[::2], boundaries[1::2]
    if len(starts) < max(validation_pulses):
        raise ValueError("Input does not contain the declared fit and validation pulses")

    pulses = {}
    for number in FIT_PULSES + validation_pulses:
        start, stop = starts[number - 1], stops[number - 1]
        first = start - 1
        if stop >= len(time) or time[-1] < time[stop] + RECOVERY_S:
            raise ValueError(f"Pulse {number} does not have a full recovery window")
        last = np.searchsorted(time, time[stop] + RECOVERY_S, side="right")
        next_start = starts[number] if number < len(starts) else len(time)
        rest_start = np.searchsorted(time, time[first] - 10)
        if rest_start < stops[number - 2] or last > next_start or last <= stop + 1:
            raise ValueError(f"Pulse {number} does not have the declared rest and recovery windows")
        rows = slice(first, last)
        pulses[number] = Pulse(
            number, time[rows] - time[first], current[rows], voltage[rows], surface[rows],
            loaded[rows].astype(bool), float(np.median(voltage[rest_start:start])),
        )
    return pulses


def scaled_exchange_current(original, multiplier):
    def exchange_current(electrolyte_concentration, surface_concentration, maximum_concentration, temperature):
        return multiplier * original(electrolyte_concentration, surface_concentration, maximum_concentration, temperature)

    return exchange_current


def build_simulation(pulse, profile, space):
    import pybamm

    params = pybamm.ParameterValues(profile["parameter_set"])
    params.update(profile["material_overrides"])
    _, inventory = derive_updates(params, 1.0, 1.0, 0.0)
    if not np.isclose(inventory["Q_Li_Ah"], profile["initial_particle_lithium_Ah"], rtol=1e-8):
        raise ValueError("Frozen particle lithium inventory does not match the profile")
    state, anchor = homogeneous_state(params, inventory, pulse.rest_voltage_V)
    params.update(state)
    params.update({
        "Current function [A]": pybamm.Interpolant(pulse.time_s, pulse.current_A, pybamm.t),
        "Initial temperature [K]": float(pulse.surface_C[0]) + 273.15,
        "Ambient temperature [K]": AMBIENT_C + 273.15,
    })
    options = {"thermal": "lumped"}
    if space.parameter == CONTACT:
        options["contact resistance"] = "true"
        params.update({CONTACT: "[input]"})
    else:
        multiplier = pybamm.InputParameter(EXCHANGE)
        for electrode in ("Negative", "Positive"):
            name = f"{electrode} electrode exchange-current density [A.m-2]"
            params.update({name: scaled_exchange_current(params[name], multiplier)})
    model = pybamm.lithium_ion.SPMe(options=options)
    sim = pybamm.Simulation(
        model, parameter_values=params, var_pts=MESH,
        solver=pybamm.IDAKLUSolver(rtol=1e-6, atol=1e-8),
    )
    return sim, anchor


def predict(pulse, sim, parameter, value):
    sol = sim.solve(
        t_eval=[0, float(pulse.time_s[-1])], t_interp=pulse.time_s,
        inputs={parameter: value},
    )
    if sol.t[-1] < pulse.time_s[-1] - 1e-8:
        raise ValueError(f"Pulse {pulse.number} stopped early: {sol.termination}")
    voltage = np.asarray(sol["Voltage [V]"](pulse.time_s))
    temperature = np.asarray(sol["Volume-averaged cell temperature [C]"](pulse.time_s))
    if any(a.shape != pulse.time_s.shape or not np.isfinite(a).all() for a in (voltage, temperature)):
        raise ValueError(f"Pulse {pulse.number} produced invalid predictions")
    return voltage, temperature


def score(pulse, voltage, temperature):
    residual = 1_000 * (voltage - pulse.voltage_V)
    recovery = (np.arange(len(residual)) > 0) & ~pulse.loaded
    return {
        "loaded_voltage_rmse_mV": float(np.sqrt(np.mean(residual[pulse.loaded] ** 2))),
        "recovery_voltage_rmse_mV": float(np.sqrt(np.mean(residual[recovery] ** 2))),
        "cell_mean_vs_surface_temperature_rmse_C": float(np.sqrt(np.mean((temperature - pulse.surface_C) ** 2))),
        "loaded_samples": int(pulse.loaded.sum()),
        "recovery_samples": int(recovery.sum()),
    }


def search(evaluate, space):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern

    width = space.upper - space.lower
    baseline = (space.baseline - space.lower) / width
    initial = dict.fromkeys((baseline, 0.0, 0.25, 0.5, 0.75, 1.0))
    samples = [(x, evaluate(space.lower + x * width), "initial") for x in initial]

    grid = np.linspace(0, 1, 101)
    for _ in range(5):
        xs = np.asarray([x for x, _, _ in samples])
        errors = [error for _, error, _ in samples]
        gp = GaussianProcessRegressor(
            kernel=Matern(length_scale=0.20, nu=2.5), alpha=1e-6, optimizer=None, normalize_y=True,
        ).fit(xs[:, None], errors)
        candidates = grid[np.min(np.abs(grid[:, None] - xs), axis=1) > 1e-8]
        mean, std = gp.predict(candidates[:, None], return_std=True)
        x = float(candidates[np.argmin(mean - 1.5 * std)])
        samples.append((x, evaluate(space.lower + x * width), "gp_lower_confidence_bound"))

    return [{"parameter_value": space.lower + x * width,
             "mean_fit_loaded_rmse_mV": error, "selection": source} for x, error, source in samples]


def run(observed, output, parameter="contact-resistance"):
    observed, output = Path(observed), Path(output)
    if output.exists():
        raise FileExistsError(output)
    profile = json.loads(PROFILE.read_text(encoding="utf-8"))
    space = SPACES[parameter]
    validation_pulses = space.validation_pulses
    pulses = read_pulses(observed, validation_pulses)
    simulations = {n: build_simulation(pulses[n], profile, space) for n in FIT_PULSES}

    def evaluate(value):
        return float(np.mean([
            score(pulses[n], *predict(pulses[n], simulations[n][0], space.parameter, value))["loaded_voltage_rmse_mV"]
            for n in FIT_PULSES
        ]))

    trials = search(evaluate, space)
    best = min(trials, key=lambda trial: trial["mean_fit_loaded_rmse_mV"])
    candidate = best["parameter_value"]
    simulations.update({n: build_simulation(pulses[n], profile, space) for n in validation_pulses})
    comparisons, traces = {}, {}
    for n, pulse in pulses.items():
        sim, anchor = simulations[n]
        base_voltage, base_temp = predict(pulse, sim, space.parameter, space.baseline)
        fit_voltage, fit_temp = predict(pulse, sim, space.parameter, candidate)
        comparisons[n] = {
            "split": "fit" if n in FIT_PULSES else "validation",
            "pre_pulse_equilibrium": anchor,
            "baseline": score(pulse, base_voltage, base_temp),
            "candidate": score(pulse, fit_voltage, fit_temp),
        }
        for field, values in (
            ("time_s", pulse.time_s), ("current_A", pulse.current_A), ("loaded", pulse.loaded),
            ("measured_voltage_V", pulse.voltage_V), ("measured_surface_C", pulse.surface_C),
            ("baseline_voltage_V", base_voltage), ("candidate_voltage_V", fit_voltage),
            ("baseline_cell_mean_C", base_temp), ("candidate_cell_mean_C", fit_temp),
        ):
            traces[f"pulse_{n}_{field}"] = values

    mean_validation = {
        name: {metric: float(np.mean([comparisons[n][name][metric] for n in validation_pulses]))
               for metric in ("loaded_voltage_rmse_mV", "recovery_voltage_rmse_mV",
                              "cell_mean_vs_surface_temperature_rmse_C")}
        for name in ("baseline", "candidate")
    }
    base, fit = mean_validation["baseline"], mean_validation["candidate"]
    accepted = (
        fit["loaded_voltage_rmse_mV"] < base["loaded_voltage_rmse_mV"]
        and fit["recovery_voltage_rmse_mV"] <= base["recovery_voltage_rmse_mV"] + RECOVERY_TOLERANCE_MV
    )
    report = {
        "status": "complete",
        "scope": "Recorded GITT discharge pulses; same-cell held-out pulse windows",
        "source": str(observed.resolve()),
        "source_sha256": hashlib.sha256(observed.read_bytes()).hexdigest(),
        "profile": profile["id"],
        "profile_sha256": hashlib.sha256(PROFILE.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mesh_points": MESH,
        "fit_pulses": FIT_PULSES,
        "validation_pulses": validation_pulses,
        "ambient_setpoint_C": AMBIENT_C,
        "initial_temperature_source": "pre-pulse measured surface temperature as mean-cell proxy",
        "current_source": "recorded current, positive discharge, linearly interpolated between raw samples",
        "recovery_window_s": RECOVERY_S,
        "parameter": space.parameter,
        "baseline_parameter_value": space.baseline,
        "parameter_search_bounds": [space.lower, space.upper],
        "fit_mean_loaded_rmse_mV_baseline": trials[0]["mean_fit_loaded_rmse_mV"],
        "trials": trials,
        "candidate_parameter_value": candidate,
        "selected_parameter_value": candidate if accepted else space.baseline,
        "adoption_gate": {"requires_lower_loaded_mean_rmse": True,
                          "maximum_recovery_mean_rmse_increase_mV": RECOVERY_TOLERANCE_MV,
                          "candidate_accepted": accepted},
        "validation_mean_of_pulse_rmse": mean_validation,
        "pulses": comparisons,
    }
    output.mkdir(parents=True)
    np.savez_compressed(output / "traces.npz", **traces)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parameter", choices=SPACES, default="contact-resistance")
    args = parser.parse_args()
    result = run(args.observed, args.output, args.parameter)
    print(json.dumps({key: result[key] for key in (
        "status", "parameter", "candidate_parameter_value", "selected_parameter_value",
        "adoption_gate", "validation_mean_of_pulse_rmse",
    )}, indent=2))
