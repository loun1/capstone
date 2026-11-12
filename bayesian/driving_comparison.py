"""Matched driving plus recovery comparison with aligned SOC, time and RPT.

This is an exploratory virtual-cell protocol, not pure driving damage or RUL.
Driving curves are piecewise linear; raw pack right-held inputs are not silently
reinterpreted. Different net charge is reported and prevents a duty-matched rank.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from simulation.rebuilt_lifetime import Engine, fingerprint, scalar, sha, write_json, pybamm
from .charging import _audit, _combine_audits, _digest, _lithium, _run_reference


@dataclass(frozen=True)
class Guards:
    voltage_lower_limit_V: float = 2.5
    voltage_upper_limit_V: float = 4.2
    voltage_tolerance_V: float = 1e-6
    current_abs_limit_A: float = 5.0
    conservation_relative_error_limit: float = 1e-6


def validate_comparison(profiles, task_context, *, horizon_s=7200.0, temperature_C=25.0,
                        duty_tolerance_Ah=1e-4):
    if not isinstance(profiles, Mapping) or not 2 <= len(profiles) <= 3:
        raise ValueError("Supply two or three named driving profiles")
    if not isinstance(task_context, Mapping):
        raise ValueError("Explicit shared task context required")
    for key in ("route_id", "direction", "traffic", "cell_basis_id", "source_kind"):
        if not isinstance(task_context.get(key), str) or not task_context[key].strip():
            raise ValueError(f"Shared task requires {key}")
    if task_context["cell_basis_id"] not in {"okane2022-spme-single-cell", "synthetic-m50t-single-cell"}:
        raise ValueError("Profiles must explicitly use the exploratory model cell basis")
    distance = task_context.get("distance_m")
    if isinstance(distance, bool) or not isinstance(distance, (int, float)) or not np.isfinite(distance) or distance <= 0:
        raise ValueError("Positive shared distance_m required")
    if not np.isfinite([horizon_s, temperature_C, duty_tolerance_Ah]).all():
        raise ValueError("Finite comparison settings required")
    if not 3600 <= horizon_s <= 10800 or not 10 <= temperature_C <= 40 or not 0 <= duty_tolerance_Ah <= .001:
        raise ValueError("Comparison settings exceed bounded diagnostic scope")
    clean, charges = {}, {}
    duration = None
    for name, profile in profiles.items():
        if not isinstance(name, str) or not name.strip() or name == "no_drive_control":
            raise ValueError("Each profile needs a nonreserved name")
        data = np.array(profile, dtype=float, copy=True)
        if (data.ndim != 2 or data.shape[1] != 2 or not 3 <= len(data) <= 1001
                or not np.isfinite(data).all() or data[0, 0] != 0
                or np.any(np.diff(data[:, 0]) <= 0) or not 1 <= data[-1, 0] <= 900
                or np.max(abs(data[:, 1])) > 5):
            raise ValueError("Finite current curve needs increasing seconds from zero, <=900s and <=5A")
        if duration is not None and abs(data[-1, 0] - duration) > 1e-9:
            raise ValueError("Driving durations must match exactly")
        duration = float(data[-1, 0])
        clean[name] = data
        charges[name] = float(np.trapezoid(data[:, 1], data[:, 0]) / 3600)
    spread = max(charges.values()) - min(charges.values())
    return clean, {"duration_s": duration, "signed_drive_Ah": charges,
                   "net_charge_spread_Ah": spread, "duty_tolerance_Ah": duty_tolerance_Ah,
                   "net_charge_matched": spread <= duty_tolerance_Ah,
                   "context": dict(task_context),
                   "context_verification": "declared by caller; no route/driver labels inferred from current"}


def _side_loss(state):
    return {"sei_loss_Ah": scalar(state, "Loss of capacity to negative SEI [A.h]"),
            "plating_loss_Ah": scalar(state, "Loss of capacity to negative lithium plating [A.h]")}


def _stage_delta(before, after):
    return {key: after[key] - before[key] for key in before}


def _drive(engine, initial, profile, lithium, charge_scale):
    duration = float(profile[-1, 0])
    step = pybamm.step.current(profile, duration=duration, period=min(1.0, duration), skip_ok=False)
    key = "harmonized_driving"
    try:
        result = engine.sim(key, [step]).solve(starting_solution=initial.copy(), calc_esoh=False)
        piece = result.cycles[-1].steps[-1]
        audit = _audit(piece, lithium, charge_scale)
        if piece.termination != "final time" or abs(audit["duration_s"] - duration) > 1e-6:
            raise ValueError("Driving segment did not finish")
        times = np.asarray(piece["Time [s]"].entries)
        local_time = times - times[0]
        current = np.asarray(piece["Current [A]"].entries)
        if not np.allclose(current, np.interp(local_time, profile[:, 0], profile[:, 1]), atol=1e-6, rtol=0):
            raise ValueError("Applied current disagrees with declared linear drive curve")
        charge = np.asarray(piece["Discharge capacity [A.h]"].entries)
        exact_ah = float(np.trapezoid(profile[:, 1], profile[:, 0]) / 3600)
        residual = float(charge[-1] - charge[0] - exact_ah)
        audit["sampled_trapezoid_charge_residual_relative"] = audit["charge_conservation_relative_error"]
        audit["charge_conservation_relative_error"] = abs(residual) / charge_scale
        audit["charge_integral_basis"] = "exact piecewise-linear source curve"
        audit["source_signed_Ah"] = exact_ah
        audit["source_charge_residual_Ah"] = residual
        if audit["charge_conservation_relative_error"] > Guards().conservation_relative_error_limit:
            raise ValueError("Drive current integral differs from model capacity change")
        return result.last_state.copy(), audit
    finally:
        engine.sims.pop(key, None)


def _evaluate_arm(engine, initial, bounds, profile, horizon_s, lithium, charge_scale):
    initial_hash = fingerprint(initial)
    initial_losses = _side_loss(initial)
    end, drive = _drive(engine, initial, profile, lithium, charge_scale)
    after_drive_losses = _side_loss(end)
    after_drive_soc = engine.soc(end, bounds)
    if not .2 < after_drive_soc <= .8 + .001:
        raise ValueError("Drive end SOC must be above recovery20% and no higher than initial80%")
    audits = [drive]
    end, recovery, _, _ = engine.advance(end, "DIS", horizon_s - drive["duration_s"], 5.0, bounds)
    audits.append(_audit(recovery, lithium, charge_scale))
    if "operating SOC" not in recovery.termination:
        raise ValueError("Common5A recovery did not reachSOC20%")
    soc_recovered = engine.soc(end, bounds)
    if abs(soc_recovered - .2) > .001:
        raise ValueError("Recovery SOC mismatch")
    after_recovery_losses = _side_loss(end)
    remaining = horizon_s - sum(a["duration_s"] for a in audits)
    if remaining <= 0:
        raise ValueError("Recovery exceeded common comparison horizon")
    end, rest, _, _ = engine.advance(end, "REST", remaining, 0, bounds)
    audits.append(_audit(rest, lithium, charge_scale))
    if rest.termination != "final time":
        raise ValueError("Common-horizon rest did not finish")
    before_rpt_losses = _side_loss(end)
    final_soc = engine.soc(end, bounds)
    horizon = sum(a["duration_s"] for a in audits)
    if abs(horizon - horizon_s) > 1e-6 or abs(final_soc - .2) > .001:
        raise ValueError("RPT start time/SOC outside shared tolerances")
    rpt, rpt_audits = _run_reference(engine, end, lithium, charge_scale)
    if fingerprint(initial) != initial_hash:
        raise ValueError("Shared initial state was mutated")
    return {"after_drive_soc": after_drive_soc, "recovery_end_soc": soc_recovered,
            "rpt_start_soc": final_soc, "horizon_s": horizon,
            "rpt_capacity_Ah": rpt["Qref_Ah"], "main_audits": audits, "rpt_audits": rpt_audits,
            "physical": _combine_audits(audits + rpt_audits, Guards()),
            "stage_side_reaction_deltas_Ah": {"drive": _stage_delta(initial_losses, after_drive_losses),
                "recovery": _stage_delta(after_drive_losses, after_recovery_losses),
                "rest": _stage_delta(after_recovery_losses, before_rpt_losses)},
            "side_reaction_interpretation": "model mechanism counters, not measured loss or independently additive RPT capacity"}


def summarize_meshes(mesh_results, duty):
    names = list(duty["signed_drive_Ah"])
    cases = {}
    for name in names:
        losses = [100 * (row["arms"]["no_drive_control"]["rpt_capacity_Ah"] - row["arms"][name]["rpt_capacity_Ah"])
                  / row["arms"]["no_drive_control"]["rpt_capacity_Ah"] for row in mesh_results]
        cases[name] = {"capacity_difference_from_no_drive_control_pp": losses[-1],
                       "mesh_loss_pp": losses,
                       "numerical_error_pp": abs(losses[-1] - losses[0]),
                       "physical_constraints_valid": all(row["arms"][arm]["physical"]["physical_constraints_valid"]
                            for row in mesh_results for arm in (name, "no_drive_control"))}
    ranked = sorted(names, key=lambda n: cases[n]["capacity_difference_from_no_drive_control_pp"])
    resolved = len(mesh_results) == 2 and all(cases[n]["physical_constraints_valid"] for n in names)
    resolved = resolved and all(cases[n]["numerical_error_pp"] <= .001 for n in names)
    resolved = resolved and all(
        cases[ranked[i+1]]["capacity_difference_from_no_drive_control_pp"] - cases[ranked[i]]["capacity_difference_from_no_drive_control_pp"]
        > cases[ranked[i+1]]["numerical_error_pp"] + cases[ranked[i]]["numerical_error_pp"]
        for i in range(len(ranked)-1))
    return {"cases": cases, "paired_numeric_order_resolved": bool(resolved),
            "duty_matched_ranking": ranked if resolved and duty["net_charge_matched"] else None,
            "ranking_reason": "resolved on tested meshes" if resolved and duty["net_charge_matched"]
                              else "duty mismatch or unresolved/invalid numerical comparison",
            "recommendation": None, "lifetime_extension_days": None}


def compare_profiles(profiles, task_context, output, *, horizon_s=7200.0, temperature_C=25.0,
                     duty_tolerance_Ah=1e-4, progress: Callable[[str], None] | None = None):
    """Run3–4arms on two meshes and save failures/progress in an exclusive folder.

    The no-drive control follows the same total time and recovery/RPT procedure.
    This removes the old RPT startSOC confound but does not isolate pure driving
    damage: recovery histories can differ and their counters remain separate.
    """
    curves, duty = validate_comparison(profiles, task_context, horizon_s=horizon_s,
                                      temperature_C=temperature_C, duty_tolerance_Ah=duty_tolerance_Ah)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = {name: sha(root / name) for name in ("bayesian/driving_comparison.py", "bayesian/charging.py",
                "simulation/rebuilt_lifetime.py", "configs/model.json", "configs/diagnostics.json")}
    report = {"schema": "harmonized_driving_comparison_v1", "status": "running",
              "scope": "exploratory drive plus recovery plus rest policy comparison, no lifetime inference",
              "duty": duty, "source_hashes": sources,
              "profiles": {name: data.tolist() for name, data in curves.items()},
              "profile_interpolation": "piecewise linear; not the raw pack right-endpoint adapter",
              "initial_soc_target": .8, "rpt_start_soc_target": .2,
              "temperature_C": temperature_C, "horizon_s": horizon_s,
              "recovery_current_A": 5.0,
              "rpt_protocol": json.loads((root / "configs/diagnostics.json").read_text())["reference_test"],
              "physical_model_compatibility_verified": False, "mesh_results": []}
    target = out / "driving_comparison.json"
    write_json(target, report)
    try:
        for mesh, rtol, atol in ((1, 1e-7, 1e-9), (2, 1e-8, 1e-10)):
            engine = Engine(mesh=mesh, rtol=rtol, atol=atol, temperature_K=temperature_C + 273.15,
                            reference_output_only=False)
            initial = engine.sim("harmonized_initial", [pybamm.step.rest(duration=1)]).solve(initial_soc=.8, calc_esoh=False).last_state.copy()
            bounds = engine.bounds(initial)
            if abs(engine.soc(initial, bounds) - .8) > .001:
                raise ValueError("Common initial SOC mismatch")
            initial_id = fingerprint(initial)
            charge_scale = scalar(initial, "Negative electrode capacity [A.h]")
            lithium = float(_lithium(initial).reshape(-1)[-1])
            result = {"mesh": mesh, "rtol": rtol, "atol": atol,
                      "initial_state_sha256": initial_id, "arms": {}}
            report["mesh_results"].append(result)
            control = np.column_stack([next(iter(curves.values()))[:, 0], np.zeros(len(next(iter(curves.values()))))])
            for name, curve in {"no_drive_control": control, **curves}.items():
                if progress:
                    progress(f"mesh{mesh} {name}")
                result["arms"][name] = _evaluate_arm(engine, initial, bounds, curve, horizon_s, lithium, charge_scale)
                if fingerprint(initial) != initial_id:
                    raise ValueError("One arm mutated common initial state")
                write_json(target, report)
            engine.sims.clear()
        if sources != {name: sha(root / name) for name in sources}:
            raise ValueError("Comparison sources changed during execution")
        report.update(status="complete", summary=summarize_meshes(report["mesh_results"], duty))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(target, report)
    return report
