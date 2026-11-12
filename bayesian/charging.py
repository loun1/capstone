"""Explore one CC–CV current decision on the repository's virtual SPMe cell."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from simulation.rebuilt_lifetime import Engine, fingerprint, scalar, sha, write_json
from optimization.bayesian_charge import BayesianChargeOptimizer, ComparisonContract, SearchConfig
import numpy as np
import pybamm


CURRENT_RANGE_A = (3.0, 3.3)
BASELINE_CURRENT_A = 3.0
CHARGE_DEADLINE_S = 3600.0
DISCHARGE_START_S = 4500.0
HORIZON_S = 7200.0
TARGET_SOC = 0.8
TEMPERATURE_K = 298.15
MESHES = (1, 2)
TOLERANCES = ((1e-7, 1e-9), (1e-8, 1e-10))
MAX_MESH_DELTA_PP = 0.001
DELIVERED_AH_TOLERANCE = 0.005
PHASE_KEYS = {
    "voltage": ("Voltage [V]",),
    "current": ("Current [A]",),
    "electrolyte": ("Electrolyte concentration [mol.m-3]",),
    "solid": ("Negative particle stoichiometry", "Positive particle stoichiometry"),
    "porosity": ("Negative electrode porosity", "Separator porosity", "Positive electrode porosity"),
}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _values(piece, name):
    values = np.asarray(piece[name].entries, dtype=float)
    if not values.size or not np.isfinite(values).all():
        raise ValueError(f"Nonfinite or missing sampled field: {name}")
    return values


def _lithium(piece):
    return _values(piece, "Total lithium [mol]") + _values(
        piece, "Total lithium lost to side reactions [mol]"
    )


def _audit(piece, lithium_initial_mol, charge_scale_Ah):
    times = _values(piece, "Time [s]")
    signed_q = _values(piece, "Discharge capacity [A.h]")
    current = _values(piece, "Current [A]")
    if times.size < 2 or np.any(np.diff(times) <= 0):
        raise ValueError("Incomplete phase or invalid time samples")
    bounds = {}
    for label, fields in PHASE_KEYS.items():
        sampled = [_values(piece, name) for name in fields]
        bounds[label] = [min(float(v.min()) for v in sampled),
                         max(float(v.max()) for v in sampled)]
    residual = float(np.trapezoid(current, times) / 3600 - (signed_q[-1] - signed_q[0]))
    return {
        "duration_s": float(times[-1] - times[0]),
        "termination": str(piece.termination),
        "bounds": bounds,
        "charge_conservation_relative_error": abs(residual) / charge_scale_Ah,
        "lithium_conservation_relative_error": float(
            np.max(np.abs(_lithium(piece) - lithium_initial_mol)) / lithium_initial_mol
        ),
    }


def _run_reference(engine, state, lithium_initial, charge_scale):
    result, _ = engine.reference(state)
    sim = engine.sims.pop(("reference", engine.diag["rest_s"]))
    audits = [_audit(piece, lithium_initial, charge_scale) for piece in sim.solution.cycles[-1].steps]
    return result, audits


def _combine_audits(audits, contract):
    bounds = {
        label: (min(a["bounds"][label][0] for a in audits),
                max(a["bounds"][label][1] for a in audits))
        for label in PHASE_KEYS
    }
    vmin, vmax = bounds["voltage"]
    imax = max(abs(value) for value in bounds["current"])
    emin, emax = bounds["electrolyte"]
    smin, smax = bounds["solid"]
    pmin, pmax = bounds["porosity"]
    charge_error = max(a["charge_conservation_relative_error"] for a in audits)
    lithium_error = max(a["lithium_conservation_relative_error"] for a in audits)
    voltage_current_valid = (
        vmin >= contract.voltage_lower_limit_V - contract.voltage_tolerance_V
        and vmax <= contract.voltage_upper_limit_V + contract.voltage_tolerance_V
        and imax <= contract.current_abs_limit_A
    )
    local_state_valid = emin > 0 and 0 <= smin <= smax <= 1 and 0 < pmin <= pmax < 1
    conservation_valid = max(charge_error, lithium_error) <= contract.conservation_relative_error_limit
    return {
        "physical_evidence_id": _digest(audits),
        "physical_validation_scope": "full_horizon_and_rpt",
        "physical_constraints_valid": voltage_current_valid and local_state_valid and conservation_valid,
        "endpoint_valid": True,
        "voltage_current_valid": voltage_current_valid,
        "local_state_valid": local_state_valid,
        "conservation_valid": conservation_valid,
        "voltage_min_V": vmin, "voltage_max_V": vmax, "current_abs_max_A": imax,
        "electrolyte_concentration_min_molm3": emin, "electrolyte_concentration_max_molm3": emax,
        "solid_stoichiometry_min": smin, "solid_stoichiometry_max": smax,
        "porosity_min": pmin, "porosity_max": pmax,
        "charge_conservation_relative_error": charge_error,
        "lithium_conservation_relative_error": lithium_error,
    }


class ChargeEvaluator:
    """A fresh cell for every candidate with paired base/refined simulations."""

    def __init__(self):
        self.initial = []
        for mesh, (rtol, atol) in zip(MESHES, TOLERANCES):
            engine = Engine(mesh=mesh, rtol=rtol, atol=atol, temperature_K=TEMPERATURE_K,
                            reference_output_only=False)
            state = engine.start()
            bounds = engine.bounds(state)
            start_soc = engine.soc(state, bounds)
            charge_scale = scalar(state, "Negative electrode capacity [A.h]")
            lithium = float(_lithium(state).reshape(-1)[-1])
            q0, audits = _run_reference(engine, state, lithium, charge_scale)
            self.initial.append((engine, state, bounds, start_soc, charge_scale, lithium, q0, audits))

        root = Path(__file__).resolve().parents[1]
        engine, _, _, start_soc, _, _, q0, _ = self.initial[-1]
        model_config = root / "configs" / "model.json"
        self._baseline_charges = [self._charge_to_target(initial, BASELINE_CURRENT_A) for initial in self.initial]
        reference_delivered_Ah = self._baseline_charges[-1][3]
        self.provenance = {
            "model_family": "SPMe",
            "parameter_set": json.loads(model_config.read_text(encoding="utf-8"))["parameter_set"],
            "scope": "exploratory_virtual_cell",
            "temperature_K": TEMPERATURE_K,
            "pybamm_version": pybamm.__version__,
            "engine_sha256": sha(root / "simulation" / "rebuilt_lifetime.py"),
            "model_config_sha256": sha(model_config),
            "options": engine.options,
            "mesh_multipliers": list(MESHES),
            "solver_tolerances_rtol_atol": [list(item) for item in TOLERANCES],
            "initial_soc": start_soc,
            "delivered_charge_reference": "native CC–CV at the baseline current to the same SOC event",
            "delivered_charge_reference_current_A": BASELINE_CURRENT_A,
            "shared_discharge_start_s": DISCHARGE_START_S,
            "shared_discharge_current_A": 5.0,
            "rpt_start_soc_target": 0.2,
        }
        self.contract = ComparisonContract(
            initial_state_id=_digest([fingerprint(initial[1]) for initial in self.initial]),
            model_id=_digest(self.provenance),
            evaluator_version=sha(__file__),
            rpt_protocol_id=_digest(engine.diag),
            q0_Ah=q0["Qref_Ah"],
            horizon_s=HORIZON_S,
            target_soc=TARGET_SOC,
            required_delivered_Ah=reference_delivered_Ah,
            delivered_Ah_tolerance=DELIVERED_AH_TOLERANCE,
            charge_deadline_s=CHARGE_DEADLINE_S,
        )

    @staticmethod
    def _charge_to_target(initial, current_A):
        engine, state, bounds, _, charge_scale, lithium, _, _ = initial
        end, cc, dq, _ = engine.advance(state, "CC", DISCHARGE_START_S, current_A, bounds)
        audits = [_audit(cc, lithium, charge_scale)]
        charge_time = audits[-1]["duration_s"]
        if "Voltage > 4.2 [V]" in cc.termination:
            end, cv, cv_dq, _ = engine.advance(end, "CV", DISCHARGE_START_S-charge_time, current_A, bounds)
            audits.append(_audit(cv, lithium, charge_scale))
            charge_time += audits[-1]["duration_s"]
            dq += cv_dq
            termination = cv.termination
        else:
            termination = cc.termination
        achieved_soc = engine.soc(end, bounds)
        if "operating SOC" not in termination or not 0 <= achieved_soc <= 1:
            raise ValueError("CC–CV stopped before the required SOC event")
        return end, charge_time, achieved_soc, -dq, audits

    def _evaluate_mesh(self, mesh_index, current_A):
        initial = self.initial[mesh_index]
        engine, _, bounds, initial_soc, charge_scale, lithium, q0, initial_audits = initial
        if current_A == BASELINE_CURRENT_A:
            end, charge_time, achieved_soc, delivered_Ah, charge_audits = self._baseline_charges[mesh_index]
        else:
            end, charge_time, achieved_soc, delivered_Ah, charge_audits = self._charge_to_target(initial, current_A)
        audits = list(charge_audits)
        if abs(achieved_soc-self.contract.target_soc) > self.contract.soc_tolerance:
            raise ValueError("CC–CV missed the target SOC")
        if abs(delivered_Ah-self.contract.required_delivered_Ah) > self.contract.delivered_Ah_tolerance:
            raise ValueError("Delivered charge does not match the common task")
        rest_s = DISCHARGE_START_S - charge_time
        if rest_s > 1e-7:
            end, rest, _, _ = engine.advance(end, "REST", rest_s, current_A, bounds)
            audits.append(_audit(rest, lithium, charge_scale))
            if rest.termination != "final time":
                raise ValueError("Post-charge rest did not finish")
        end, discharge, _, _ = engine.advance(
            end, "DIS", HORIZON_S-DISCHARGE_START_S, current_A, bounds
        )
        audits.append(_audit(discharge, lithium, charge_scale))
        if "operating SOC" not in discharge.termination:
            raise ValueError("Shared discharge did not return to the RPT start SOC")
        rpt_start_soc = engine.soc(end, bounds)
        if abs(rpt_start_soc - 0.2) > self.contract.soc_tolerance:
            raise ValueError("Shared discharge missed the RPT start SOC")
        rest_s = HORIZON_S - sum(phase["duration_s"] for phase in audits)
        if rest_s <= 0:
            raise ValueError("Shared discharge exceeded the comparison horizon")
        end, rest, _, _ = engine.advance(end, "REST", rest_s, current_A, bounds)
        audits.append(_audit(rest, lithium, charge_scale))
        if rest.termination != "final time":
            raise ValueError("Rest before post-exposure RPT did not finish")
        rpt_start_soc_after_rest = engine.soc(end, bounds)
        if abs(rpt_start_soc_after_rest - initial_soc) > self.contract.soc_tolerance:
            raise ValueError("RPT start SOC drifted outside the common tolerance after rest")
        actual_horizon = sum(phase["duration_s"] for phase in audits)
        if abs(actual_horizon - HORIZON_S) > self.contract.horizon_tolerance_s:
            raise ValueError("Main trajectory did not cover the common comparison horizon")
        rpt, rpt_audits = _run_reference(engine, end, lithium, charge_scale)
        physical = _combine_audits(initial_audits + audits + rpt_audits, self.contract)
        return {
            "mesh": engine.mesh,
            "horizon_s": actual_horizon,
            "charge_time_s": charge_time,
            "achieved_soc": achieved_soc,
            "delivered_Ah": delivered_Ah,
            "rpt_start_soc_before_final_rest": rpt_start_soc,
            "rpt_start_soc_after_final_rest": rpt_start_soc_after_rest,
            "initial_rpt_capacity_Ah": q0["Qref_Ah"],
            "final_rpt_capacity_Ah": rpt["Qref_Ah"],
            "initial_rpt_audits": initial_audits,
            "main_audits": audits,
            "final_rpt_audits": rpt_audits,
            "physical": physical,
        }

    def evaluate(self, suggestion, contract):
        if contract != self.contract:
            raise ValueError("Study contract differs from the actual evaluator")
        current_A = suggestion["parameters"]["current_A"]
        if not np.isfinite(current_A) or not CURRENT_RANGE_A[0] <= current_A <= CURRENT_RANGE_A[1]:
            raise ValueError("Charging current is outside the exploratory domain")
        coarse, fine = [self._evaluate_mesh(index, current_A) for index in range(len(self.initial))]
        loss_pp = [100 * (r["initial_rpt_capacity_Ah"] - r["final_rpt_capacity_Ah"]) / contract.q0_Ah
                   for r in (coarse, fine)]
        mesh_delta = abs(loss_pp[1] - loss_pp[0])
        return {
            "status": "complete",
            "candidate_id": suggestion["id"],
            "parameters": dict(suggestion["parameters"]),
            **contract.evaluator_fields(),
            **{key: fine[key] for key in ("horizon_s", "charge_time_s", "achieved_soc", "delivered_Ah",
                                         "initial_rpt_capacity_Ah", "final_rpt_capacity_Ah")},
            **fine["physical"],
            "physical_constraints_valid": all(r["physical"]["physical_constraints_valid"]
                                              for r in (coarse, fine)),
            "numerically_resolved": mesh_delta <= MAX_MESH_DELTA_PP,
            "numerical_error_pp": mesh_delta,
            "numerical_evidence_id": _digest([coarse, fine]),
            "numerical_error_method": "absolute change in loss across two mesh and tolerance settings",
            "mesh_loss_pp": loss_pp,
            "mesh_results": [coarse, fine],
        }


def run_baseline(current_A=BASELINE_CURRENT_A):
    evaluator = ChargeEvaluator()
    params = {"current_A": float(current_A)}
    suggestion = {"id": _digest(params), "parameters": params}
    result = evaluator.evaluate(suggestion, evaluator.contract)
    return {
        "provenance": evaluator.provenance,
        "contract": asdict(evaluator.contract),
        "objective_pp": result["mesh_loss_pp"][-1],
        "meets_deadline": result["charge_time_s"] <= evaluator.contract.charge_deadline_s,
        "result": result,
    }


def run_search(output_dir, budget=5):
    """Run or resume a capped exploratory BO study without issuing a recommendation."""
    if isinstance(budget, bool) or not isinstance(budget, int) or not 1 <= budget <= 20:
        raise ValueError("Set an integer evaluation budget between 1 and 20")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    evaluator = ChargeEvaluator()
    config = SearchConfig(evaluator.contract, bounds={"current_A": CURRENT_RANGE_A}, initial_points=3)
    state = out / "charge_bo_state.json"
    if state.exists():
        optimizer = BayesianChargeOptimizer.load(state, config)
    else:
        optimizer = BayesianChargeOptimizer.create(state, config)
    while optimizer.summary()["evaluation_count"] < budget:
        optimizer.evaluate_next(evaluator.evaluate)
    rows = list(optimizer.observations)
    feasible = sorted((row for row in rows if row["feasible"]), key=lambda row: row["objective_pp"])
    resolved = len(feasible) >= 2 and all(
        other["objective_pp"] - feasible[0]["objective_pp"]
        > other["evaluation"]["numerical_error_pp"] + feasible[0]["evaluation"]["numerical_error_pp"]
        for other in feasible[1:]
    )
    report = {
        "provenance": evaluator.provenance,
        "contract": asdict(evaluator.contract),
        "summary": optimizer.summary(),
        "feasible_ranking_resolved_on_tested_meshes": resolved,
        "recommendation": None,
        "observations": rows,
    }
    write_json(out / "charging_search.json", report)
    return report
