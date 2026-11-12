"""Extend a preserved charging BO study and independently repeat its incumbent.

The original journal is never changed. Stability here covers the tested mesh,
solver and repeated deterministic evaluation, not real-cell model uncertainty.
"""
from __future__ import annotations

from dataclasses import asdict
import copy
import json
from pathlib import Path
import shutil
import time

from .charging import ChargeEvaluator, CURRENT_RANGE_A
from optimization.bayesian_charge import BayesianChargeOptimizer, SearchConfig
from simulation.rebuilt_lifetime import sha, write_json


def _feasible(rows):
    return sorted((r for r in rows if r["status"] == "complete" and r["feasible"]),
                  key=lambda r: (r["objective_pp"], r["suggestion"]["id"]))


def candidate_summary(row, deadline_s=3600.0):
    evaluation = row["evaluation"]
    times = [m["charge_time_s"] for m in evaluation.get("mesh_results", [])]
    return {"candidate_id": row["suggestion"]["id"],
            "current_A": row["suggestion"]["parameters"]["current_A"],
            "status": row["status"], "feasible": row["feasible"],
            "objective_pp": row["objective_pp"], "reason": row.get("reason"),
            "charge_time_s": evaluation.get("charge_time_s"),
            "deadline_margin_s": deadline_s-evaluation["charge_time_s"] if "charge_time_s" in evaluation else None,
            "mesh_charge_times_s": times, "numerical_error_pp": evaluation.get("numerical_error_pp"),
            "physical_constraints_valid": evaluation.get("physical_constraints_valid")}


def analyze_observations(rows, *, deadline_s=3600.0):
    feasible = _feasible(rows)
    if not feasible:
        return {"best_tested_feasible": None, "ranking_resolved": False,
                "numerically_indistinguishable_candidates": [], "candidate_table": [candidate_summary(r, deadline_s) for r in rows]}
    best = feasible[0]
    unresolved = [other for other in feasible[1:]
                  if other["objective_pp"] - best["objective_pp"] <=
                  other["evaluation"]["numerical_error_pp"] + best["evaluation"]["numerical_error_pp"]]
    return {"best_tested_feasible": candidate_summary(best, deadline_s),
            "ranking_resolved": len(feasible) >= 2 and not unresolved,
            "numerically_indistinguishable_candidates": [candidate_summary(r, deadline_s) for r in unresolved],
            "candidate_table": [candidate_summary(r, deadline_s) for r in rows]}


def compare_repeat(original, repeated, *, deadline_s=3600.0):
    """Compare validated observation rows, retaining failed/invalid repeats."""
    result = {"original": candidate_summary(original, deadline_s),
              "repeat": candidate_summary(repeated, deadline_s),
              "repeat_valid": repeated["status"] == "complete" and repeated["feasible"],
              "objective_repeat_consistent": False, "deadline_stable_on_tested_runs": False}
    if not result["repeat_valid"]:
        return result
    first, second = original["evaluation"], repeated["evaluation"]
    if original["suggestion"]["parameters"] != repeated["suggestion"]["parameters"]:
        raise ValueError("Independent repeat changed candidate parameters")
    change = abs(original["objective_pp"] - repeated["objective_pp"])
    numerical_bound = first["numerical_error_pp"] + second["numerical_error_pp"]
    time_change = abs(first["charge_time_s"] - second["charge_time_s"])
    all_times = [m["charge_time_s"] for evaluation in (first, second) for m in evaluation["mesh_results"]]
    worst_time = max(all_times)
    result.update(objective_absolute_change_pp=change, combined_mesh_error_pp=numerical_bound,
                  objective_repeat_consistent=change <= max(1e-9, numerical_bound),
                  charge_time_absolute_change_s=time_change,
                  worst_tested_mesh_charge_time_s=worst_time,
                  worst_tested_deadline_margin_s=deadline_s-worst_time,
                  deadline_stable_on_tested_runs=bool(original["feasible"] and worst_time <= deadline_s),
                  scope="same model and initial state; repeated solver and two tested meshes only")
    return result


def run_stability(prior_state, prior_report, output, *, budget=12, progress=None):
    if type(budget) is not int or not 5 <= budget <= 20:
        raise ValueError("Set an integer total search budget from5 to20")
    prior_state, prior_report, out = Path(prior_state), Path(prior_report), Path(output)
    if out.exists():
        raise FileExistsError("Use a new stability output directory")
    old = BayesianChargeOptimizer.load(prior_state)
    earlier = json.loads(prior_report.read_text(encoding="utf-8-sig"))
    if old.observations != earlier["observations"] or old.pending is not None:
        raise ValueError("Prior report and committed journal disagree or contain pending work")
    if len(old.observations) != 4:
        raise ValueError("This expansion requires the preserved initial four observations")
    source_hashes = {"prior_state": sha(prior_state), "prior_report": sha(prior_report)}
    root = Path(__file__).resolve().parents[1]
    code_hashes = {name: sha(root/name) for name in ("bayesian/charging_stability.py", "bayesian/charging.py",
                    "optimization/bayesian_charge.py", "simulation/rebuilt_lifetime.py",
                    "configs/model.json", "configs/diagnostics.json")}
    out.mkdir(parents=True, exist_ok=False)
    target = out/"charging_stability.json"
    report = {"schema": "charging_bo_stability_v1", "status": "initializing",
              "total_search_budget": budget, "independent_repeat_budget": 1,
              "prior_paths": {"state": str(prior_state.resolve()), "report": str(prior_report.resolve())},
              "original_source_hashes": source_hashes, "code_hashes": code_hashes,
              "scope": "bounded exploratory virtual-cell charging search and deterministic repeat",
              "recommendation": None, "global_optimum_established": False,
              "physical_model_compatibility_verified": False,
              "seed_stability_tested": False, "parameter_uncertainty_tested": False,
              "evaluation_timings": []}
    write_json(target, report)
    started = time.perf_counter()
    try:
        if progress:
            progress("Reconstructing the shared two-mesh evaluator and comparison contract")
        evaluator = ChargeEvaluator()
        config = SearchConfig(evaluator.contract, bounds={"current_A": CURRENT_RANGE_A}, initial_points=3)
        state_path = out/"charge_bo_state.json"
        # A byte-for-byte copy preserves pending IDs/RNG and leaves source intact.
        same_contract = json.dumps(asdict(config), sort_keys=True) == json.dumps(asdict(old.config), sort_keys=True)
        if same_contract:
            shutil.copyfile(prior_state, state_path)
            optimizer = BayesianChargeOptimizer.load(state_path, config)
            report.update(start_mode="copied_original_journal", reused_observation_count=4,
                          original_contract_reproduced=True)
        else:
            optimizer = BayesianChargeOptimizer.create(state_path, config)
            report.update(start_mode="fresh_deterministic_search", reused_observation_count=0,
                          original_contract_reproduced=False,
                          fresh_start_reason="Reconstructed contract/config differs; old physical observations not imported")
        report.update(status="searching", provenance=evaluator.provenance, contract=asdict(evaluator.contract),
                      prior_four_analysis=analyze_observations(old.observations))
        write_json(target, report)
        while len(optimizer.observations) < budget:
            suggestion = optimizer.ask()
            index = len(optimizer.observations)+1
            if progress:
                progress(f"Evaluating {index}/{budget}: {suggestion['parameters']['current_A']:.9f} A on both meshes")
            before = time.perf_counter()
            row = optimizer.evaluate_next(evaluator.evaluate)
            report["evaluation_timings"].append({"index": index, "current_A": suggestion["parameters"]["current_A"],
                                                  "elapsed_wall_s": time.perf_counter()-before, "status": row["status"]})
            report.update(observations=copy.deepcopy(optimizer.observations), search_summary=optimizer.summary(),
                          current_analysis=analyze_observations(optimizer.observations),
                          elapsed_wall_s=time.perf_counter()-started)
            write_json(target, report)
            if progress:
                progress(f"Completed {index}/{budget}: {row['status']}, feasible={row['feasible']}, loss={row['objective_pp']}")
        rows = copy.deepcopy(optimizer.observations)
        report.update(status="repeating_best", initial_four_current_analysis=analyze_observations(rows[:4]),
                      final_analysis=analyze_observations(rows), observations=rows)
        write_json(target, report)
        feasible = _feasible(rows)
        if not feasible:
            report.update(status="complete_no_feasible_candidate", repeat_check=None)
        else:
            best = feasible[0]
            if progress:
                progress(f"Independent fresh evaluator repeat of best tested feasible {best['suggestion']['parameters']['current_A']:.9f} A")
            # Release large state/solver caches before independent construction.
            del evaluator
            repeat_evaluator = ChargeEvaluator()
            repeat_started = time.perf_counter()
            if repeat_evaluator.contract != config.contract:
                report["repeat_check"] = {"repeat_valid": False, "reason": "Fresh evaluator contract did not reproduce"}
            else:
                try:
                    evaluation = repeat_evaluator.evaluate(copy.deepcopy(best["suggestion"]), config.contract)
                except Exception as error:
                    evaluation = {"status": "failed", "reason": f"{type(error).__name__}: {error}"}
                repeated = optimizer._observation(best["suggestion"], evaluation)
                report["independent_repeat_observation"] = repeated
                report["repeat_check"] = compare_repeat(best, repeated)
            report["repeat_wall_s"] = time.perf_counter()-repeat_started
            report["status"] = "complete"
        before = report["initial_four_current_analysis"]["best_tested_feasible"]
        after = report["final_analysis"]["best_tested_feasible"]
        report["improvement_vs_initial_four_pp"] = before["objective_pp"]-after["objective_pp"] if before and after else None
        report["same_best_current_as_initial_four"] = before["current_A"] == after["current_A"] if before and after else None
        if source_hashes != {"prior_state": sha(prior_state), "prior_report": sha(prior_report)}:
            raise ValueError("Original inputs changed during expansion")
        if code_hashes != {name: sha(root/name) for name in code_hashes}:
            raise ValueError("Source/config changed during expansion")
        report["original_sources_unchanged"] = True
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report["elapsed_wall_s"] = time.perf_counter()-started
        write_json(target, report)
    return report
