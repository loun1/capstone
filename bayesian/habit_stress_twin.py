"""Short isothermal SPMe replay of paired virtual habits on two meshes."""
from pathlib import Path

import numpy as np

from simulation.rebuilt_lifetime import Engine, fingerprint, pybamm, scalar, sha, write_json
from .charging import _combine_audits, _lithium
from .driving_comparison import Guards, _drive, validate_comparison


HABITS = ("steady", "quick_launch", "quick_brake", "both")


def replay_habits(stress_report, output, *, progress=None):
    """Run the fixed example without recovery, reference tests, or extrapolation."""
    output = Path(output)
    target = output / "habit_stress_spme.json"
    if target.exists():
        raise FileExistsError("Habit SPMe replay output already exists")
    demo = stress_report["demo"]
    curves = demo["profiles"]
    if set(curves) != set(HABITS):
        raise ValueError("The demo must supply the four paired habit profiles")
    context = demo["context"]
    if context.get("source_kind") is None or "synthetic" not in context["source_kind"]:
        raise ValueError("Habit replay requires the virtual scenario context")
    for name in HABITS[1:]:
        validate_comparison({"steady": curves["steady"], name: curves[name]}, context)

    root = Path(__file__).resolve().parents[1]
    sources = {name: sha(root / name) for name in (
        "bayesian/habit_stress_twin.py", "bayesian/driving_comparison.py", "bayesian/charging.py",
        "simulation/rebuilt_lifetime.py", "configs/model.json", "configs/diagnostics.json",
    )}
    report = {"schema": "virtual_habit_stress_spme_v1", "status": "running", "context": context,
              "scope": "Short isothermal virtual-cell driving replay",
              "initial_soc_target": .8, "temperature_C": 25., "demo_seed": demo["episode_seed"],
              "initial_state_shared_within_mesh": True, "source_hashes": sources,
              "stress_report_sha256": sha(output / "habit_stress.json"),
              "mesh_results": [], "lifetime_extension_days": None}
    write_json(target, report)
    try:
        for mesh, rtol, atol in ((1, 1e-7, 1e-9), (2, 1e-8, 1e-10)):
            engine = Engine(mesh=mesh, rtol=rtol, atol=atol, temperature_K=298.15,
                            reference_output_only=False)
            initial = engine.sim("habit_initial", [pybamm.step.rest(duration=1)]).solve(
                initial_soc=.8, calc_esoh=False).last_state.copy()
            bounds = engine.bounds(initial)
            initial_soc = engine.soc(initial, bounds)
            if abs(initial_soc - .8) > .001:
                raise ValueError("Common initial SOC mismatch")
            state_id = fingerprint(initial)
            lithium = float(_lithium(initial).reshape(-1)[-1])
            charge_scale = scalar(initial, "Negative electrode capacity [A.h]")
            row = {"mesh": mesh, "rtol": rtol, "atol": atol, "initial_soc": initial_soc,
                   "initial_state_sha256": state_id, "arms": {}}
            report["mesh_results"].append(row)
            for name in HABITS:
                if progress:
                    progress(f"mesh{mesh} {name}")
                profile = np.asarray(curves[name], dtype=float)
                end, audit = _drive(engine, initial, profile, lithium, charge_scale)
                physical = _combine_audits([audit], Guards())
                physical["physical_validation_scope"] = "drive_only"
                if fingerprint(initial) != state_id:
                    raise ValueError("A replay mutated the shared initial state")
                row["arms"][name] = {"audit": audit, "physical": physical,
                                     "end_soc": engine.soc(end, bounds)}
                write_json(target, report)
            engine.sims.clear()
        if sources != {name: sha(root / name) for name in sources}:
            raise ValueError("Replay sources changed during execution")
        report["all_arms_pass_model_checks"] = all(arm["physical"]["physical_constraints_valid"]
                                                 for row in report["mesh_results"] for arm in row["arms"].values())
        report["status"] = "complete"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(target, report)
    return report
