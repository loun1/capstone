import copy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from bayesian.charging_stability import analyze_observations, compare_repeat, run_stability
from optimization.bayesian_charge import BayesianChargeOptimizer, SearchConfig
from optimization.test_bayesian_charge import contract, evaluation


class FakeEvaluator:
    """Deterministic software fixture, never actual battery evidence."""
    def __init__(self, comparison=None):
        self.contract = comparison or contract()
        self.provenance = {"scope": "synthetic test fixture"}

    def evaluate(self, suggestion, comparison):
        current = suggestion["parameters"]["current_A"]
        data = evaluation(suggestion, comparison, loss=.01+(current-3.12)**2, time_s=11000/current)
        data["mesh_results"] = [{"charge_time_s": data["charge_time_s"]}, {"charge_time_s": data["charge_time_s"]}]
        return data


class ChargingStabilityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prior = self.root/"prior_state.json"
        self.report = self.root/"prior_report.json"
        self.config = SearchConfig(contract(), bounds={"current_A": (3., 3.3)}, initial_points=3)
        self.optimizer = BayesianChargeOptimizer.create(self.prior, self.config)
        evaluator = FakeEvaluator()
        for _ in range(4):
            self.optimizer.evaluate_next(evaluator.evaluate)
        self.report.write_text(json.dumps({"observations": self.optimizer.observations}), encoding="utf-8")

    def test_resumes_copy_and_independently_repeats_without_mutating_original(self):
        original = self.prior.read_bytes()
        with patch("bayesian.charging_stability.ChargeEvaluator", side_effect=lambda: FakeEvaluator()) as factory:
            result = run_stability(self.prior, self.report, self.root/"extended", budget=5)
        self.assertEqual(factory.call_count, 2)
        self.assertEqual(result["start_mode"], "copied_original_journal")
        self.assertEqual(result["reused_observation_count"], 4)
        self.assertEqual(len(result["observations"]), 5)
        self.assertEqual(result["observations"][:4], self.optimizer.observations)
        self.assertTrue(result["repeat_check"]["objective_repeat_consistent"])
        self.assertTrue(result["repeat_check"]["deadline_stable_on_tested_runs"])
        self.assertEqual(self.prior.read_bytes(), original)
        self.assertIsNone(result["recommendation"])
        self.assertFalse(result["global_optimum_established"])

    def test_changed_contract_starts_fresh_without_importing_old_physics(self):
        changed = replace(contract(), model_id="different-test-model")
        with patch("bayesian.charging_stability.ChargeEvaluator", side_effect=lambda: FakeEvaluator(changed)):
            result = run_stability(self.prior, self.report, self.root/"fresh", budget=5)
        self.assertEqual(result["start_mode"], "fresh_deterministic_search")
        self.assertEqual(result["reused_observation_count"], 0)
        self.assertFalse(result["original_contract_reproduced"])
        self.assertTrue(all(r["evaluation"]["model_id"] == changed.model_id for r in result["observations"]))

    def test_disagreeing_report_is_rejected_before_physical_evaluation(self):
        self.report.write_text(json.dumps({"observations": []}), encoding="utf-8")
        with patch("bayesian.charging_stability.ChargeEvaluator") as factory:
            with self.assertRaisesRegex(ValueError, "disagree"):
                run_stability(self.prior, self.report, self.root/"bad", budget=5)
        factory.assert_not_called()

    def test_invalid_budget_and_existing_output_fail_before_evaluation(self):
        for budget in (True, 4, 21, 12.0):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                run_stability(self.prior, self.report, self.root/"bad", budget=budget)
        with self.assertRaises(FileExistsError):
            run_stability(self.prior, self.report, self.root, budget=5)

    def test_failed_or_deadline_infeasible_rows_cannot_be_best(self):
        rows = copy.deepcopy(self.optimizer.observations)
        rows[0].update(status="failed", feasible=False, objective_pp=None)
        result = analyze_observations(rows)
        self.assertIsNotNone(result["best_tested_feasible"])
        self.assertNotEqual(result["best_tested_feasible"]["candidate_id"], rows[0]["suggestion"]["id"])
        infeasible = next(r for r in rows if r["reason"] == "charge_deadline_exceeded")
        self.assertNotEqual(result["best_tested_feasible"]["candidate_id"], infeasible["suggestion"]["id"])

    def test_close_losses_within_mesh_error_do_not_claim_unique_ranking(self):
        rows = copy.deepcopy(self.optimizer.observations)
        for row in rows:
            row.update(status="complete", feasible=True, objective_pp=.02)
            row["evaluation"]["numerical_error_pp"] = 1e-5
        result = analyze_observations(rows)
        self.assertFalse(result["ranking_resolved"])
        self.assertEqual(len(result["numerically_indistinguishable_candidates"]), 3)

    def test_failed_repeat_remains_inconclusive(self):
        best = next(r for r in self.optimizer.observations if r["feasible"])
        failed = {"suggestion": best["suggestion"], "status": "failed", "feasible": False,
                  "objective_pp": None, "reason": "solver failed", "evaluation": {"status": "failed"}}
        result = compare_repeat(best, failed)
        self.assertFalse(result["repeat_valid"])
        self.assertFalse(result["objective_repeat_consistent"])

    def test_coarse_mesh_deadline_violation_is_not_hidden_by_fine_pass(self):
        best = copy.deepcopy(next(r for r in self.optimizer.observations if r["feasible"]))
        repeated = copy.deepcopy(best)
        repeated["evaluation"]["mesh_results"][0]["charge_time_s"] = 3600.1
        result = compare_repeat(best, repeated)
        self.assertTrue(result["objective_repeat_consistent"])
        self.assertFalse(result["deadline_stable_on_tested_runs"])
        repeated["objective_pp"] += .01
        self.assertFalse(compare_repeat(best, repeated)["objective_repeat_consistent"])


if __name__ == "__main__":
    unittest.main()
