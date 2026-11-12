"""Synthetic software contracts only; no measured or DFN validation is claimed."""
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from optimization.bayesian_charge import (
    AdmissionError, BayesianChargeOptimizer, ComparisonContract, SearchConfig)


def contract():
    return ComparisonContract(initial_state_id="SYNTHETIC-state-sha256",
        model_id="SYNTHETIC-model", evaluator_version="fixture-v1",
        rpt_protocol_id="SYNTHETIC-common-rpt", q0_Ah=5.0, horizon_s=7200.0,
        target_soc=.8, required_delivered_Ah=3.0, charge_deadline_s=3600.0)


def evaluation(suggestion, comparison, *, loss=.05, time_s=3500.0, error=1e-6):
    return {"status": "complete", "candidate_id": suggestion["id"],
        "parameters": suggestion["parameters"], **comparison.evaluator_fields(),
        "achieved_soc": comparison.target_soc,
        "delivered_Ah": comparison.required_delivered_Ah,
        "initial_rpt_capacity_Ah": comparison.q0_Ah,
        "final_rpt_capacity_Ah": comparison.q0_Ah*(1-loss/100),
        "charge_time_s": time_s, "numerically_resolved": True,
        "numerical_evidence_id": "SYNTHETIC-resolution-fixture", "numerical_error_pp": error,
        "physical_constraints_valid": True, "endpoint_valid": True,
        "voltage_current_valid": True, "local_state_valid": True, "conservation_valid": True,
        "physical_evidence_id": "SYNTHETIC-whole-horizon-and-rpt-fixture",
        "physical_validation_scope": "full_horizon_and_rpt",
        "voltage_min_V": 2.6, "voltage_max_V": 4.2, "current_abs_max_A": 4.,
        "electrolyte_concentration_min_molm3": 500.,
        "electrolyte_concentration_max_molm3": 1500.,
        "solid_stoichiometry_min": .01, "solid_stoichiometry_max": .95,
        "porosity_min": .1, "porosity_max": .4,
        "charge_conservation_relative_error": 1e-9,
        "lithium_conservation_relative_error": 1e-9}


def admission(config):
    return {"model_id": config.contract.model_id,
        "admission_fingerprint": config.admission_fingerprint, "calibrated_model_valid": True,
        "independent_data_valid": True, "numerical_resolution_verified": True,
        "calibration_evidence_id": "SYNTHETIC-calibration-fixture",
        "independent_validation_evidence_id": "SYNTHETIC-validation-fixture",
        "numerical_evidence_id": "SYNTHETIC-error-fixture",
        "validation_rmse_pp": .5, "validation_max_abs_pp": 1.0}


class BayesianChargeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)/"study.json"
        self.config = SearchConfig(contract(), candidate_pool_size=128)
        self.bo = BayesianChargeOptimizer.create(self.path, self.config)

    def add(self, loss=.05, time_s=3500.0, error=1e-6):
        point = self.bo.ask()
        return self.bo.tell(point["id"], evaluation(point, self.config.contract,
            loss=loss, time_s=time_s, error=error))

    def test_pending_survives_restart_and_tell_once(self):
        point = self.bo.ask()
        self.assertEqual(point, self.bo.ask())
        loaded = BayesianChargeOptimizer.load(self.path, self.config)
        self.assertEqual(point, loaded.ask())
        row = loaded.tell(point["id"], evaluation(point, self.config.contract))
        self.assertAlmostEqual(row["objective_pp"], .05)
        self.assertTrue(row["feasible"])
        with self.assertRaises(ValueError):
            loaded.tell(point["id"], evaluation(point, self.config.contract))
        self.assertEqual(BayesianChargeOptimizer.load(self.path).summary()["evaluation_count"], 1)

    def test_restart_reproduces_gp_sequence(self):
        other = BayesianChargeOptimizer.create(Path(self.tmp.name)/"other.json", self.config)
        for index in range(9):
            left, right = self.bo.ask(), other.ask()
            self.assertEqual(left, right)
            current = left["parameters"]["current_A"]
            loss = .01+(current-3.19)**2
            time_s = 11000/current
            self.bo.tell(left["id"], evaluation(left, self.config.contract, loss=loss, time_s=time_s))
            other.tell(right["id"], evaluation(right, self.config.contract, loss=loss, time_s=time_s))
            other = BayesianChargeOptimizer.load(other.path)
        self.assertEqual(self.bo.ask(), other.ask())
        self.assertGreater(self.bo.summary()["feasible_count"], 0)

    def test_deadline_infeasible_is_not_failure_or_best(self):
        row = self.add(loss=.001, time_s=3601)
        self.assertEqual(row["status"], "complete")
        self.assertFalse(row["feasible"])
        self.assertIsNotNone(row["objective_pp"])
        self.add(loss=.03)
        self.add(loss=.04)
        self.bo.update_admission(admission(self.config))
        best = self.bo.best()
        self.assertAlmostEqual(best["objective_pp"], .03)
        self.assertFalse(best["global_optimum_established"])

    def test_incomplete_charge_cannot_win(self):
        for changed in ({"achieved_soc": .7}, {"delivered_Ah": 2.0},
                        {"horizon_s": 7100}, {"q0_Ah": 4.9},
                        {"initial_rpt_capacity_Ah": 4.9},
                        {"initial_state_id": "different"},
                        {"rpt_protocol_id": "different"},
                        {"comparison_fingerprint": "different"},
                        {"numerically_resolved": False}):
            with self.subTest(changed=changed):
                point = self.bo.ask()
                record = evaluation(point, self.config.contract, loss=0)
                record.update(changed)
                row = self.bo.tell(point["id"], record)
                self.assertEqual(row["status"], "invalid")
                self.assertIsNone(row["objective_pp"])
                self.assertFalse(row["feasible"])

    def test_model_or_parameters_must_match(self):
        for key, value in (("model_id", "different"), ("candidate_id", "different"),
                           ("parameters", {"current_A": 3.299})):
            point = self.bo.ask()
            record = evaluation(point, self.config.contract)
            record[key] = value
            row = self.bo.tell(point["id"], record)
            self.assertEqual(row["status"], "invalid")

    def test_failures_and_timeouts_never_get_zero_loss(self):
        def fail(point, comparison):
            raise RuntimeError("synthetic solver failure")
        def timeout(point, comparison):
            raise TimeoutError("synthetic time cap")
        self.assertEqual(self.bo.evaluate_next(fail)["status"], "failed")
        self.assertEqual(self.bo.evaluate_next(timeout)["status"], "timeout")
        self.assertTrue(all(r["objective_pp"] is None for r in self.bo.observations))
        self.assertEqual(self.bo.summary()["complete_count"], 0)
        self.assertIsNotNone(self.bo.ask())

    def test_nonfinite_rejected_without_consuming_pending(self):
        point = self.bo.ask()
        record = evaluation(point, self.config.contract)
        record["final_rpt_capacity_Ah"] = float("nan")
        with self.assertRaises(ValueError):
            self.bo.tell(point["id"], record)
        self.assertEqual(point, BayesianChargeOptimizer.load(self.path).ask())

    def test_admission_and_error_limits(self):
        self.add(loss=.02)
        self.add(loss=.04)
        with self.assertRaises(AdmissionError):
            self.bo.best()
        evidence = admission(self.config)
        evidence["validation_rmse_pp"] = 1.01
        with self.assertRaises(AdmissionError):
            self.bo.update_admission(evidence)
        evidence = admission(self.config)
        evidence["numerical_resolution_verified"] = False
        with self.assertRaises(AdmissionError):
            self.bo.update_admission(evidence)
        self.bo.update_admission(admission(self.config))
        self.assertEqual(self.bo.best()["label"], "best_observed_feasible")
        self.assertEqual(BayesianChargeOptimizer.load(self.path).best(), self.bo.best())

    def test_ranking_unresolved_under_error_bounds(self):
        self.add(loss=.02, error=.001)
        self.add(loss=.021, error=.001)
        self.bo.update_admission(admission(self.config))
        with self.assertRaisesRegex(AdmissionError, "ranking is unresolved"):
            self.bo.best()

    def test_invalid_bounds_numbers_and_contract(self):
        for bounds in ({"current_A": (3.3, 3.0)}, {"current_A": (0, 3.0)},
                       {"current_A": (3.0, float("inf"))},
                       {"current_A": (True, 3.0)}):
            with self.subTest(bounds=bounds), self.assertRaises(ValueError):
                SearchConfig(contract(), bounds=bounds)
        with self.assertRaises(ValueError):
            replace(contract(), horizon_s=3000)
        with self.assertRaises(ValueError):
            replace(contract(), target_soc=80)
        with self.assertRaises(ValueError):
            replace(self.config, seed=True)
        with self.assertRaises(ValueError):
            replace(self.config, duplicate_tolerance=0)
        for changed in ({"validation_rmse_limit_pp": 1.01}, {"validation_max_abs_limit_pp": 2.01}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(self.config, **changed)

    def test_changed_contract_cannot_resume_and_score_tampering_detected(self):
        self.add()
        changed = replace(self.config, contract=replace(contract(), initial_state_id="other"))
        with self.assertRaises(ValueError):
            BayesianChargeOptimizer.load(self.path, changed)
        lines = self.path.read_text().splitlines()
        data = json.loads(lines[-1])
        data["payload"]["observations"][0]["objective_pp"] = 0
        lines[-1] = json.dumps(data)
        self.path.write_text("\n".join(lines)+"\n")
        with self.assertRaises(ValueError):
            BayesianChargeOptimizer.load(self.path)

    def test_no_duplicate_candidates_and_multidimensional_extension(self):
        bo = BayesianChargeOptimizer.create(Path(self.tmp.name)/"multi.json",
            replace(self.config, bounds={"current_A": (3., 3.3), "second_stage_A": (2., 3.)}))
        points = []
        for _ in range(12):
            suggestion = bo.ask()
            values = np.array(list(suggestion["parameters"].values()))
            self.assertTrue(all(np.max(np.abs(values-old)) > 1e-8 for old in points))
            points.append(values)
            bo.tell(suggestion["id"], {"status": "failed", "reason": "synthetic fixture"})
        self.assertEqual(bo.summary()["evaluation_count"], 12)

    def test_existing_state_not_overwritten(self):
        before = self.path.read_bytes()
        with self.assertRaises(FileExistsError):
            BayesianChargeOptimizer.create(self.path, self.config)
        self.assertEqual(before, self.path.read_bytes())

    def test_admission_cannot_transfer_to_other_contract_or_bounds(self):
        evidence = admission(self.config)
        changed = [replace(self.config, bounds={"current_A": (3., 4.)}),
                   replace(self.config, validation_rmse_limit_pp=.8)]
        for name, value in (("initial_state_id", "other-initial-state"),
                            ("evaluator_version", "other-evaluator"),
                            ("rpt_protocol_id", "other-rpt"),
                            ("horizon_s", 8000.), ("q0_Ah", 4.9),
                            ("target_soc", .9), ("required_delivered_Ah", 3.1),
                            ("charge_deadline_s", 3700.)):
            changed.append(replace(self.config, contract=replace(self.config.contract, **{name: value})))
        for index, config in enumerate(changed):
            with self.subTest(config=config):
                other = BayesianChargeOptimizer.create(Path(self.tmp.name)/f"other{index}.jsonl", config)
                with self.assertRaisesRegex(ValueError, "different study contract or search domain"):
                    other.update_admission(evidence)
                self.assertIsNone(other.admission)
        missing = dict(evidence)
        missing.pop("admission_fingerprint")
        with self.assertRaises(ValueError):
            self.bo.update_admission(missing)

    def test_torn_final_append_recovers_and_stale_writer_is_rejected(self):
        self.add(loss=.03)
        stale = BayesianChargeOptimizer.load(self.path)
        expected = self.bo.ask()
        with self.path.open("ab") as stream:
            stream.write(b'{"incomplete":')
        loaded = BayesianChargeOptimizer.load(self.path)
        self.assertEqual(loaded.ask(), expected)
        loaded.tell(expected["id"], evaluation(expected, self.config.contract, loss=.04))
        self.assertEqual(BayesianChargeOptimizer.load(self.path).summary()["evaluation_count"], 2)
        with self.assertRaises(RuntimeError):
            stale.ask()
        with self.assertRaisesRegex(RuntimeError, "reload"):
            stale.ask()

    def test_consumed_pending_cannot_be_reused_by_stale_evaluator(self):
        consumed = self.bo.ask()
        current = BayesianChargeOptimizer.load(self.path)
        current.tell(consumed["id"], evaluation(consumed, self.config.contract))
        next_candidate = current.ask()
        self.assertNotEqual(consumed["id"], next_candidate["id"])
        called = []
        def expensive_evaluator(point, comparison):
            called.append(point["id"])
            return evaluation(point, comparison)
        with self.assertRaisesRegex(RuntimeError, "reload"):
            self.bo.evaluate_next(expensive_evaluator)
        self.assertEqual(called, [])
        with self.assertRaisesRegex(RuntimeError, "reload"):
            self.bo.ask()
        self.assertEqual(BayesianChargeOptimizer.load(self.path).ask(), next_candidate)

    def test_rejected_physics_or_contradictory_extrema_never_become_feasible(self):
        modifications = [{name: False} for name in (
            "physical_constraints_valid", "endpoint_valid", "voltage_current_valid",
            "local_state_valid", "conservation_valid")]
        modifications += [{"physical_evidence_id": ""}, {"physical_validation_scope": "charge_only"},
            {"voltage_max_V": 7.}, {"voltage_min_V": 2.}, {"current_abs_max_A": 20.},
            {"voltage_max": 7.}, {"current_min_A": -20.},
            {"electrolyte_concentration_min_molm3": -1.},
            {"electrolyte_concentration_max_molm3": 100.},
            {"solid_stoichiometry_min": -.01}, {"solid_stoichiometry_max": 1.01},
            {"porosity_min": 0.}, {"porosity_max": 1.},
            {"charge_conservation_relative_error": .01},
            {"lithium_conservation_relative_error": .01}]
        for changed in modifications:
            with self.subTest(changed=changed):
                point = self.bo.ask()
                record = evaluation(point, self.config.contract, loss=0.)
                record.update(changed)
                row = self.bo.tell(point["id"], record)
                self.assertEqual(row["status"], "invalid")
                self.assertFalse(row["feasible"])
                self.assertIsNone(row["objective_pp"])
        self.bo.update_admission(admission(self.config))
        with self.assertRaises(AdmissionError):
            self.bo.best()


if __name__ == "__main__":
    unittest.main()
