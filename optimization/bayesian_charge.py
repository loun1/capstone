"""Persistent constrained ask/tell search; this module never runs a battery model.

The objective is standardized RPT capacity loss in percentage points of a shared
Q0, measured after a common comparison horizon. A faster or shorter charge cannot
win by delivering less charge or stopping at a lower SOC. Numerical/software
checks do not supply the measured-model admission evidence required by best().
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Mapping
import warnings

import numpy as np
from scipy.special import ndtr
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern


SCHEMA = "capstone-charge-bo-v1"
STATUSES = {"complete", "failed", "timeout", "invalid"}


class AdmissionError(RuntimeError):
    """Measured calibration or numerical evidence is insufficient."""


def _number(value: Any, name: str, minimum: float | None = None) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"Invalid {name}")
    return result


def _label(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty identity")
    return value


def _digest(value: Any) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComparisonContract:
    initial_state_id: str
    model_id: str
    evaluator_version: str
    rpt_protocol_id: str
    q0_Ah: float
    horizon_s: float
    target_soc: float
    required_delivered_Ah: float
    charge_deadline_s: float
    soc_tolerance: float = 0.001
    delivered_Ah_tolerance: float = 0.001
    horizon_tolerance_s: float = 1e-6
    q0_tolerance_Ah: float = 1e-9
    voltage_lower_limit_V: float = 2.5
    voltage_upper_limit_V: float = 4.2
    current_abs_limit_A: float = 5.0
    voltage_tolerance_V: float = 1e-6
    conservation_relative_error_limit: float = 1e-6

    def __post_init__(self):
        for name in ("initial_state_id", "model_id", "evaluator_version", "rpt_protocol_id"):
            _label(getattr(self, name), name)
        for name in ("q0_Ah", "horizon_s", "required_delivered_Ah", "charge_deadline_s"):
            if _number(getattr(self, name), name, 0) <= 0:
                raise ValueError(f"{name} must be positive")
        target = _number(self.target_soc, "target_soc", 0)
        if target > 1 or self.charge_deadline_s > self.horizon_s:
            raise ValueError("SOC must use 0..1; deadline cannot exceed comparison horizon")
        for name in ("soc_tolerance", "delivered_Ah_tolerance", "horizon_tolerance_s", "q0_tolerance_Ah"):
            _number(getattr(self, name), name, 0)
        if self.soc_tolerance > .01 or self.delivered_Ah_tolerance >= self.required_delivered_Ah:
            raise ValueError("Comparison tolerances would permit a different charge task")
        lower = _number(self.voltage_lower_limit_V, "voltage_lower_limit_V", 0)
        upper = _number(self.voltage_upper_limit_V, "voltage_upper_limit_V", 0)
        if lower <= 0 or lower >= upper:
            raise ValueError("Physical voltage limits must be positive and increasing")
        if _number(self.current_abs_limit_A, "current_abs_limit_A", 0) <= 0:
            raise ValueError("Physical current limit must be positive")
        _number(self.voltage_tolerance_V, "voltage_tolerance_V", 0)
        if self.voltage_tolerance_V > .001:
            raise ValueError("Voltage event tolerance must not relax physical limits")
        if not 0 <= _number(self.conservation_relative_error_limit, "conservation_relative_error_limit") <= 1e-4:
            raise ValueError("Conservation tolerance must be at most 1e-4")

    @property
    def fingerprint(self) -> str:
        return _digest({"objective": "rpt_capacity_loss_pp", **asdict(self)})

    def evaluator_fields(self) -> dict:
        """Identity fields the evaluator must echo from its actual run evidence."""
        return {"comparison_fingerprint": self.fingerprint,
                **{name: getattr(self, name) for name in (
                    "initial_state_id", "model_id", "evaluator_version", "rpt_protocol_id",
                    "q0_Ah", "horizon_s", "target_soc", "required_delivered_Ah")}}


@dataclass(frozen=True)
class SearchConfig:
    contract: ComparisonContract
    bounds: Mapping[str, tuple[float, float]] = field(
        default_factory=lambda: {"current_A": (3.0, 3.3)})
    seed: int = 20261105
    initial_points: int = 4
    candidate_pool_size: int = 1024
    duplicate_tolerance: float = 1e-8
    improvement_margin_pp: float = 0.0
    validation_rmse_limit_pp: float = 1.0
    validation_max_abs_limit_pp: float = 2.0

    def __post_init__(self):
        if not isinstance(self.contract, ComparisonContract) or not self.bounds:
            raise ValueError("A comparison contract and nonempty bounds are required")
        for name, limits in self.bounds.items():
            _label(name, "parameter")
            if not isinstance(limits, (tuple, list)) or len(limits) != 2:
                raise ValueError("Bounds must contain [lower, upper]")
            low, high = (_number(x, name) for x in limits)
            if low >= high:
                raise ValueError("Bounds must be increasing")
            if name == "current_A" and low <= 0:
                raise ValueError("Charge current must be positive")
        for name, minimum in (("seed", 0), ("initial_points", 2), ("candidate_pool_size", 32)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid {name}")
        if not 0 < _number(self.duplicate_tolerance, "duplicate_tolerance") < .1:
            raise ValueError("Duplicate tolerance must be in (0, .1) normalized coordinates")
        for name in ("improvement_margin_pp", "validation_rmse_limit_pp", "validation_max_abs_limit_pp"):
            _number(getattr(self, name), name, 0)
        # Approved capstone 1-A acceptance limits may be tightened, not relaxed
        # by opening another study with a more permissive configuration.
        if self.validation_rmse_limit_pp > 1.0 or self.validation_max_abs_limit_pp > 2.0:
            raise ValueError("Approved validation limits are RMSE <=1%p and max error <=2%p")

    @property
    def admission_fingerprint(self) -> str:
        """Bind evidence to this exact contract, search domain and error limits.

        This deliberately includes every immutable configuration field. A change
        of state, Q0, RPT, horizon, evaluator, SOC/Ah task, deadline, bounds or
        admission limits requires evidence to be explicitly reviewed again.
        """
        return _digest(asdict(self))


def _config_from_dict(data: dict) -> SearchConfig:
    data = copy.deepcopy(data)
    data["contract"] = ComparisonContract(**data["contract"])
    return SearchConfig(**data)


class BayesianChargeOptimizer:
    """One writer, one pending evaluation; every accepted mutation is persisted.

    ask() is idempotent while an evaluation is pending. tell() consumes that ID
    exactly once, including invalid/failed/timeout results. No fabricated penalty
    or zero objective is inserted for those outcomes. Use a separate state file
    per writer; concurrent writer coordination is deliberately outside this API.
    """

    @classmethod
    def create(cls, state_path: str | Path, config: SearchConfig):
        obj = cls(state_path, config)
        if obj.path.exists():
            raise FileExistsError("State exists; use load() to resume")
        obj._save(exclusive=True)
        return obj

    def __init__(self, state_path: str | Path, config: SearchConfig):
        self.path = Path(state_path)
        # Copy nested bounds so caller mutation cannot alter an active study.
        self.config = _config_from_dict(asdict(config))
        self.names = sorted(self.config.bounds)
        self.lower = np.array([self.config.bounds[n][0] for n in self.names], dtype=float)
        self.upper = np.array([self.config.bounds[n][1] for n in self.names], dtype=float)
        self.rng = np.random.default_rng(self.config.seed)
        self.observations: list[dict] = []
        self.pending: dict | None = None
        self.admission: dict | None = None
        self._commit_hash: str | None = None
        self._write_failed = False

    @staticmethod
    def _read_journal(path: Path) -> tuple[dict, str, int]:
        """Read a checksummed append journal, ignoring only a torn final line."""
        latest, previous, valid_bytes = None, None, 0
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                try:
                    entry = json.loads(line)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise ValueError("Corrupt optimizer journal") from exc
                expected = _digest({"previous": previous, "payload": entry["payload"]})
                if entry.get("previous") != previous or entry.get("digest") != expected:
                    raise ValueError("Optimizer journal checksum or sequence mismatch")
                latest, previous = entry["payload"], expected
                valid_bytes += len(line)
        if latest is None:
            raise ValueError("Optimizer journal has no committed snapshot")
        return latest, previous, valid_bytes

    @classmethod
    def load(cls, state_path: str | Path, config: SearchConfig | None = None):
        data, commit_hash, _ = cls._read_journal(Path(state_path))
        if data.get("schema") != SCHEMA:
            raise ValueError("Unsupported optimizer state")
        stored = _config_from_dict(data["config"])
        if data.get("config_fingerprint") != _digest(asdict(stored)):
            raise ValueError("State configuration fingerprint mismatch")
        if config is not None and _digest(asdict(config)) != _digest(asdict(stored)):
            raise ValueError("Study contract or configuration changed; use a new study")
        obj = cls(state_path, stored)
        obj.rng.bit_generator.state = data["rng_state"]
        obj.observations = data["observations"]
        obj.pending = data["pending"]
        obj.admission = data["admission"]
        obj._commit_hash = commit_hash
        obj._check_state()
        return obj

    def _check_state(self):
        seen = set()
        for index, row in enumerate(self.observations):
            suggestion = row["suggestion"]
            self._validate_suggestion(suggestion, index)
            if suggestion["id"] in seen:
                raise ValueError("Duplicate persisted observation ID")
            seen.add(suggestion["id"])
            # Recompute objective and status rather than trusting edited scores.
            expected = self._observation(suggestion, row["evaluation"])
            if expected != row:
                raise ValueError("Persisted observation score or contract was altered")
        if self.pending is not None:
            self._validate_suggestion(self.pending, len(self.observations))
        if self.admission is not None:
            self._validate_admission(self.admission)

    def _ensure_usable(self):
        if self._write_failed:
            raise RuntimeError("Persistence failed; reload the study before continuing")
        if self._commit_hash is not None:
            _, latest_hash, _ = self._read_journal(self.path)
            if latest_hash != self._commit_hash:
                self._write_failed = True
                raise RuntimeError("Study changed in another writer; reload before using pending work")

    def _save(self, exclusive=False):
        self._ensure_usable()
        try:
            self._write_snapshot(exclusive)
        except Exception:
            # The filesystem may have committed before reporting an error.
            # Never reuse an in-memory pending/observation after ambiguity.
            self._write_failed = True
            raise

    def _write_snapshot(self, exclusive=False):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema": SCHEMA, "config": asdict(self.config),
                "config_fingerprint": _digest(asdict(self.config)),
                "rng_state": self.rng.bit_generator.state,
                "observations": self.observations, "pending": self.pending,
                "admission": self.admission}
        entry = {"previous": self._commit_hash, "payload": data}
        entry["digest"] = _digest(entry)
        text = json.dumps(entry, sort_keys=True, separators=(",", ":"), allow_nan=False)+"\n"
        if exclusive:
            with self.path.open("x", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
        else:
            _, latest_hash, valid_bytes = self._read_journal(self.path)
            if latest_hash != self._commit_hash:
                raise RuntimeError("Another writer changed the study; reload before continuing")
            # Append avoids replacing open files on Windows. A partial final
            # record from a crash is discarded before the next complete commit.
            with self.path.open("r+b") as stream:
                stream.seek(valid_bytes)
                stream.truncate()
                stream.write(text.encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())
        self._commit_hash = entry["digest"]

    def _unit(self, parameters: Mapping[str, float]) -> np.ndarray:
        if set(parameters) != set(self.names):
            raise ValueError("Candidate parameters differ from configured dimensions")
        raw = np.array([_number(parameters[n], n) for n in self.names])
        if np.any(raw < self.lower) or np.any(raw > self.upper):
            raise ValueError("Candidate outside search bounds")
        return (raw - self.lower) / (self.upper - self.lower)

    def _suggestion(self, point: np.ndarray, index: int) -> dict:
        parameters = {n: float(v) for n, v in zip(self.names, self.lower + point*(self.upper-self.lower))}
        body = {"index": index, "parameters": parameters,
                "comparison_fingerprint": self.config.contract.fingerprint}
        return {"id": _digest(body)[:24], **body}

    def _validate_suggestion(self, suggestion: dict, index: int):
        point = self._unit(suggestion["parameters"])
        if suggestion != self._suggestion(point, index):
            raise ValueError("Persisted suggestion identity mismatch")

    def _fresh_mask(self, pool: np.ndarray) -> np.ndarray:
        previous = [self._unit(r["suggestion"]["parameters"]) for r in self.observations]
        if not previous:
            return np.ones(len(pool), dtype=bool)
        distance = np.max(np.abs(pool[:, None, :] - np.array(previous)[None, :, :]), axis=2)
        return np.all(distance > self.config.duplicate_tolerance, axis=1)

    def ask(self) -> dict:
        self._ensure_usable()
        if self.pending is not None:
            return copy.deepcopy(self.pending)
        dimension = len(self.names)
        # Include deterministic midpoint/endpoints, then use persistent RNG.
        anchors = [np.full(dimension, .5), np.zeros(dimension), np.ones(dimension)]
        pool = np.vstack(anchors + [self.rng.random((self.config.candidate_pool_size, dimension))])
        pool = pool[self._fresh_mask(pool)]
        if not len(pool):
            raise RuntimeError("Search space exhausted at duplicate tolerance")
        valid = [r for r in self.observations if r["status"] == "complete"]
        if len(valid) < self.config.initial_points:
            if not self.observations:
                point = pool[0]
            else:
                old = np.array([self._unit(r["suggestion"]["parameters"]) for r in self.observations])
                distances = np.min(np.linalg.norm(pool[:, None, :] - old[None, :, :], axis=2), axis=1)
                point = pool[int(np.argmax(distances))]
        else:
            point = pool[int(np.argmax(self._acquisition(valid, pool)))]
        self.pending = self._suggestion(point, len(self.observations))
        self._save()
        return copy.deepcopy(self.pending)

    def _predict(self, x: np.ndarray, values: np.ndarray, noise: np.ndarray,
                 pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        center = float(np.mean(values))
        scale = max(float(np.std(values)), 1e-6)
        kernel = ConstantKernel(1.0, (1e-3, 1e3))*Matern(
            length_scale=np.full(x.shape[1], .3), length_scale_bounds=(.02, 10), nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel,
            alpha=np.maximum((noise/scale)**2, 1e-10), normalize_y=False,
            n_restarts_optimizer=0, random_state=self.config.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x, (values-center)/scale)
        mean, sigma = gp.predict(pool, return_std=True)
        return mean*scale+center, np.maximum(sigma*scale, 1e-12)

    def _acquisition(self, valid: list[dict], pool: np.ndarray) -> np.ndarray:
        x = np.array([self._unit(r["suggestion"]["parameters"]) for r in valid])
        y = np.array([r["objective_pp"] for r in valid])
        error = np.array([r["evaluation"]["numerical_error_pp"] for r in valid])
        mean, sigma = self._predict(x, y, error, pool)
        times = np.array([r["evaluation"]["charge_time_s"] for r in valid])
        time_mean, time_sigma = self._predict(x, times, np.full(len(times), 1e-6), pool)
        probability_feasible = ndtr((self.config.contract.charge_deadline_s-time_mean)/time_sigma)
        feasible = [r["objective_pp"] for r in valid if r["feasible"]]
        if not feasible:
            # With no feasible incumbent, first seek deadline feasibility.
            return probability_feasible + .01*sigma/max(float(np.max(sigma)), 1e-12)
        improvement = min(feasible)-mean-self.config.improvement_margin_pp
        z = improvement/sigma
        ei = improvement*ndtr(z)+sigma*np.exp(-.5*z*z)/math.sqrt(2*math.pi)
        return np.maximum(ei, 0)*probability_feasible

    def _observation(self, suggestion: dict, evaluation: dict) -> dict:
        if not isinstance(evaluation, dict):
            raise ValueError("Evaluator must return a JSON dictionary")
        # Reject nonfinite data even in failed results; persist JSON, not NaN.
        json.dumps(evaluation, allow_nan=False)
        status = evaluation.get("status")
        if status not in STATUSES:
            raise ValueError("Evaluator status must be complete, failed, timeout, or invalid")
        row = {"suggestion": copy.deepcopy(suggestion), "evaluation": copy.deepcopy(evaluation),
               "status": status, "objective_pp": None, "feasible": False, "reason": None}
        if status != "complete":
            row["reason"] = str(evaluation.get("reason", status))
            return row
        c = self.config.contract
        try:
            if evaluation.get("candidate_id") != suggestion["id"]:
                raise ValueError("Evaluator candidate identity mismatch")
            if evaluation.get("parameters") != suggestion["parameters"]:
                raise ValueError("Evaluator parameters mismatch")
            for name, expected in c.evaluator_fields().items():
                actual = evaluation.get(name)
                if isinstance(expected, str):
                    if actual != expected:
                        raise ValueError(f"Comparison contract mismatch: {name}")
                else:
                    tolerance = {"q0_Ah": c.q0_tolerance_Ah, "horizon_s": c.horizon_tolerance_s}.get(name, 1e-12)
                    if abs(_number(actual, name)-expected) > tolerance:
                        raise ValueError(f"Comparison contract mismatch: {name}")
            self._validate_physical_evidence(evaluation)
            achieved = _number(evaluation.get("achieved_soc"), "achieved_soc", 0)
            if achieved > 1 or abs(achieved-c.target_soc) > c.soc_tolerance:
                raise ValueError("Required target SOC was not achieved")
            if abs(_number(evaluation.get("delivered_Ah"), "delivered_Ah", 0)-c.required_delivered_Ah) > c.delivered_Ah_tolerance:
                raise ValueError("Delivered charge differs from the common comparison task")
            before = _number(evaluation.get("initial_rpt_capacity_Ah"), "initial_rpt_capacity_Ah", 0)
            after = _number(evaluation.get("final_rpt_capacity_Ah"), "final_rpt_capacity_Ah", 0)
            if abs(before-c.q0_Ah) > c.q0_tolerance_Ah or after <= 0:
                raise ValueError("RPT initial capacity/Q0 mismatch or nonpositive final capacity")
            if evaluation.get("numerically_resolved") is not True:
                raise ValueError("Evaluation has no numerical-resolution evidence")
            _label(evaluation.get("numerical_evidence_id"), "numerical_evidence_id")
            _number(evaluation.get("numerical_error_pp"), "numerical_error_pp", 0)
            charge_time = _number(evaluation.get("charge_time_s"), "charge_time_s", 0)
            if charge_time > c.horizon_s or charge_time == 0:
                raise ValueError("Charge time is incompatible with comparison horizon")
            objective = 100*(before-after)/c.q0_Ah
            # Loss smaller than zero beyond numerical error indicates a bad RPT comparison.
            if objective < -evaluation["numerical_error_pp"]:
                raise ValueError("Apparent capacity gain exceeds the numerical error bound")
            row.update(objective_pp=objective, feasible=charge_time <= c.charge_deadline_s)
            if not row["feasible"]:
                row["reason"] = "charge_deadline_exceeded"
        except ValueError as exc:
            row.update(status="invalid", reason=str(exc))
        return row

    def _validate_physical_evidence(self, evaluation: dict):
        """Require actual whole-horizon/RPT safety evidence, not a success code.

        Local concentration/stoichiometry/porosity extrema must cover every
        spatial degree of freedom and saved solver stage used by the evaluator.
        Conservation errors are maximum relative residuals over the complete
        comparison and RPT; a flag cannot override contradictory numeric data.
        """
        for name in ("physical_constraints_valid", "endpoint_valid", "voltage_current_valid",
                     "local_state_valid", "conservation_valid"):
            if evaluation.get(name) is not True:
                raise ValueError(f"Physical validation missing or rejected: {name}")
        _label(evaluation.get("physical_evidence_id"), "physical_evidence_id")
        if evaluation.get("physical_validation_scope") != "full_horizon_and_rpt":
            raise ValueError("Physical evidence must cover the full comparison horizon and RPT")
        c = self.config.contract
        low = _number(evaluation.get("voltage_min_V"), "voltage_min_V")
        high = _number(evaluation.get("voltage_max_V"), "voltage_max_V")
        if low > high or low < c.voltage_lower_limit_V-c.voltage_tolerance_V or high > c.voltage_upper_limit_V+c.voltage_tolerance_V:
            raise ValueError("Voltage extrema violate configured physical limits")
        current = _number(evaluation.get("current_abs_max_A"), "current_abs_max_A", 0)
        if current > c.current_abs_limit_A:
            raise ValueError("Current extremum violates configured physical limit")
        # Reject contradictory commonly supplied scalar extrema as well.
        for alias in ("voltage_max", "max_voltage_V"):
            if alias in evaluation and _number(evaluation[alias], alias) > c.voltage_upper_limit_V+c.voltage_tolerance_V:
                raise ValueError("Additional voltage maximum contradicts physical validity")
        for name in ("current_min_A", "current_max_A"):
            if name in evaluation and abs(_number(evaluation[name], name)) > c.current_abs_limit_A:
                raise ValueError("Additional current extremum contradicts physical validity")
        electrolyte_min = _number(evaluation.get("electrolyte_concentration_min_molm3"), "electrolyte_concentration_min_molm3")
        electrolyte_max = _number(evaluation.get("electrolyte_concentration_max_molm3"), "electrolyte_concentration_max_molm3")
        if electrolyte_min <= 0 or electrolyte_min > electrolyte_max:
            raise ValueError("Local electrolyte concentration is nonphysical")
        sto_min = _number(evaluation.get("solid_stoichiometry_min"), "solid_stoichiometry_min")
        sto_max = _number(evaluation.get("solid_stoichiometry_max"), "solid_stoichiometry_max")
        if not 0 <= sto_min <= sto_max <= 1:
            raise ValueError("Local solid concentration/stoichiometry is nonphysical")
        porosity_min = _number(evaluation.get("porosity_min"), "porosity_min")
        porosity_max = _number(evaluation.get("porosity_max"), "porosity_max")
        if not 0 < porosity_min <= porosity_max < 1:
            raise ValueError("Local porosity is nonphysical")
        for name in ("charge_conservation_relative_error", "lithium_conservation_relative_error"):
            if _number(evaluation.get(name), name, 0) > c.conservation_relative_error_limit:
                raise ValueError(f"Conservation check exceeds configured limit: {name}")

    def tell(self, candidate_id: str, evaluation: dict) -> dict:
        self._ensure_usable()
        if self.pending is None or candidate_id != self.pending["id"]:
            raise ValueError("tell() requires the current pending candidate ID exactly once")
        row = self._observation(self.pending, evaluation)
        self.observations.append(row)
        self.pending = None
        self._save()
        return copy.deepcopy(row)

    def evaluate_next(self, evaluator: Callable[[dict, ComparisonContract], dict]) -> dict:
        suggestion = self.ask()
        try:
            evaluation = evaluator(copy.deepcopy(suggestion), self.config.contract)
        except TimeoutError as exc:
            evaluation = {"status": "timeout", "reason": str(exc)}
        except Exception as exc:
            evaluation = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        return self.tell(suggestion["id"], evaluation)

    def _validate_admission(self, evidence: dict):
        if evidence.get("admission_fingerprint") != self.config.admission_fingerprint:
            raise ValueError("Admission evidence applies to a different study contract or search domain")
        if evidence.get("model_id") != self.config.contract.model_id:
            raise ValueError("Admission evidence applies to a different model")
        for name in ("calibrated_model_valid", "independent_data_valid", "numerical_resolution_verified"):
            if evidence.get(name) is not True:
                raise AdmissionError(f"Missing admission: {name}")
        for name in ("calibration_evidence_id", "independent_validation_evidence_id", "numerical_evidence_id"):
            _label(evidence.get(name), name)
        rmse = _number(evidence.get("validation_rmse_pp"), "validation_rmse_pp", 0)
        maximum = _number(evidence.get("validation_max_abs_pp"), "validation_max_abs_pp", 0)
        if rmse > self.config.validation_rmse_limit_pp or maximum > self.config.validation_max_abs_limit_pp:
            raise AdmissionError("Measured validation errors exceed configured admission limits")

    def update_admission(self, evidence: dict):
        """Supply reviewed real evidence; tests never invoke this automatically."""
        self._ensure_usable()
        self._validate_admission(evidence)
        self.admission = copy.deepcopy(evidence)
        self._save()

    def best(self) -> dict:
        self._ensure_usable()
        if self.admission is None:
            raise AdmissionError("Exploratory study: calibrated measured-data admission is missing")
        self._validate_admission(self.admission)
        feasible = sorted((r for r in self.observations if r["status"] == "complete" and r["feasible"]),
                          key=lambda r: (r["objective_pp"], r["suggestion"]["id"]))
        if len(feasible) < 2:
            raise AdmissionError("At least two feasible resolved evaluations are required")
        selected = feasible[0]
        for other in feasible[1:]:
            if other["objective_pp"]-selected["objective_pp"] <= (
                    other["evaluation"]["numerical_error_pp"]+selected["evaluation"]["numerical_error_pp"]):
                raise AdmissionError("Observed ranking is unresolved at supplied numerical-error bounds")
        return {"label": "best_observed_feasible", "global_optimum_established": False,
                "parameters": copy.deepcopy(selected["suggestion"]["parameters"]),
                "objective_pp": selected["objective_pp"], "candidate_id": selected["suggestion"]["id"],
                "comparison_fingerprint": self.config.contract.fingerprint,
                "admission_fingerprint": self.config.admission_fingerprint,
                "admission": copy.deepcopy(self.admission)}

    def summary(self) -> dict:
        self._ensure_usable()
        return {"schema": SCHEMA, "comparison_fingerprint": self.config.contract.fingerprint,
                "admission_fingerprint": self.config.admission_fingerprint,
                "evaluation_count": len(self.observations),
                "complete_count": sum(r["status"] == "complete" for r in self.observations),
                "feasible_count": sum(r["status"] == "complete" and r["feasible"] for r in self.observations),
                "invalid_failed_timeout_count": sum(r["status"] != "complete" for r in self.observations),
                "pending": copy.deepcopy(self.pending), "measured_admission_supplied": self.admission is not None,
                "global_optimum_established": False,
                "mode": "admitted_search" if self.admission else "exploratory_search"}
