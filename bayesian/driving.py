"""Predict a driver's future cell current from whole matched trips.

The reference and target are distinct vehicles with current already normalized
to the same model cell basis. Current is in amperes, positive for discharge.
Inputs must be continuous trip logs prepared by the caller. The forecast does
not convert pack telemetry or reconstruct missing trip intervals.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.linalg import solve


def _utc(value: datetime | str) -> datetime:
    stamp = datetime.fromisoformat(value) if isinstance(value, str) else value
    if stamp.utcoffset() is None:
        raise ValueError("Trip timestamps must include a UTC offset")
    return stamp.astimezone(timezone.utc)


@dataclass(frozen=True)
class Trip:
    vehicle_id: str
    driver_id: str
    trip_id: str
    starts_at: datetime
    route_id: str
    direction: str
    traffic: str
    distance_m: float
    cell_basis_id: str
    elapsed_s: np.ndarray
    cell_current_A: np.ndarray

    def __post_init__(self):
        for name in ("vehicle_id", "driver_id", "trip_id", "route_id", "direction",
                     "traffic", "cell_basis_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"Trip requires {name}")
        object.__setattr__(self, "starts_at", _utc(self.starts_at))
        if not np.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError("Trip distance must be positive and finite")
        elapsed = np.array(self.elapsed_s, dtype=float, copy=True)
        current = np.array(self.cell_current_A, dtype=float, copy=True)
        if (elapsed.ndim != 1 or len(elapsed) < 3 or current.shape != elapsed.shape
                or not np.isfinite(elapsed).all() or not np.isfinite(current).all()
                or elapsed[0] != 0 or not np.all(np.diff(elapsed) > 0)):
            raise ValueError("Trip requires finite samples on increasing elapsed seconds from zero")
        elapsed.setflags(write=False)
        current.setflags(write=False)
        object.__setattr__(self, "elapsed_s", elapsed)
        object.__setattr__(self, "cell_current_A", current)

    @property
    def completes_at(self) -> datetime:
        return self.starts_at + timedelta(seconds=float(self.elapsed_s[-1]))


def read_trips(path: str | Path, *, max_sample_gap_s: float = 10.0) -> list[Trip]:
    """Read a long-form cell-current CSV and reject gaps over the declared limit.

    The caller confirms the model cell basis, pack-to-cell conversion, and
    start-to-finish coverage. A gap limit is a check on the supplied timestamps.
    """
    if not np.isfinite(max_sample_gap_s) or max_sample_gap_s <= 0:
        raise ValueError("CSV sample-gap limit must be positive and finite")
    metadata = ("driver_id", "starts_at", "route_id", "direction", "traffic", "distance_m",
                "cell_basis_id")
    samples = ("elapsed_s", "cell_current_A")
    groups: dict[tuple[str, str], list[dict]] = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if {"cycle_id", "timestamp_s", "current_a"}.issubset(reader.fieldnames or ()):
            raise ValueError("Pack telemetry is not a GP trip history. Run 'python -m bayesian prepare-input' "
                             "with metadata and pack config first; driver/route/UTC trip context is still required.")
        required = {"vehicle_id", "trip_id", *metadata, *samples}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Trip CSV requires columns: {', '.join(sorted(required))}")
        for row in reader:
            if any(row[name] is None for name in required):
                raise ValueError(f"Incomplete trip sample at CSV line {reader.line_num}")
            groups.setdefault((row["vehicle_id"], row["trip_id"]), []).append(row)
    trips = []
    for (vehicle, trip_id), rows in groups.items():
        first = rows[0]
        if any(row[name] != first[name] for row in rows for name in metadata):
            raise ValueError(f"Inconsistent metadata for trip {vehicle}/{trip_id}")
        trip = Trip(
            vehicle_id=vehicle, driver_id=first["driver_id"], trip_id=trip_id,
            starts_at=_utc(first["starts_at"]), route_id=first["route_id"],
            direction=first["direction"], traffic=first["traffic"],
            distance_m=float(first["distance_m"]), cell_basis_id=first["cell_basis_id"],
            elapsed_s=np.array([float(row["elapsed_s"]) for row in rows]),
            cell_current_A=np.array([float(row["cell_current_A"]) for row in rows]),
        )
        if np.max(np.diff(trip.elapsed_s)) > max_sample_gap_s:
            raise ValueError(f"Sample gap exceeds {max_sample_gap_s} s for trip {vehicle}/{trip_id}")
        trips.append(trip)
    return trips


@dataclass(frozen=True)
class DrivingPrediction:
    reference_profile: np.ndarray
    expected_profile: np.ndarray
    sample_profiles: np.ndarray
    current_interval_90_A: np.ndarray
    validation: dict
    context: dict

    def to_dict(self) -> dict:
        return {
            "model": "matched_reference_plus_personal_functional_gp",
            "profile_columns": ["elapsed_s", "cell_current_A"],
            "current_sign": "positive_discharge_negative_regeneration",
            "context": self.context,
            "reference_profile": self.reference_profile.tolist(),
            "expected_profile": self.expected_profile.tolist(),
            "sample_profiles": self.sample_profiles.tolist(),
            "current_interval_90_A": self.current_interval_90_A.tolist(),
            "validation": self.validation,
        }


def _matrix(trips: list[Trip], progress: np.ndarray) -> np.ndarray:
    return np.stack([
        np.interp(progress, trip.elapsed_s / trip.elapsed_s[-1], trip.cell_current_A)
        for trip in trips
    ])


def _fit(target: np.ndarray, reference: np.ndarray, kernel: np.ndarray):
    baseline = reference.mean(axis=0)
    residuals = target - baseline
    residual_mean = residuals.mean(axis=0)
    centered = residuals - residual_mean
    empirical_cov = centered.T @ centered / (len(target) - 1)
    typical_variance = max(float(np.mean(np.diag(empirical_cov))), 1e-6)
    # Shrink the empirical trip covariance toward a smooth Matérn covariance.
    # Its amplitude describes variation between trips, separate from uncertainty
    # about the driver's shared residual trajectory.
    identity = np.eye(len(baseline))
    trip_cov = 0.5 * empirical_cov + 0.5 * typical_variance * kernel + 1e-9 * identity
    prior_scale = max(float(np.mean(residual_mean**2)), typical_variance, 1e-6)
    prior_cov = prior_scale * kernel
    training_cov = prior_cov + trip_cov / len(target)
    posterior_mean = baseline + prior_cov @ solve(training_cov, residual_mean, assume_a="pos")
    posterior_cov = prior_cov - prior_cov @ solve(training_cov, prior_cov, assume_a="pos")
    predictive_cov = posterior_cov + trip_cov
    return baseline, posterior_mean, (predictive_cov + predictive_cov.T) / 2


def predict_next(
    trips: list[Trip], *, reference_vehicle: str, target_vehicle: str,
    route_id: str, direction: str, traffic: str, distance_m: float, duration_s: float,
    samples: int = 8, seed: int = 20261106, as_of: datetime | str | None = None,
    holdout_trips: int = 2, progress_points: int = 81, distance_tolerance_fraction: float = 0.02,
    reference_driver: str | None = None, target_driver: str | None = None,
) -> DrivingPrediction:
    """Fit matched histories and backtest on chronological whole target trips.

    Normalized progress is elapsed time divided by trip duration. The future
    duration is supplied as a scenario, so no future timestamps or current enter
    the fit. Uncertainty describes cell-current variation on this progress grid.
    """
    if not reference_vehicle or not target_vehicle or reference_vehicle == target_vehicle:
        raise ValueError("Select distinct target and reference vehicles")
    if not all(np.isfinite(value) and value > 0 for value in (distance_m, duration_s)):
        raise ValueError("Forecast distance and duration must be positive and finite")
    if not 0 <= distance_tolerance_fraction <= 0.1:
        raise ValueError("Distance matching tolerance must be between 0 and 10%")
    if any(type(value) is not int for value in (samples, holdout_trips, progress_points)):
        raise ValueError("Sample, holdout and progress counts must be integers")
    if not (1 <= samples <= 100 and holdout_trips >= 1 and 11 <= progress_points <= 201):
        raise ValueError("Invalid sample, holdout or progress count")

    def match(trip: Trip) -> bool:
        return (
            trip.route_id == route_id and trip.direction == direction and trip.traffic == traffic
            and abs(trip.distance_m - distance_m) <= distance_m * distance_tolerance_fraction
        )

    cutoff = None if as_of is None else _utc(as_of)
    cohort = {target_vehicle: [], reference_vehicle: []}
    selected_drivers = {target_vehicle: target_driver, reference_vehicle: reference_driver}
    seen = set()
    for trip in trips:
        if trip.vehicle_id not in cohort or not match(trip):
            continue
        if cutoff is not None and trip.completes_at > cutoff:
            continue
        selected = selected_drivers[trip.vehicle_id]
        if selected is not None and trip.driver_id != selected:
            continue
        identity = (trip.vehicle_id, trip.trip_id)
        if identity in seen:
            raise ValueError(f"Duplicate trip: {trip.vehicle_id}/{trip.trip_id}")
        seen.add(identity)
        cohort[trip.vehicle_id].append(trip)
    for vehicle, group in cohort.items():
        if len({trip.driver_id for trip in group}) > 1:
            raise ValueError(f"Select one driver for shared vehicle {vehicle}")
        group.sort(key=lambda trip: trip.starts_at)
        if any(first.completes_at > second.starts_at for first, second in zip(group, group[1:])):
            raise ValueError("Matched trips for a vehicle overlap in time")
    personal, reference = cohort[target_vehicle], cohort[reference_vehicle]
    if len(personal) < 2 + holdout_trips or len(reference) < 2:
        raise ValueError("Need at least two earlier personal and reference trips plus personal holdouts")
    cell_basis = personal[0].cell_basis_id
    if any(trip.cell_basis_id != cell_basis for trip in personal + reference):
        raise ValueError("Matched histories must use the same model cell basis")

    progress = np.linspace(0, 1, progress_points)
    scaled_distance = np.sqrt(3) * abs(progress[:, None] - progress[None, :]) / 0.08
    kernel = (1 + scaled_distance) * np.exp(-scaled_distance)
    folds = []
    errors, baseline_errors, coverage = [], [], []
    for held_out in personal[-holdout_trips:]:
        previous = [trip for trip in personal if trip.completes_at <= held_out.starts_at]
        reference_then = [trip for trip in reference if trip.completes_at <= held_out.starts_at]
        if len(previous) < 2 or len(reference_then) < 2:
            raise ValueError(f"Insufficient chronological history for holdout {held_out.trip_id}")
        history = _matrix(previous, progress)
        _, mean, cov = _fit(history, _matrix(reference_then, progress), kernel)
        observed = _matrix([held_out], progress)[0]
        error = mean - observed
        baseline_error = history.mean(axis=0) - observed
        inside = abs(error) <= 1.64485362695 * np.sqrt(np.diag(cov))
        errors.append(error)
        baseline_errors.append(baseline_error)
        coverage.append(inside)
        folds.append({
            "test_trip_id": held_out.trip_id,
            "forecast_origin": held_out.starts_at.isoformat(),
            "personal_training_trip_ids": [trip.trip_id for trip in previous],
            "reference_training_trip_ids": [trip.trip_id for trip in reference_then],
            "gp_rmse_A": float(np.sqrt(np.mean(error**2))),
            "prior_personal_mean_rmse_A": float(np.sqrt(np.mean(baseline_error**2))),
            "predictive_interval_90_coverage": float(np.mean(inside)),
        })
    validation = {
        "scheme": "rolling_origin_whole_trip",
        "test_folds": folds,
        "gp_mae_A": float(np.mean(abs(np.array(errors)))),
        "gp_rmse_A": float(np.sqrt(np.mean(np.array(errors)**2))),
        "prior_personal_mean_mae_A": float(np.mean(abs(np.array(baseline_errors)))),
        "prior_personal_mean_rmse_A": float(np.sqrt(np.mean(np.array(baseline_errors)**2))),
        "predictive_interval_90_coverage": float(np.mean(coverage)),
    }

    baseline, mean, cov = _fit(_matrix(personal, progress), _matrix(reference, progress), kernel)
    time = duration_s * progress
    # Draw a whole path at once. Sampling grid points independently would
    # artificially increase current reversals and distort SPMe ageing.
    rng = np.random.default_rng(seed)
    draws = mean + rng.standard_normal((samples, progress_points)) @ np.linalg.cholesky(cov).T
    sample_profiles = np.stack([np.broadcast_to(time, draws.shape), draws], axis=-1)
    interval = 1.64485362695 * np.sqrt(np.diag(cov))
    return DrivingPrediction(
        np.column_stack([time, baseline]), np.column_stack([time, mean]), sample_profiles,
        np.column_stack([mean - interval, mean + interval]), validation,
        {
            "target_vehicle": target_vehicle, "reference_vehicle": reference_vehicle,
            "target_driver": personal[0].driver_id, "reference_driver": reference[0].driver_id,
            "cell_basis_id": cell_basis,
            "route_id": route_id, "direction": direction, "traffic": traffic,
            "distance_m": distance_m, "duration_s": duration_s,
            "distance_tolerance_fraction": distance_tolerance_fraction,
            "as_of": None if cutoff is None else cutoff.isoformat(),
            "personal_training_trip_ids": [trip.trip_id for trip in personal],
            "reference_training_trip_ids": [trip.trip_id for trip in reference],
        },
    )


def demo_trips() -> list[Trip]:
    """Create deterministic synthetic current histories for the local pilot."""
    rng = np.random.default_rng(581)
    start = datetime(2026, 10, 1, 8, tzinfo=timezone.utc)
    progress = np.linspace(0, 1, 121)
    common = (1.35 + 0.75 * np.sin(2 * np.pi * progress)**2
              - 2.15 * np.exp(-((progress - 0.74) / 0.044)**2))
    personal_shift = (0.52 * np.exp(-((progress - 0.28) / 0.13)**2)
                      - 0.25 * np.exp(-((progress - 0.58) / 0.09)**2))
    trips = []
    for vehicle, count, shift in (("reference", 8, 0), ("personal", 6, personal_shift)):
        for index in range(count):
            amp, offset = rng.normal(0, [0.11, 0.08])
            current = common + shift + amp * np.sin(2 * np.pi * progress) + offset
            current *= np.minimum(1, np.minimum(progress, 1 - progress) / 0.04)
            days = index if vehicle == "reference" else index + 2
            trips.append(Trip(
                vehicle_id=vehicle, driver_id=f"{vehicle}-driver", trip_id=f"{vehicle}-{index + 1:02d}",
                starts_at=start + timedelta(days=days), route_id="demo-city-loop",
                direction="clockwise", traffic="moderate", distance_m=3000.0,
                cell_basis_id="synthetic-m50t-single-cell",
                elapsed_s=(300 + index * 4) * progress, cell_current_A=current,
            ))
    return trips
