"""Evaluate a GP correction to a lithium-loss proxy at RPT intervals.

The target is experimental voltage-curve DMA, and the frozen physical input is
calendar-SEI lithium loss on a capacity basis. Their definitions differ.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


FEATURES = (
    "prior_ageing_EFC",
    "interval_ageing_EFC",
    "interval_elapsed_days",
    "age_set_mean_surface_temperature_C",
)
SCALES = np.array([400., 20., 10., 10.])
Z95 = 1.959963984540054
MAX_SCALED_NEIGHBOR_DISTANCE = 2.


@dataclass(frozen=True)
class Intervals:
    cell: str
    nominal_temperature_C: float
    features: np.ndarray
    observed_delta_pp: np.ndarray
    physical_delta_pp: np.ndarray

    @property
    def exposure(self):
        """Interval exposure in hundreds of equivalent full cycles."""
        return self.features[:, 1] / 100.


def read_json(path, hashes, root):
    raw = path.read_bytes()
    hashes[path.relative_to(root).as_posix()] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode("utf-8-sig"))


def load_intervals(root: Path):
    """Bind four raw experimental summaries to their archived frozen SEI runs."""
    archive = root / "work/validation_proposal_20261103"
    latest = archive / "latest_spme_review_20261103"
    hashes = {}
    observations = read_json(latest / "ABCD_dma_temperature_v1/report.json", hashes, root)
    if not observations["all52_rows_formula_checks_pass"]:
        raise ValueError("Source DMA formula audit has not passed")
    audits = {row["cell"]: row for row in observations["cells"]}
    warm = latest / "calendar_SEI_SOH_v1"
    cold = latest / "calendar_SEI_AB_v1"
    warm_report = read_json(warm / "report.json", hashes, root)
    cold_report = read_json(cold / "report.json", hashes, root)
    if cold_report["C_refit_performed"]:
        raise ValueError("Cold replay must use the frozen C coefficient")
    cold_cells = {row["cell"]: row for row in cold_report["results"] if row["primary_scenario"]}
    frozen_multiplier = float(warm_report["C"]["diffusivity_multiplier"])
    cohorts = {}
    for cell in "CDAB":
        folder, model = (warm, warm_report[cell]) if cell in "CD" else (cold, cold_cells[cell])
        summary_root = root / "work/external_m50_benchmark" if cell in "CD" else archive / "additional_conditions_v2"
        source = summary_root / f"kirkaldy_expt2_cell{cell}_performance_summary.csv"
        with source.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        hashes[source.relative_to(root).as_posix()] = source_hash
        if source_hash != audits[cell]["source_sha256"]:
            raise ValueError(f"Unexpected published summary bytes for {cell}")
        ids = [int(row["Ageing Sets"]) for row in rows]
        points = model["points"]
        if ids != list(range(13)) or [point["rpt"] for point in points] != ids:
            raise ValueError(f"RPT identities do not match for {cell}")
        def column(name):
            return np.array([float(row[name]) for row in rows])

        capacity_dma_mAh = float(rows[0]["Cell Capacity [mA h]"])
        capacity_c10_Ah = float(rows[0]["C/10 Capacity [mA h]"]) / 1000.
        positive_capacity = column("PE Capacity [mA h]")
        offset = column("Electrode Offset [mA h]")
        dma_lli = (positive_capacity[0] - positive_capacity + offset - offset[0]) / capacity_dma_mAh
        if not np.allclose(dma_lli, column("LLI"), rtol=0, atol=1e-12):
            raise ValueError(f"Published LLI definition mismatch for {cell}")
        elapsed_days = np.array([point["elapsed_to_RPT_start_s"] for point in points]) / 86400.
        efc = column("Charge Throughput [A h]") / (2 * capacity_c10_Ah)
        temperature = [float(row["Age set av. temperature [°C]"]) for row in rows[1:]]
        features = np.column_stack((efc[:-1], np.diff(efc), np.diff(elapsed_days), temperature))
        physical = []
        initial_inventory = None
        for point in points:
            call = point["PDE_call"]
            saved = read_json(folder / f"RPT_{call:03d}/report.json", hashes, root)
            exposure_key = "elapsed_to_RPT_start_s" if cell in "CD" else "effective_exposure_298K_s"
            if (saved["call"] != call or saved["multiplier"] != frozen_multiplier
                    or not np.isclose(saved["predicted_Q_Ah"], point["predicted_Q_Ah"], rtol=0, atol=1e-11)
                    or not np.isclose(saved[exposure_key], point[exposure_key], rtol=0, atol=1e-6)):
                raise ValueError(f"Frozen physical archive binding failed for {cell} RPT{point['rpt']}")
            loss = float(saved["analytic_advance"]["total_lithium_loss_Ah"])
            budget = float(saved["aged_initial_Li_Ah"]) + loss
            if initial_inventory is None:
                initial_inventory = budget
            if not np.isclose(budget, initial_inventory, rtol=0, atol=1e-10):
                raise ValueError(f"Archived inventory accounting failed for {cell}")
            physical.append(100. * loss / (capacity_dma_mAh / 1000.))
        if (capacity_dma_mAh <= 0 or capacity_c10_Ah <= 0 or efc[0] != 0 or elapsed_days[0] != 0
                or not np.isfinite(features).all() or np.any(features[:, 1:3] <= 0)
                or not np.isfinite(dma_lli).all() or not np.isfinite(physical).all()
                or physical[0] != 0 or np.any(np.diff(physical) < 0)):
            raise ValueError(f"Invalid archived exposure or lithium loss for {cell}")
        cohorts[cell] = Intervals(cell, float(audits[cell]["nominal_temperature_C"]), features,
                                  100. * np.diff(dma_lli), np.diff(physical))
    return cohorts, hashes, frozen_multiplier


def rate_kernel(left, right):
    differences = (left[:, None, :] - right[None, :, :]) / SCALES
    distance = np.sqrt(3. * np.sum(differences ** 2, axis=-1))
    return (1. + distance) * np.exp(-distance)


def interval_noise_shape(exposure):
    # Adjacent targets share an RPT endpoint. Include both endpoint error and
    # interval-specific discrepancy with one C-trained noise scale.
    endpoint_difference = np.diff(np.eye(len(exposure) + 1), axis=0)
    return endpoint_difference @ endpoint_difference.T + np.diag(exposure ** 2)


def fit_residual_gp(train):
    """Fit a mean-zero GP in correction rate with interval-change observations."""
    response = train.observed_delta_pp - train.physical_delta_pp
    weight = train.exposure
    signal = rate_kernel(train.features, train.features) * np.outer(weight, weight)
    noise = interval_noise_shape(weight)

    def factor(scales):
        signal_sd, noise_sd = np.exp(scales)
        return cho_factor(signal_sd ** 2 * signal + noise_sd ** 2 * noise, lower=True)

    def loss(scales):
        cholesky = factor(scales)
        return .5 * response @ cho_solve(cholesky, response) + np.log(np.diag(cholesky[0])).sum()

    bounds = np.log([[.001, 10.], [.001, 5.]])
    fitted = minimize(loss, np.log([.3, .12]), bounds=bounds, method="L-BFGS-B")
    if not fitted.success:
        raise RuntimeError(f"C-only GP fit failed: {fitted.message}")
    cholesky = factor(fitted.x)
    return {
        "training": train,
        "signal_sd": float(np.exp(fitted.x[0])),
        "noise_sd": float(np.exp(fitted.x[1])),
        "cholesky": cholesky,
        "alpha": cho_solve(cholesky, response),
        "mean_correction_rate": float(response.sum() / weight.sum()),
        "negative_log_likelihood_without_constant": float(fitted.fun),
        "scale_at_bound": bool(np.isclose(fitted.x[:, None], bounds, atol=1e-5).any()),
    }


def predict_intervals(fitted, features, nominal_temperature_C):
    """Predict interval corrections and uncertainty from completed exposure."""
    features = np.asarray(features, dtype=float)
    if (features.ndim != 2 or features.shape[1] != len(FEATURES) or not len(features)
            or not np.isfinite(features).all() or np.any(features[:, 1:3] <= 0)
            or not np.isfinite(nominal_temperature_C)):
        raise ValueError("Prediction requires finite exposure rows and nominal ageing temperature")
    train = fitted["training"]
    weights = features[:, 1] / 100.
    signal_variance = fitted["signal_sd"] ** 2
    cross = signal_variance * rate_kernel(features, train.features) * np.outer(weights, train.exposure)
    prior = signal_variance * rate_kernel(features, features) * np.outer(weights, weights)
    latent_covariance = prior - cross @ cho_solve(fitted["cholesky"], cross.T)
    latent_covariance = .5 * (latent_covariance + latent_covariance.T)
    observed_covariance = latent_covariance + fitted["noise_sd"] ** 2 * interval_noise_shape(weights)
    correction = cross @ fitted["alpha"]
    distance = np.linalg.norm((features[:, None] - train.features[None, :]) / SCALES, axis=-1).min(axis=1)
    supported = (distance <= MAX_SCALED_NEIGHBOR_DISTANCE) & (
        nominal_temperature_C == train.nominal_temperature_C
    )
    return correction, latent_covariance, observed_covariance, distance, supported


def score(cohort, fitted, prediction):
    correction, latent_covariance, covariance, distance, supported = prediction
    means = {
        "frozen_calendar_sei_proxy": cohort.physical_delta_pp,
        "physics_plus_C_mean_residual_rate": cohort.physical_delta_pp + fitted["mean_correction_rate"] * cohort.exposure,
        "physics_plus_gp": cohort.physical_delta_pp + correction,
        "gp_when_supported_else_physics": cohort.physical_delta_pp + np.where(supported, correction, 0.),
    }
    interval_half_width = Z95 * np.sqrt(np.maximum(0., np.diag(covariance)))
    accumulation = np.tril(np.ones(covariance.shape))
    cumulative_half_width = Z95 * np.sqrt(np.maximum(0., np.diag(accumulation @ covariance @ accumulation.T)))
    error = means["physics_plus_gp"] - cohort.observed_delta_pp
    within = abs(error) <= interval_half_width
    cumulative_within = abs(np.cumsum(error)) <= cumulative_half_width
    statuses = np.where(~supported, "insufficient_training_support",
                        np.where(within, "within_model_interval", "review_dma_disagreement"))
    metrics = {}
    for name, mean in means.items():
        delta_error = mean - cohort.observed_delta_pp
        metrics[name] = {
            "interval_RMSE_pp": float(np.sqrt(np.mean(delta_error ** 2))),
            "cumulative_RMSE_pp": float(np.sqrt(np.mean(np.cumsum(delta_error) ** 2))),
            "final_cumulative_error_pp": float(delta_error.sum()),
        }
    return {
        "nominal_ageing_temperature_C": cohort.nominal_temperature_C,
        "metrics": metrics,
        "nominal_95_percent_predictive_interval_coverage": {
            "interval_count": int(within.sum()),
            "cumulative_count": int(cumulative_within.sum()),
            "total": len(error),
            "includes_exploratory_out_of_support_predictions": bool((~supported).any()),
        },
        "gp_latent_interval_covariance_pp_squared": latent_covariance.tolist(),
        "gp_observation_interval_covariance_pp_squared": covariance.tolist(),
        "intervals": [
            {
                "start_RPT": index, "end_RPT": index + 1,
                "features": dict(zip(FEATURES, features.tolist())),
                "observed_DMA_LLI_delta_pp": float(cohort.observed_delta_pp[index]),
                "observed_DMA_LLI_cumulative_pp": float(observed_cumulative),
                "predicted_delta_pp": {name: float(mean[index]) for name, mean in means.items()},
                "predicted_cumulative_pp": {name: float(np.sum(mean[:index + 1])) for name, mean in means.items()},
                "gp_residual_correction_delta_pp": float(correction[index]),
                "gp_interval_predictive_half_width_pp": float(interval_half_width[index]),
                "gp_cumulative_predictive_half_width_pp": float(cumulative_half_width[index]),
                "distance_to_nearest_scaled_training_row": float(distance[index]),
                "review_status": str(statuses[index]),
            }
            for index, (features, observed_cumulative) in enumerate(zip(cohort.features, np.cumsum(cohort.observed_delta_pp)))
        ],
    }


def evaluate(root: Path):
    cohorts, hashes, multiplier = load_intervals(root)
    fitted = fit_residual_gp(cohorts["C"])
    results = {}
    for cell, cohort in cohorts.items():
        prediction = predict_intervals(fitted, cohort.features, cohort.nominal_temperature_C)
        results[cell] = score(cohort, fitted, prediction)
    return {
        "status": "archived_RPT_DMA_proxy_comparison_complete",
        "data_source": "Kirkaldy Expt 2,2, 70–85% SOC (https://doi.org/10.5281/zenodo.10637534)",
        "split": {"fit": ["C"], "same_condition_heldout": ["D"], "cold_condition_heldout": ["A", "B"],
                  "note": "Retrospective benchmark. The archived calendar-SEI coefficient was previously fit to C capacity."},
        "target": "RPT-to-RPT change in author DMA LLI, in percentage points relative to the initial DMA cell capacity",
        "physics_proxy": "Archived analytic calendar-SEI lithium loss in Ah divided by the same initial DMA cell capacity. No simulated DMA observation operator was applied.",
        "update_equation": "next proxy = prior proxy + frozen physics interval change + (interval EFC / 100) * GP mean rate",
        "forecast_information": "Exposure completed before the next RPT target. Prior EFC, interval EFC/days, interval mean temperature and physical prior, with no heldout DMA label as an input.",
        "exposure_limits": "EFC uses published charge-plus-discharge ageing Ah divided by twice the initial C/10 capacity. Interval duration uses archived RPT timing and may include unknown operation.",
        "model": {
            "kernel": "exact Matérn 3/2",
            "feature_length_scales": dict(zip(FEATURES, SCALES.tolist())),
            "signal_sd_pp_per_100_EFC": fitted["signal_sd"],
            "noise_scale_pp": fitted["noise_sd"],
            "noise_model": "C-fitted scale for RPT endpoint differencing plus interval-specific residual noise",
            "mean_baseline_correction_pp_per_100_EFC": fitted["mean_correction_rate"],
            "negative_log_likelihood_without_constant": fitted["negative_log_likelihood_without_constant"],
            "gp_scale_at_bound": fitted["scale_at_bound"],
            "same_nominal_temperature_required_for_supported_prediction": True,
            "max_scaled_neighbor_distance": MAX_SCALED_NEIGHBOR_DISTANCE,
            "support_rule_scope": "Engineering rule; outside support retain frozen proxy in the gated output",
        },
        "frozen_calendar_SEI_diffusivity_multiplier": multiplier,
        "uncertainty_scope": "Pointwise model-based 95% predictive intervals, including RPT endpoint noise. No independent calibration of coverage or fault labels.",
        "review_scope": "RPT disagreement review only. No fault probability or validated anomaly classifier.",
        "production_update": "External RPT-interval proxy; SPMe/PyBaMM states and parameters remain outside this update.",
        "source_sha256": hashes,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cells": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capstone-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.capstone_root.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "report.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    for cell in ("D", "A", "B"):
        metrics = report["cells"][cell]["metrics"]
        print(f"{cell}: interval RMSE {metrics['frozen_calendar_sei_proxy']['interval_RMSE_pp']:.4f} -> "
              f"{metrics['physics_plus_gp']['interval_RMSE_pp']:.4f} pp (research GP)")


if __name__ == "__main__":
    main()
