"""Fit two matched full-current forecasts for the fixed virtual habit route.

Both fit the mean of paired train current differences. The train mean steady
curve is then added back. Future environments supply scoring curves only, not
features, so the prediction targets the mean waveform for the given ramps.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Real

import numpy as np
from scipy.linalg import solve

from . import habit_stress as habit


ELAPSED_S = habit.scenario.ELAPSED_S
TRAIN_SEEDS = habit.TRAIN_SEEDS
TRAIN_RAMPS = habit.TRAIN_RAMPS
VALIDATION_SEEDS = tuple(range(2101, 2133))
TEST_SEEDS = tuple(range(2301, 2333))
VALIDATION_RAMPS = ((9., 9.), (9., 14.), (11., 12.), (13., 10.), (15., 9.), (15., 14.))
TEST_RAMPS = tuple(product((8.5, 10.5, 13.5, 15.5), (8.5, 10.5, 12.5, 14.5)))
RAMP_BOUNDS_S = ((8., 16.), (8., 15.))
MODEL_BOUNDS = {
    'gp': {'length_scale': (.12, 2.), 'log10_ridge': (-8., -.3)},
    'quadratic': {'log10_ridge': (-8., -.3)},
}
DEFAULT_PARAMS = {
    'gp': {'length_scale': .5, 'log10_ridge': -5.},
    'quadratic': {'log10_ridge': -5.},
}
CONTEXT = {'route_id': habit.ROUTE, 'direction': 'forward', 'traffic': habit.TRAFFIC,
           'distance_m': sum(habit.scenario.ROUTES[habit.ROUTE][0]), 'cell_basis_id': habit.scenario.CELL_BASIS_ID,
           'source_kind': 'synthetic_habit_waveform'}


@dataclass(frozen=True)
class WaveformSplit:
    name: str
    seeds: tuple[int, ...]
    ramps_s: tuple[tuple[float, float], ...]
    reference_current_A: np.ndarray  # environment x time
    target_current_A: np.ndarray  # environment x ramp pair x time


def build_waveform_split(name: str) -> WaveformSplit:
    """Build one declared split, allowing test curves to stay unread during tuning."""
    designs = {'train': (TRAIN_SEEDS, TRAIN_RAMPS), 'validation': (VALIDATION_SEEDS, VALIDATION_RAMPS),
               'test': (TEST_SEEDS, TEST_RAMPS)}
    if name not in designs:
        raise ValueError(f'Unknown waveform split: {name}')
    seeds, ramps = designs[name]
    reference, target = [], []
    for seed in seeds:
        schedule = habit._environment(seed)
        reference.append(habit._driving_curve(schedule, habit.STEADY)[0][:, 1])
        target.append([habit._driving_curve(schedule, pair)[0][:, 1] for pair in ramps])
    reference, target = np.array(reference), np.array(target)
    reference.setflags(write=False)
    target.setflags(write=False)
    return WaveformSplit(name, seeds, ramps, reference, target)


def build_waveform_data() -> dict[str, WaveformSplit]:
    """Build all declared splits. Use build_waveform_split for staged tuning."""
    return {name: build_waveform_split(name) for name in ('train', 'validation', 'test')}


def _checked_ramps(ramps_s) -> np.ndarray:
    try:
        rows = [tuple(pair) for pair in ramps_s]
    except TypeError as error:
        raise ValueError('Supply launch and brake ramp pairs') from error
    if not rows or any(len(pair) != 2 for pair in rows):
        raise ValueError('Supply at least one launch and brake ramp pair')
    for pair in rows:
        for value, bounds in zip(pair, RAMP_BOUNDS_S):
            if isinstance(value, bool) or not isinstance(value, Real) or not bounds[0] <= value <= bounds[1]:
                raise ValueError('Ramps must be finite seconds, launch 8–16 and brake 8–15')
    return habit._scale_ramps(rows)


def _matern(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
    distance = np.sqrt(3.) * np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1) / length
    return (1. + distance) * np.exp(-distance)


def _anchored_kernel(left: np.ndarray, right: np.ndarray, length: float) -> np.ndarray:
    steady = habit._scale_ramps([habit.STEADY])
    return _matern(left, right, length) - _matern(left, steady, length) @ _matern(steady, right, length)


@dataclass(frozen=True)
class WaveformModel:
    method: str
    params: dict[str, float]
    train_seeds: tuple[int, ...]
    train_ramps_s: tuple[tuple[float, float], ...]
    steady_mean_A: np.ndarray
    weights: np.ndarray

    def predict_currents(self, ramps_s) -> np.ndarray:
        """Return one current row for each requested ramp pair."""
        scaled = _checked_ramps(ramps_s)
        if self.method == 'gp':
            basis = _anchored_kernel(scaled, habit._scale_ramps(self.train_ramps_s), self.params['length_scale'])
        else:
            basis = habit._features(scaled, 2)
        result = self.steady_mean_A + basis @ self.weights
        result[np.all(scaled == 1., axis=1)] = self.steady_mean_A
        return result

    def predict(self, launch_ramp_s: float, brake_ramp_s: float) -> np.ndarray:
        """Return [elapsed_s, cell_current_A] suitable for a continuous input."""
        current, = self.predict_currents([(launch_ramp_s, brake_ramp_s)])
        return np.column_stack([ELAPSED_S, current])

    def reference_profile(self) -> np.ndarray:
        """Return the common mean steady training curve."""
        return np.column_stack([ELAPSED_S, self.steady_mean_A])

    def predict_residual(self, launch_ramp_s: float, brake_ramp_s: float) -> np.ndarray:
        """Return the signed current added to the steady training reference."""
        profile = self.predict(launch_ramp_s, brake_ramp_s)
        profile[:, 1] -= self.steady_mean_A
        return profile


def fit_waveform(train: WaveformSplit, method: str, **params: float) -> WaveformModel:
    """Fit the common train-only target with the requested regularization."""
    if train.name != 'train':
        raise ValueError('Waveform fitting requires the training split')
    if method not in MODEL_BOUNDS or params.keys() - MODEL_BOUNDS[method].keys():
        raise ValueError(f'Unknown waveform method or parameters: {method} {tuple(params)}')
    settings = {**DEFAULT_PARAMS[method], **params}
    for key, value in settings.items():
        low, high = MODEL_BOUNDS[method][key]
        if isinstance(value, bool) or not isinstance(value, Real) or not low <= value <= high:
            raise ValueError(f'{key} must be finite and within [{low}, {high}]')
    x = _checked_ramps(train.ramps_s)
    expected_reference_shape = (len(train.seeds), len(ELAPSED_S))
    expected_target_shape = (len(train.seeds), len(train.ramps_s), len(ELAPSED_S))
    if (not train.seeds or train.reference_current_A.shape != expected_reference_shape
            or train.target_current_A.shape != expected_target_shape
            or not np.isfinite(train.reference_current_A).all() or not np.isfinite(train.target_current_A).all()):
        raise ValueError('Expected paired finite training curves on the common elapsed grid')
    ridge = 10. ** settings['log10_ridge']
    steady = train.reference_current_A.mean(axis=0)
    residual = (train.target_current_A - train.reference_current_A[:, None, :]).mean(axis=0)
    if method == 'gp':
        kernel = _anchored_kernel(x, x, settings['length_scale'])
        weights = solve(kernel + ridge * np.eye(len(x)), residual, assume_a='pos')
    else:
        features = habit._features(x, 2)
        weights = solve(features.T @ features + ridge * np.eye(features.shape[1]), features.T @ residual, assume_a='pos')
    steady.setflags(write=False)
    weights.setflags(write=False)
    return WaveformModel(method, settings, train.seeds, train.ramps_s, steady, weights)


def paired_charge_metrics(target_A, reference_A) -> dict[str, np.ndarray]:
    """Integrate paired signed delta and discharge-only excess on linear curves."""
    target, reference = np.broadcast_arrays(np.asarray(target_A, dtype=float), np.asarray(reference_A, dtype=float))
    if target.shape[-1:] != ELAPSED_S.shape or not np.isfinite(target).all() or not np.isfinite(reference).all():
        raise ValueError('Paired charge metrics require finite currents on the common elapsed grid')
    delta = target - reference
    left = target[..., :-1]

    def crossing(curve):
        start, end = curve[..., :-1], curve[..., 1:]
        opposite_signs = ((start < 0) & (end > 0)) | ((start > 0) & (end < 0))
        return np.divide(-start, end - start, out=np.zeros_like(start), where=opposite_signs)

    # The positive discharge excess is piecewise linear until the target,
    # reference, or their difference crosses zero. Insert all three breaks.
    bounds = np.sort(np.stack([np.zeros_like(left), crossing(target), crossing(reference), crossing(delta),
                               np.ones_like(left)], axis=-1), axis=-1)
    target_at = left[..., None] + np.diff(target, axis=-1)[..., None] * bounds
    reference_at = reference[..., :-1, None] + np.diff(reference, axis=-1)[..., None] * bounds
    excess_at = np.maximum(np.minimum(target_at, target_at - reference_at), 0.)
    integrated = np.sum(np.diff(bounds, axis=-1) * (excess_at[..., 1:] + excess_at[..., :-1]) / 2, axis=-1)
    return {
        'signed_net_delta_Ah': np.trapezoid(delta, ELAPSED_S, axis=-1) / 3600,
        'positive_discharge_excess_Ah': np.sum(integrated * np.diff(ELAPSED_S), axis=-1) / 3600,
    }


def score_waveform(model: WaveformModel, split: WaveformSplit, *, include_burdens: bool = True) -> dict:
    """Score a fixed model on complete episodes without fitting on this split."""
    prediction = model.predict_currents(split.ramps_s)
    error = prediction - split.target_current_A
    result = {
        'episode_rmse_A': float(np.sqrt(np.mean(error**2))),
        'ramp_mean_rmse_A': float(np.sqrt(np.mean(error.mean(axis=0)**2))),
        'episode_mae_A': float(np.mean(abs(error))),
        'ramp_count': len(split.ramps_s), 'environment_count': len(split.seeds),
        'predicted_current_A': prediction.tolist(),
        'maximum_predicted_current_magnitude_A': float(np.max(abs(prediction))),
    }
    if include_burdens:
        def burdens(current):
            return habit._values(np.column_stack([ELAPSED_S, current]))

        predicted_burdens = np.array([burdens(current) for current in prediction])
        observed_burdens = np.array([[burdens(current) for current in row] for row in split.target_current_A])
        burden_error = predicted_burdens - observed_burdens
        result['burdens'] = {
            'metric_order': list(habit.METRICS),
            'predicted_curve_by_ramp': predicted_burdens.tolist(),
            'mean_of_observed_curves_by_ramp': observed_burdens.mean(axis=0).tolist(),
            'episode_rmse': habit._as_metrics(np.sqrt(np.mean(burden_error**2, axis=(0, 1)))),
            'ramp_mean_bias_rmse': habit._as_metrics(np.sqrt(np.mean(burden_error.mean(axis=0)**2, axis=0))),
        }
        predicted_charge = paired_charge_metrics(prediction, model.steady_mean_A)
        observed_charge = paired_charge_metrics(split.target_current_A, split.reference_current_A[:, None, :])
        result['signed_residual_A'] = (prediction - model.steady_mean_A).tolist()
        result['training_steady_reference_A'] = model.steady_mean_A.tolist()
        result['relative_to_reference'] = {
            'prediction_reference': 'fixed training mean steady curve',
            'observed_reference': 'paired steady curve under the same generated environment',
            'definitions': {'signed_net_delta_Ah': 'signed integral of target minus reference',
                            'positive_discharge_excess_Ah': 'integral of positive part of target discharge minus reference discharge'},
            'predicted_by_ramp': {name: values.tolist() for name, values in predicted_charge.items()},
            'mean_observed_by_ramp': {name: values.mean(axis=0).tolist() for name, values in observed_charge.items()},
            'episode_rmse': {name: float(np.sqrt(np.mean((values - observed_charge[name])**2)))
                             for name, values in predicted_charge.items()},
        }
    return result
