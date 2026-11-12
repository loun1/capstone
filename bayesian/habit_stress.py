"""Virtual driving habit comparison and GP prediction of paired burden changes.

All arms finish the same road under the same realized stop and travel schedule.
The current generator is an approximate virtual vehicle. Its current metrics
describe electrical loading, without inferring cell temperature, aging, or life.
"""
from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve

from . import scenario_gp as scenario


ROUTE = 'urban_grid'
TRAFFIC = 'moderate'
STEADY = (16, 15)
FACTORIAL_ARMS = {'steady': STEADY, 'quick_launch': (8, 15), 'quick_brake': (16, 8), 'both': (8, 8)}
FACTORIAL_SEEDS = tuple(range(1701, 1733))
DEMO_SEED = 1701
TRAIN_SEEDS = tuple(range(1301, 1325))
TEST_SEEDS = tuple(range(1501, 1533))
TRAIN_RAMPS = tuple(product((8, 12, 16), (8, 11, 15)))
TEST_RAMPS = tuple(product((9, 10, 13, 15), (9, 10, 12, 14)))
METRICS = ('peak_discharge_A', 'peak_regen_A', 'i_squared_A2s', 'absolute_throughput_Ah')
DIRECTION_TOLERANCES = dict(zip(METRICS, (.02, .02, .2, .0001)))
KERNEL_LENGTH = .5
KERNEL_JITTER = 1e-8


def current_burdens(profile: np.ndarray) -> dict[str, float]:
    """Integrate a time/current curve exactly under linear interpolation."""
    data = np.asarray(profile, dtype=float)
    if (data.ndim != 2 or data.shape[1] != 2 or len(data) < 2
            or not np.isfinite(data).all() or np.any(np.diff(data[:, 0]) <= 0)):
        raise ValueError('Expected finite current on strictly increasing time points')
    current = data[:, 1]
    duration = np.diff(data[:, 0])
    left, right = current[:-1], current[1:]
    abs_left, abs_right = abs(left), abs(right)
    abs_average = (abs_left + abs_right) / 2
    crosses_zero = ((left < 0) & (right > 0)) | ((left > 0) & (right < 0))
    abs_average[crosses_zero] = (left[crosses_zero]**2 + right[crosses_zero]**2) / (
        2 * (abs_left[crosses_zero] + abs_right[crosses_zero]))
    return {
        'peak_discharge_A': float(max(0., current.max())),
        'peak_regen_A': float(max(0., -current.min())),
        'i_squared_A2s': float(np.sum(duration * (left**2 + left * right + right**2) / 3)),
        'absolute_throughput_Ah': float(np.dot(duration, abs_average) / 3600),
    }


def _driving_curve(schedule: dict, ramps: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    count = len(schedule['travel_s'])
    launch, brake = ramps
    speed, grade = scenario._speed(ROUTE, np.array(schedule['start_s']), np.array(schedule['travel_s']),
                                   np.full(count, launch), np.full(count, brake))
    current = scenario._current(speed, grade, schedule['auxiliary_W'])
    return np.column_stack([scenario.ELAPSED_S, current]), speed


def _environment(seed: int) -> dict:
    # The first RNG stream of the existing generator selects environment
    # independently of the two style streams. Only environment keys are used.
    schedule = scenario.generate_episode(ROUTE, TRAFFIC, seed).schedule
    return {key: schedule[key] for key in ('start_s', 'travel_s', 'inter_segment_wait_s', 'auxiliary_W')}


def _values(curve: np.ndarray) -> list[float]:
    burdens = current_burdens(curve)
    return [burdens[metric] for metric in METRICS]


def _delta_samples(seeds, ramp_design) -> np.ndarray:
    rows = []
    for seed in seeds:
        schedule = _environment(seed)
        reference = np.array(_values(_driving_curve(schedule, STEADY)[0]))
        rows.append([np.array(_values(_driving_curve(schedule, ramps)[0])) - reference for ramps in ramp_design])
    return np.array(rows)


def _as_metrics(values) -> dict[str, float]:
    return dict(zip(METRICS, map(float, values)))


def _factorial() -> tuple[dict, dict]:
    samples = {name: [] for name in FACTORIAL_ARMS}
    currents = {name: [] for name in FACTORIAL_ARMS}
    demo = None
    for seed in FACTORIAL_SEEDS:
        schedule = _environment(seed)
        arms = {name: _driving_curve(schedule, ramps) for name, ramps in FACTORIAL_ARMS.items()}
        for name, (curve, _) in arms.items():
            samples[name].append(_values(curve))
            currents[name].append(curve[:, 1].tolist())
        if seed == DEMO_SEED:
            demo = {
                'episode_seed': seed, 'schedule': schedule,
                'profiles': {name: curve.tolist() for name, (curve, _) in arms.items()},
                'speed_m_s': {name: speed.tolist() for name, (_, speed) in arms.items()},
                'integrated_distance_m': {name: float(np.trapezoid(speed, scenario.ELAPSED_S))
                                          for name, (_, speed) in arms.items()},
                'burdens': {name: current_burdens(curve) for name, (curve, _) in arms.items()},
                'context': {'route_id': ROUTE, 'direction': 'forward', 'traffic': TRAFFIC,
                            'distance_m': sum(scenario.ROUTES[ROUTE][0]), 'cell_basis_id': scenario.CELL_BASIS_ID,
                            'source_kind': 'synthetic_controlled_habit_stress'},
            }
    if demo is None:
        raise RuntimeError('Prespecified factorial demonstration seed is missing')
    arrays = {name: np.array(values) for name, values in samples.items()}
    means = {name: matrix.mean(axis=0) for name, matrix in arrays.items()}
    changes = {name: matrix - arrays['steady'] for name, matrix in arrays.items()}
    summary = {
        name: {'mean': _as_metrics(means[name]), 'paired_mean_delta_vs_steady': _as_metrics(change.mean(axis=0)),
               'paired_delta_std': _as_metrics(change.std(axis=0, ddof=1)),
               'percent_change_of_mean_vs_steady': _as_metrics(100 * change.mean(axis=0) / means['steady'])}
        for name, change in changes.items()
    }
    interaction = changes['both'] - changes['quick_launch'] - changes['quick_brake']
    return {'seeds': list(FACTORIAL_SEEDS), 'ramps_s': FACTORIAL_ARMS,
            'current_A': currents, 'observed': samples, 'summary': summary,
            'paired_factorial_interaction_mean': _as_metrics(interaction.mean(axis=0))}, demo


def _scale_ramps(ramps) -> np.ndarray:
    return (np.array(ramps, dtype=float) - [8., 8.]) / [8., 7.]


def _matern(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    distance = np.sqrt(3.) * np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1) / KERNEL_LENGTH
    return (1. + distance) * np.exp(-distance)


def _kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    anchor = _scale_ramps([STEADY])
    # Condition the unit-variance prior on the exact zero change at steady.
    return _matern(left, right) - _matern(left, anchor) @ _matern(anchor, right)


def _features(scaled_ramps: np.ndarray, degree: int) -> np.ndarray:
    launch, brake = scaled_ramps.T
    columns = [launch - 1., brake - 1.]
    if degree == 2:
        columns.extend([launch**2 - 1., launch * brake - 1., brake**2 - 1.])
    return np.column_stack(columns)


def _predict(train: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    x_train, x_test = _scale_ramps(TRAIN_RAMPS), _scale_ramps(TEST_RAMPS)
    target = train.mean(axis=0)
    predicted = {'no_change': np.zeros((len(x_test), len(METRICS)))}
    fits = {}
    for name, degree in (('linear', 1), ('quadratic', 2)):
        weights = np.linalg.lstsq(_features(x_train, degree), target, rcond=None)[0]
        predicted[name] = _features(x_test, degree) @ weights
        fits[name] = {'feature_order': ['launch', 'brake'] + (['launch^2', 'launch*brake', 'brake^2'] if degree == 2 else []),
                      'normalized_and_centered_feature_coefficients': weights.tolist()}
    k = _kernel(x_train, x_train)
    cross = _kernel(x_test, x_train)
    columns, scalar_fits = [], {}
    for index, metric in enumerate(METRICS):
        scale = max(float(np.sqrt(np.mean(target[:, index]**2))), 1e-10)
        scaled_samples = train[:, :, index] / scale
        # One independent row per shared environment. Its cross-design
        # covariance accounts for repeatedly using each environment across ramps.
        noise = np.cov(scaled_samples, rowvar=False, ddof=1) / len(train)
        weights = solve(k + noise + KERNEL_JITTER * np.eye(len(x_train)),
                        target[:, index] / scale, assume_a='pos')
        columns.append(scale * (cross @ weights))
        scalar_fits[metric] = {'output_scale': scale, 'scaled_mean_noise_covariance': noise.tolist(),
                               'scaled_kernel_weights': weights.tolist()}
    predicted['gp'] = np.column_stack(columns)
    fits['gp'] = {'kernel': 'anchored Matern 3/2', 'normalized_length_scale': KERNEL_LENGTH,
                  'diagonal_jitter': KERNEL_JITTER, 'scalar_fits': scalar_fits}
    return predicted, fits


def _score(observed: np.ndarray, predicted: np.ndarray) -> dict:
    error = observed - predicted
    tolerance = np.array(list(DIRECTION_TOLERANCES.values()))
    meaningful = abs(observed) > tolerance
    direction = np.where(abs(predicted) > tolerance, np.sign(predicted), 0)
    correct = meaningful & (np.sign(observed) == direction)
    tested = meaningful.sum(axis=(0, 1))
    count = correct.sum(axis=(0, 1))
    return {'episode_rmse': _as_metrics(np.sqrt(np.mean(error**2, axis=(0, 1)))),
            'test_ramp_mean_rmse': _as_metrics(np.sqrt(np.mean(error.mean(axis=0)**2, axis=0))),
            'direction_agreement': {metric: {'correct_count': int(a), 'eligible_count': int(b),
                                              'fraction': float(a / b) if b else None}
                                    for metric, a, b in zip(METRICS, count, tested)}}


def evaluate_habit_stress() -> dict:
    """Fix models on training environments, then score independent environments and ramps."""
    seed_sets = (TRAIN_SEEDS, TEST_SEEDS, FACTORIAL_SEEDS)
    if sum(map(len, seed_sets)) != len(set().union(*seed_sets)):
        raise ValueError('Experiment seeds must be disjoint')
    train = _delta_samples(TRAIN_SEEDS, TRAIN_RAMPS)
    predictions, fits = _predict(train)
    test = _delta_samples(TEST_SEEDS, TEST_RAMPS)
    factorial, demo = _factorial()
    root = Path(__file__).resolve().parent
    return {
        'schema': 'synthetic_habit_stress_v1', 'status': 'complete',
        'scope': 'controlled fixed virtual route; unseen ramp values and environment seeds within the generator',
        'real_driver_validation': False, 'actual_cell_temperature_predicted': False, 'battery_life_predicted': False,
        'context': demo['context'], 'elapsed_s': scenario.ELAPSED_S.tolist(), 'metric_order': list(METRICS),
        'metric_definitions': {'peak_discharge_A': 'maximum positive cell current',
                               'peak_regen_A': 'magnitude of maximum negative cell current',
                               'i_squared_A2s': 'exact linear-segment integral of squared current; fixed-resistance heat proxy',
                               'absolute_throughput_Ah': 'exact linear-segment absolute charge, with each zero crossing split'},
        'controls': {'ramps_s_order': ['launch', 'brake'], 'reference_ramps_s': list(STEADY),
                     'shared_within_seed': ['vehicle', 'route', 'distance', 'travel times', 'stops', 'auxiliary power'],
                     'cruise_speed': 'adjusted within each road section to meet its shared travel distance and time',
                     'vehicle_assumptions': scenario.VEHICLE_ASSUMPTIONS, 'route_sections': scenario.ROUTES[ROUTE]},
        'factorial': factorial,
        'gp': {'train_seeds': list(TRAIN_SEEDS), 'test_seeds': list(TEST_SEEDS),
               'train_ramps_s': TRAIN_RAMPS, 'test_ramps_s': TEST_RAMPS,
               'input_scaling': {'offset_s': [8., 8.], 'span_s': [8., 7.]},
               'input_features': ['launch_ramp_s', 'brake_ramp_s'],
               'array_axes': ['environment seed', 'ramp pair', 'metric'],
               'fit_target': 'paired metric change from steady within the same environment',
               'direction_tolerances': DIRECTION_TOLERANCES,
               'direction_rule': 'abs(truth) > tolerance; predictions within tolerance count as zero',
               'train_delta': train.tolist(), 'test_delta': test.tolist(),
               'test_prediction': {name: matrix.tolist() for name, matrix in predictions.items()},
               'fit_evidence': fits, 'test_scores': {name: _score(test, matrix) for name, matrix in predictions.items()},
               'test_tuning': False},
        'demo': demo,
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ('habit_stress.py', 'scenario_gp.py')},
    }


def run_habit_stress(output) -> dict:
    """Write the experiment and selected replay inputs into a new directory."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    report = evaluate_habit_stress()
    for name, content in (('habit_stress.json', report), ('demo_profiles.json', report['demo'])):
        (path / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report
