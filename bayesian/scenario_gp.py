"""Compare synthetic driving forecasts with and without a condition baseline.

The generator approximates longitudinal power on fixed virtual roads. Its
currents are exploratory single-cell inputs, not fitted vehicle telemetry.
Each target episode has a reference driver on the same realized route and
traffic schedule. Test pairs are never included in a fitted baseline or GP.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .driving import _fit


ELAPSED_S = np.arange(251, dtype=float)
TRAIN_SEEDS = tuple(range(101, 117))
TEST_SEEDS = tuple(range(501, 509))
LENGTH_SCALE_S = 8.0
DEMO_EPISODE = ('urban_grid', 'moderate', 501)
CELL_BASIS_ID = 'okane2022-spme-single-cell'
ROUTES = {
    'urban_grid': ((300., 360., 310.), (.012, -.007, -.001)),
    'urban_arterial': ((500., 540., 490.), (.004, -.002, -.001)),
}
TRAFFIC = {
    'clear': ((47, 56, 49), (10, 15)),
    'moderate': ((53, 60, 58), (30, 28)),
}
VEHICLE_ASSUMPTIONS = {
    'mass_kg': 1570.0, 'gravity_m_s2': 9.81, 'rolling_coefficient': .011,
    'air_density_kg_m3': 1.2, 'drag_area_m2': .62, 'drive_efficiency': .89,
    'regeneration_efficiency': .62, 'nominal_cell_voltage_V': 3.6,
    'series_cells': 108, 'parallel_cells': 32,
}


@dataclass(frozen=True)
class Episode:
    reference_A: np.ndarray
    target_A: np.ndarray
    reference_speed_m_s: np.ndarray
    target_speed_m_s: np.ndarray
    schedule: dict


def _speed(route: str, starts: np.ndarray, travel: np.ndarray,
           launch: np.ndarray, brake: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths, grades = ROUTES[route]
    speed = np.zeros_like(ELAPSED_S)
    grade = np.zeros_like(ELAPSED_S)
    for distance, slope, start, duration, up, down in zip(lengths, grades, starts, travel, launch, brake):
        t = np.arange(duration + 1, dtype=float)
        shape = np.minimum(np.minimum(t / up, (duration - t) / down), 1.)
        # Match distance by choosing the physically traversed cruise speed,
        # not by rescaling the derived electric current or its integral.
        speed[start:start + duration + 1] = distance * shape / np.trapezoid(shape, t)
        grade[start:start + duration + 1] = slope
    if not np.isclose(np.trapezoid(speed, ELAPSED_S), sum(lengths), rtol=0, atol=1e-9):
        raise ValueError('Generated speed did not traverse the declared distance')
    return speed, grade


def _current(speed: np.ndarray, grade: np.ndarray, auxiliary_W: float) -> np.ndarray:
    car = VEHICLE_ASSUMPTIONS
    acceleration = np.gradient(speed, ELAPSED_S)
    force = car['mass_kg'] * (acceleration + car['gravity_m_s2'] * (grade + car['rolling_coefficient']))
    force += .5 * car['air_density_kg_m3'] * car['drag_area_m2'] * speed**2
    wheel_W = speed * force
    pack_W = np.where(wheel_W >= 0., wheel_W / car['drive_efficiency'],
                      wheel_W * car['regeneration_efficiency']) + auxiliary_W
    # Fixed nominal voltage and equal parallel current sharing are assumptions.
    cell_A = pack_W / (car['nominal_cell_voltage_V'] * car['series_cells'] * car['parallel_cells'])
    if not np.isfinite(cell_A).all() or np.max(abs(cell_A)) > 5 or np.trapezoid(cell_A, ELAPSED_S) <= 0:
        raise ValueError('Generated current exceeds the exploratory cell input bounds')
    return cell_A


def generate_episode(route: str, traffic: str, seed: int) -> Episode:
    """Generate paired drivers under a common independent episode draw."""
    route_index = tuple(ROUTES).index(route)
    traffic_index = tuple(TRAFFIC).index(traffic)
    sequence = np.random.SeedSequence([seed, route_index, traffic_index])
    environment, reference_rng, target_rng = [np.random.default_rng(s) for s in sequence.spawn(3)]
    nominal_travel, nominal_dwell = TRAFFIC[traffic]
    travel = np.array(nominal_travel) + environment.integers(-2, 3, 3)
    dwell = np.array(nominal_dwell) + environment.integers(-2, 3, 2)
    starts = environment.integers(3, 6) + np.r_[0, np.cumsum(travel[:-1] + dwell)]
    auxiliary_W = float(environment.uniform(420., 640.))
    if starts[-1] + travel[-1] > ELAPSED_S[-1]:
        raise ValueError('Generated trip exceeds the shared comparison horizon')
    reference_launch = reference_rng.integers(13, 18, 3)
    reference_brake = reference_rng.integers(12, 17, 3)
    target_launch = target_rng.integers(7, 11, 3)
    target_brake = target_rng.integers(8, 13, 3)
    reference_speed, grade = _speed(route, starts, travel, reference_launch, reference_brake)
    target_speed, _ = _speed(route, starts, travel, target_launch, target_brake)
    return Episode(_current(reference_speed, grade, auxiliary_W),
                   _current(target_speed, grade, auxiliary_W), reference_speed, target_speed,
                   {'start_s': starts.tolist(), 'travel_s': travel.tolist(), 'inter_segment_wait_s': dwell.tolist(),
                    'auxiliary_W': auxiliary_W, 'reference_launch_s': reference_launch.tolist(),
                    'reference_brake_s': reference_brake.tolist(), 'target_launch_s': target_launch.tolist(),
                    'target_brake_s': target_brake.tolist()})


def _metrics(observed: np.ndarray, prediction: np.ndarray) -> dict:
    residual = prediction - observed
    return {
        'rmse_A': float(np.sqrt(np.mean(residual**2))),
        'mae_A': float(np.mean(abs(residual))),
        'signed_charge_mae_Ah': float(np.mean(abs(np.trapezoid(residual, ELAPSED_S, axis=-1) / 3600))),
        'peak_discharge_mae_A': float(np.mean(abs(prediction.max(axis=-1) - observed.max(axis=-1)))),
    }


def _curve(values: np.ndarray) -> list:
    return np.column_stack([ELAPSED_S, values]).tolist()


def evaluate_scenarios() -> dict:
    """Evaluate predeclared held-out episodes in known virtual conditions."""
    contexts = [(route, traffic) for route in ROUTES for traffic in TRAFFIC]
    training = {pair: [generate_episode(*pair, seed) for seed in TRAIN_SEEDS] for pair in contexts}
    # Same target driver and a condition-independent arithmetic mean provide
    # a fairer baseline than a population average of reference drivers alone.
    all_target_mean = np.mean([ep.target_A for members in training.values() for ep in members], axis=0)
    all_reference_mean = np.mean([ep.reference_A for members in training.values() for ep in members], axis=0)
    d = np.sqrt(3.) * abs(ELAPSED_S[:, None] - ELAPSED_S[None, :]) / LENGTH_SCALE_S
    kernel = (1. + d) * np.exp(-d)
    forecasts = {}
    for pair, members in training.items():
        reference = np.stack([ep.reference_A for ep in members])
        target = np.stack([ep.target_A for ep in members])
        baseline, gp, covariance = _fit(target, reference, kernel)
        forecasts[pair] = (
            {'all_conditions_reference_mean': all_reference_mean,
             'all_conditions_personal_mean': all_target_mean,
             'condition_reference_mean': baseline,
             'condition_personal_mean': target.mean(axis=0),
             'condition_residual_gp': gp},
            1.6448536269514722 * np.sqrt(np.maximum(np.diag(covariance), 0.)),
        )

    conditions, episodes, targets, gp_half_widths = {}, [], [], []
    predictions = {name: [] for name in next(iter(forecasts.values()))[0]}
    demo = None
    for (route, traffic), (models, half_width) in forecasts.items():
        truth = []
        for seed in TEST_SEEDS:
            ep = generate_episode(route, traffic, seed)
            truth.append(ep.target_A)
            episodes.append({'route_id': route, 'traffic': traffic, 'seed': seed,
                             'target_current_A': ep.target_A.tolist(),
                             'metrics': {name: _metrics(ep.target_A, curve) for name, curve in models.items()}})
            if (route, traffic, seed) == DEMO_EPISODE:
                demo = {
                    'episode_seed': seed, 'schedule': ep.schedule,
                    'context': {'route_id': route, 'direction': 'forward', 'traffic': traffic,
                                'distance_m': sum(ROUTES[route][0]), 'cell_basis_id': CELL_BASIS_ID,
                                'source_kind': 'synthetic_condition_residual_gp'},
                    'profiles': {'matched_reference': _curve(ep.reference_A),
                                 'gp_mean': _curve(models['condition_residual_gp']),
                                 'heldout_target': _curve(ep.target_A)},
                    'condition_mean_reference_profile': _curve(models['condition_reference_mean']),
                    'reference_speed_m_s': ep.reference_speed_m_s.tolist(),
                    'target_speed_m_s': ep.target_speed_m_s.tolist(),
                    'integrated_reference_distance_m': float(np.trapezoid(ep.reference_speed_m_s, ELAPSED_S)),
                    'integrated_target_distance_m': float(np.trapezoid(ep.target_speed_m_s, ELAPSED_S)),
                    'gp_rmse_A': _metrics(ep.target_A, models['condition_residual_gp'])['rmse_A'],
                    'gp_abs_max_A': float(np.max(abs(models['condition_residual_gp']))),
                    'gp_signed_charge_Ah': float(np.trapezoid(models['condition_residual_gp'], ELAPSED_S) / 3600),
                }
        observed = np.stack(truth)
        coverage = np.mean(abs(observed - models['condition_residual_gp']) <= half_width)
        conditions[f'{route}/{traffic}'] = {
            'distance_m': sum(ROUTES[route][0]), 'direction': 'forward',
            'train_reference_count': len(TRAIN_SEEDS), 'train_personal_count': len(TRAIN_SEEDS),
            'test_episode_count': len(TEST_SEEDS),
            'metrics': {name: _metrics(observed, curve) for name, curve in models.items()},
            'gp_interval_90_pointwise_coverage': float(coverage),
            'forecasts_A': {name: curve.tolist() for name, curve in models.items()},
            'gp_interval_90_half_width_A': half_width.tolist(),
        }
        targets.extend(truth)
        gp_half_widths.extend([half_width] * len(truth))
        for name, curve in models.items():
            predictions[name].extend([curve] * len(truth))
    observed = np.stack(targets)
    root = Path(__file__).resolve().parent
    if demo is None:
        raise RuntimeError('Predesignated demo episode is missing from the test split')
    return {
        'schema': 'synthetic_condition_residual_gp_v1', 'status': 'complete',
        'scope': 'new synthetic episodes under four previously observed route/traffic combinations',
        'real_road_validation': False, 'unseen_condition_validation': False,
        'elapsed_s': ELAPSED_S.tolist(), 'current_sign': 'positive_discharge_negative_regeneration',
        'split': {'train_seeds': list(TRAIN_SEEDS), 'test_seeds': list(TEST_SEEDS),
                  'train_episode_count': len(TRAIN_SEEDS) * len(contexts), 'test_episode_count': len(episodes),
                  'test_sample_count': len(episodes) * len(ELAPSED_S),
                  'model_inputs': 'route and traffic class, elapsed time, past reference and personal curves',
                  'demo_rule': 'fixed route, traffic and first test seed before fitting'},
        'generator': {'routes': ROUTES, 'traffic': TRAFFIC, 'vehicle_assumptions': VEHICLE_ASSUMPTIONS,
                      'reference_launch_s_range': [13, 17], 'target_launch_s_range': [7, 10],
                      'reference_brake_s_range': [12, 16], 'target_brake_s_range': [8, 12],
                      'shared_environment': 'same path, section travel times, stops and auxiliary load per pair'},
        'model': {'method': 'condition-specific reference mean plus functional GP residual',
                  'kernel': 'Matern 3/2', 'fixed_length_scale_s': LENGTH_SCALE_S,
                  'length_scale_rationale': 'fixed before evaluation, order of a virtual launch/braking ramp',
                  'test_tuning': False},
        'metrics': {name: _metrics(observed, np.stack(curves)) for name, curves in predictions.items()},
        'gp_interval_90_pointwise_coverage': float(np.mean(
            abs(observed - np.stack(predictions['condition_residual_gp'])) <= np.stack(gp_half_widths))),
        'conditions': conditions, 'heldout_episodes': episodes, 'demo': demo,
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          for name in ('scenario_gp.py', 'driving.py')},
        'limits': ['Generator choices are assumptions, not vehicle calibration.',
                   'Paired physical reference is one generated run, distinct from the training reference mean.',
                   'Predicted current does not reconstruct a speed curve or certify completion of the distance.',
                   'The matched actual pair is synthetic. No driver population, unseen-route or battery life validation.'],
    }


def run_scenario_gp(output) -> dict:
    """Write the forecast and its predesignated comparison inputs to a new directory."""
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    report = evaluate_scenarios()
    for name, content in (('scenario_gp.json', report), ('demo_profiles.json', report['demo'])):
        (path / name).write_text(json.dumps(content, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def demo_comparison(report: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Return unmodified curves and declared context for the twin comparison."""
    demo = report['demo']
    return {name: np.array(profile) for name, profile in demo['profiles'].items()}, dict(demo['context'])
