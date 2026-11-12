"""Predict changes in virtual driving loads with the selected quadratic model."""
from __future__ import annotations

import hashlib
from numbers import Real
from pathlib import Path

import numpy as np

from . import habit_stress


def predict_habit_burden(launch_ramp_s: float, brake_ramp_s: float) -> dict:
    """Return steady-relative load changes within the benchmark's ramp domain."""
    names = ('launch_ramp_s', 'brake_ramp_s')
    inputs = (launch_ramp_s, brake_ramp_s)
    bounds = {}
    for name, value, candidates in zip(names, inputs, zip(*habit_stress.TRAIN_RAMPS)):
        lower, upper = min(candidates), max(candidates)
        if isinstance(value, bool) or not isinstance(value, Real) or not lower <= value <= upper:
            raise ValueError(f'{name} must be a finite number from {lower} to {upper} seconds')
        bounds[name] = [lower, upper]

    train = habit_stress._delta_samples(habit_stress.TRAIN_SEEDS, habit_stress.TRAIN_RAMPS)
    train_features = habit_stress._features(habit_stress._scale_ramps(habit_stress.TRAIN_RAMPS), 2)
    weights = np.linalg.lstsq(train_features, train.mean(axis=0), rcond=None)[0]
    query_features = habit_stress._features(habit_stress._scale_ramps([inputs]), 2)
    predicted = (query_features @ weights)[0]
    source_root = Path(__file__).resolve().parent
    return {
        'schema': 'synthetic_habit_quadratic_prediction_v1',
        'status': 'complete',
        'method': 'quadratic',
        'context': {'route_id': habit_stress.ROUTE, 'traffic': habit_stress.TRAFFIC,
                    'cell_basis_id': habit_stress.scenario.CELL_BASIS_ID},
        'ramps_s': dict(zip(names, map(float, inputs))),
        'reference_ramps_s': dict(zip(names, habit_stress.STEADY)),
        'supported_ramps_s': bounds,
        'predicted_delta_vs_steady': habit_stress._as_metrics(predicted),
        'training': {'environment_seeds': list(habit_stress.TRAIN_SEEDS), 'ramp_design_s': habit_stress.TRAIN_RAMPS,
                     'feature_order': ['launch', 'brake', 'launch^2', 'launch*brake', 'brake^2'],
                     'coefficient_basis': 'normalize each ramp to [0,1] and subtract each feature at the steady reference',
                     'metric_order': list(habit_stress.METRICS), 'coefficients': weights.tolist()},
        'scope': 'scalar changes in synthetic current burdens on the fixed virtual route',
        'real_driver_validation': False, 'battery_life_predicted': False,
        'source_sha256': {name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
                          for name in ('habit_predictor.py', 'habit_stress.py', 'scenario_gp.py')},
    }
