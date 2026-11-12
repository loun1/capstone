"""Replay causal voltage scheduling and cloud reconstruction on a ROM trace.

Every policy sends current, temperature, time and the already available
SPMe(T) prediction. It optionally sends the ROM equivalent voltage. The GP
predicts the residual voltage from earlier transmitted measurements only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve


CONTEXT_COLUMNS = ('elapsed_s', 'applied_representative_cell_current_A', 'module_sensor_temperature_C',
                   'known_replay_cell_voltage_V')
VOLTAGE_COLUMN = 'observed_ROM_equivalent_cell_voltage_V'
WIRE_SCHEMA = 'rom_equivalent_voltage_telemetry_jsonl_v1'
Z95 = 1.959963984540054


@dataclass(frozen=True)
class Settings:
    prior_sd_V: float = .020
    observation_noise_V: float = .001
    time_scale_s: float = 30.
    current_scale_A: float = .40
    temperature_scale_C: float = 2.
    max_history: int = 32
    trigger_sd_V: float = .007
    trigger_predicted_change_V: float = .015
    max_observation_gap_s: float = 10.

    def __post_init__(self):
        positive = [self.prior_sd_V, self.observation_noise_V, self.time_scale_s, self.current_scale_A,
                    self.temperature_scale_C, self.trigger_sd_V, self.trigger_predicted_change_V,
                    self.max_observation_gap_s]
        if not np.isfinite(positive).all() or min(positive) <= 0 or type(self.max_history) is not int or self.max_history < 1:
            raise ValueError('Telemetry GP settings must be finite and positive')


class ResidualGP:
    """Keep a bounded causal measurement window and predict ROM equivalent V."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.scale = np.array([settings.time_scale_s, settings.current_scale_A, settings.temperature_scale_C])
        self.points: list[np.ndarray] = []
        self.residuals: list[float] = []

    def _kernel(self, left, right):
        distance = np.sqrt(3.) * np.linalg.norm(left[:, None, :] - right[None, :, :], axis=-1)
        return self.settings.prior_sd_V**2 * (1. + distance) * np.exp(-distance)

    def predict(self, context: np.ndarray) -> tuple[float, float]:
        variance = self.settings.prior_sd_V**2 + self.settings.observation_noise_V**2
        mean = float(context[3])
        if self.points:
            points = np.stack(self.points)
            gram = self._kernel(points, points) + self.settings.observation_noise_V**2 * np.eye(len(points))
            cross = self._kernel(points, context[None, :3] / self.scale)[:, 0]
            solved = cho_solve(cho_factor(gram, lower=True), np.column_stack([self.residuals, cross]))
            mean += float(cross @ solved[:, 0])
            variance -= float(cross @ solved[:, 1])
        return mean, float(np.sqrt(max(variance, 0.)))

    def observe(self, context: np.ndarray, voltage: float):
        self.points.append(context[:3] / self.scale)
        self.residuals.append(voltage - float(context[3]))
        if len(self.points) > self.settings.max_history:
            self.points.pop(0)
            self.residuals.pop(0)


def load_replay(path: str | Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read the known predictors and ROM voltage from a completed replay."""
    raw = Path(path).read_bytes()
    replay = json.loads(raw)
    if replay.get('schema') != 'pack_spme_replay_v1' or replay.get('status') != 'complete':
        raise ValueError('Use a completed pack replay result JSON')
    if len(replay['trace']) != replay['interval_count']:
        raise ValueError('Replay interval count does not match its trace')
    trace = replay['trace']
    context = np.array([[row[key] for key in ('elapsed_end_s', 'cell_current_a', 'observed_module_temperature_c',
                                              'model_voltage_end_v')] for row in trace], dtype=float)
    observed = np.array([row['observed_cell_voltage_v'] for row in trace], dtype=float)
    series = replay['mapping']['series_cells']
    pack = np.array([row['observed_pack_voltage_v'] for row in trace])
    if type(series) is not int or series < 1 or not np.allclose(pack / series, observed, rtol=0, atol=1e-12):
        raise ValueError('Replay voltage is inconsistent with pack-to-series equivalent conversion')
    _check_trace(context, observed)
    if not np.isfinite(observed).all():
        raise ValueError('The offline benchmark requires complete observed ROM voltage')
    return context, observed, {
        'file': str(Path(path).resolve()), 'sha256': sha256(raw).hexdigest(), 'cycle_id': replay['cycle_id'],
        'model_variant': replay['model'].get('variant'), 'temperature_mode': replay['temperature']['mode'],
        'voltage_basis': f'single ROM pack voltage channel divided by {series} series cells',
        'source_scope': replay['scope'], 'interval_count': replay['interval_count'],
    }


def _check_trace(context: np.ndarray, voltage: np.ndarray):
    if (context.ndim != 2 or context.shape[1] != len(CONTEXT_COLUMNS) or len(context) < 2
            or voltage.shape != (len(context),) or not np.isfinite(context).all()
            or np.any(np.diff(context[:, 0]) <= 0)):
        raise ValueError('Telemetry requires finite context with increasing times and aligned voltage slots')


def _line(value) -> bytes:
    return (json.dumps(value, separators=(',', ':'), allow_nan=False) + '\n').encode('utf-8')


def encode_telemetry(context, voltage, policy: str, *, settings=Settings(), uniform_count=None) -> tuple[bytes, dict]:
    """Choose each observation before reading its voltage, then serialize it."""
    context, voltage = np.asarray(context, dtype=float), np.asarray(voltage, dtype=float)
    _check_trace(context, voltage)
    if policy not in ('full', 'uniform', 'adaptive'):
        raise ValueError('Choose full, uniform or adaptive observation policy')
    uniform_indices = set()
    if policy == 'uniform':
        if type(uniform_count) is not int or not 1 <= uniform_count <= len(context):
            raise ValueError('Uniform selection needs an integer observation count inside the trace length')
        uniform_indices = set(np.rint(np.linspace(0, len(context) - 1, uniform_count)).astype(int))
    packet = [_line({'schema': WIRE_SCHEMA, 'context_columns': CONTEXT_COLUMNS, 'optional_last_column': VOLTAGE_COLUMN,
                     'gp_settings': asdict(settings)})]
    gp = ResidualGP(settings)
    previous_time, previous_voltage = None, None
    forecasts, spreads, sent, reasons = [], [], [], []
    for index, row in enumerate(context):
        mean, spread = gp.predict(row)
        if policy == 'full':
            reason = 'full'
        elif policy == 'uniform':
            reason = 'uniform' if index in uniform_indices else None
        elif previous_time is None:
            reason = 'initial'
        elif float(row[0]) - previous_time >= settings.max_observation_gap_s:
            reason = 'elapsed_gap'
        elif spread >= settings.trigger_sd_V:
            reason = 'posterior_sd'
        elif abs(mean - previous_voltage) >= settings.trigger_predicted_change_V:
            reason = 'predicted_change'
        else:
            reason = None
        record = row.tolist()
        if reason is not None:
            # The scheduler has already decided. Unsent values are unavailable
            # to the GP, to later scheduling decisions and to the serialized row.
            previous_time, previous_voltage = float(row[0]), float(voltage[index])
            if not np.isfinite(previous_voltage):
                raise ValueError(f'Selected voltage is missing or nonfinite at index {index}')
            record.append(previous_voltage)
            gp.observe(row, previous_voltage)
        packet.append(_line(record))
        forecasts.append(mean)
        spreads.append(spread)
        sent.append(reason is not None)
        reasons.append(reason)
    return b''.join(packet), {'prior_mean_V': np.array(forecasts), 'prior_sd_V': np.array(spreads),
                              'sent_mask': np.array(sent), 'selection_reasons': reasons}


def reconstruct_cloud(payload: bytes) -> dict:
    """Reconstruct each missing voltage from the serialized payload alone."""
    lines = payload.splitlines()
    if len(lines) < 3:
        raise ValueError('Telemetry payload requires a header and at least two records')
    header_bytes, *records = lines
    header = json.loads(header_bytes)
    if (header.get('schema') != WIRE_SCHEMA or tuple(header['context_columns']) != CONTEXT_COLUMNS
            or header['optional_last_column'] != VOLTAGE_COLUMN or len(records) < 2):
        raise ValueError('Unrecognized telemetry payload')
    gp = ResidualGP(Settings(**header['gp_settings']))
    means, spreads, reconstructed, sent = [], [], [], []
    last_time = -np.inf
    for encoded in records:
        row = np.asarray(json.loads(encoded), dtype=float)
        if (row.shape not in ((len(CONTEXT_COLUMNS),), (len(CONTEXT_COLUMNS) + 1,)) or not np.isfinite(row).all()
                or row[0] <= last_time):
            raise ValueError('Invalid or noncausal telemetry record')
        last_time = row[0]
        context = row[:len(CONTEXT_COLUMNS)]
        mean, spread = gp.predict(context)
        has_voltage = len(row) == len(CONTEXT_COLUMNS) + 1
        if has_voltage:
            gp.observe(context, float(row[-1]))
        means.append(mean)
        spreads.append(spread)
        reconstructed.append(float(row[-1]) if has_voltage else mean)
        sent.append(has_voltage)
    return {'prior_mean_V': np.array(means), 'prior_sd_V': np.array(spreads),
            'reconstructed_V': np.array(reconstructed), 'sent_mask': np.array(sent)}


def _metrics(voltage, mean, spread, mask) -> dict:
    error = voltage[mask] - mean[mask]
    if not len(error):
        return {'count': 0, 'rmse_mV': None, 'mae_mV': None, 'coverage_95': None}
    return {'count': len(error), 'rmse_mV': float(1000 * np.sqrt(np.mean(error**2))),
            'mae_mV': float(1000 * np.mean(abs(error))),
            'coverage_95': float(np.mean(abs(error) <= Z95 * spread[mask]))}


def compare_policies(context, voltage, *, settings=Settings()) -> tuple[dict, dict[str, bytes]]:
    """Compare three schedules and score a common hidden subset as well."""
    context, voltage = np.asarray(context, dtype=float), np.asarray(voltage, dtype=float)
    if not np.isfinite(voltage).all():
        raise ValueError('The offline benchmark requires complete observed ROM voltage')
    adaptive = encode_telemetry(context, voltage, 'adaptive', settings=settings)
    uniform_count = int(np.count_nonzero(adaptive[1]['sent_mask']))
    runs = {'full': encode_telemetry(context, voltage, 'full', settings=settings),
            'uniform': encode_telemetry(context, voltage, 'uniform', settings=settings, uniform_count=uniform_count),
            'adaptive': adaptive}
    full_size = len(runs['full'][0])
    common_hidden = ~(runs['uniform'][1]['sent_mask'] | runs['adaptive'][1]['sent_mask'])
    scores = {}
    for name, (payload, edge) in runs.items():
        cloud = reconstruct_cloud(payload)
        for field in ('prior_mean_V', 'prior_sd_V', 'sent_mask'):
            if not np.array_equal(cloud[field], edge[field]):
                raise RuntimeError(f'Cloud reconstruction diverged from causal edge predictor: {name}/{field}')
        sent = cloud['sent_mask']
        if not np.array_equal(cloud['reconstructed_V'][sent], voltage[sent]):
            raise RuntimeError('Serialized observed voltage differs from its source')
        mean, spread = cloud['prior_mean_V'], cloud['prior_sd_V']
        scores[name] = {
            'observations_sent': int(sent.sum()), 'total_wire_bytes': len(payload),
            'wire_savings_percent': 100. * (1. - len(payload) / full_size),
            'voltage_observation_savings_percent': 100. * (1. - float(np.mean(sent))),
            'hidden': _metrics(voltage, mean, spread, ~sent),
            'common_hidden': _metrics(voltage, mean, spread, common_hidden),
            'all_causal_prior_predictions': _metrics(voltage, mean, spread, np.ones(len(sent), dtype=bool)),
            'full_reconstruction_rmse_mV': float(1000 * np.sqrt(np.mean((voltage - cloud['reconstructed_V'])**2))),
            'sent_indices_zero_based': np.flatnonzero(sent).tolist(),
            'selection_reasons': edge['selection_reasons'],
            'prior_mean_V': mean.tolist(), 'prior_sd_V': spread.tolist(),
            'reconstructed_V': cloud['reconstructed_V'].tolist(),
        }
    return {
        'schema': 'offline_causal_gp_voltage_scheduling_v1', 'status': 'complete',
        'scope': 'one short ROM replay of pack-equivalent voltage, simulated edge and cloud',
        'settings': asdict(settings), 'methods': scores, 'sample_count': len(voltage),
        'common_hidden_indices_zero_based': np.flatnonzero(common_hidden).tolist(),
        'known_context_columns': list(CONTEXT_COLUMNS), 'optional_voltage_column': VOLTAGE_COLUMN,
        'equal_budget': 'uniform spacing by index with count set retrospectively from the adaptive run',
        'wire_byte_basis': 'UTF-8 compact JSONL header plus all context rows and transmitted voltages, LF included',
        'interval_basis': 'nominal pointwise predictive intervals, with voltage noise included',
        'causality': 'prior voltage values are supplied only after selection; cloud forecasts use only received packets',
        'cloud_payload_reconstruction_verified': True,
        'network_upload_executed': False, 'real_cell_validation': False, 'soh_or_lifetime_inferred': False,
    }, {name: value[0] for name, value in runs.items()}


def run(replay_path: str | Path, output: str | Path) -> dict:
    context, voltage, source = load_replay(replay_path)
    report, payloads = compare_policies(context, voltage)
    target = Path(output)
    target.mkdir(parents=True, exist_ok=False)
    report['source'] = source
    report['source_code_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    for name, payload in payloads.items():
        file = target / f'cloud_{name}.jsonl'
        file.write_bytes(payload)
        report['methods'][name]['payload_file'] = file.name
        report['methods'][name]['payload_sha256'] = sha256(payload).hexdigest()
    (target / 'gp_telemetry_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', required=True, help='Completed pack replay result.json')
    parser.add_argument('--output', required=True, help='New output directory')
    args = parser.parse_args()
    report = run(args.replay, args.output)
    for name, values in report['methods'].items():
        print(f"{name}: {values['observations_sent']}/{report['sample_count']} observations, "
              f"{values['total_wire_bytes']} bytes, hidden RMSE {values['hidden']['rmse_mV']} mV")


if __name__ == '__main__':
    main()
