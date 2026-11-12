"""Check that hidden ROM values never reach the scheduler or cloud."""
import json
import unittest

import numpy as np

from .gp_telemetry import compare_policies, encode_telemetry, reconstruct_cloud


class GPTelemetryTests(unittest.TestCase):
    def setUp(self):
        time = np.arange(80.)
        current = .2 + .1 * np.sin(time / 15.)
        model = 4.0 - .05 * current
        self.context = np.column_stack([time, current, np.full(len(time), 20.), model])
        self.voltage = model + .003 * np.cos(time / 17.)

    def test_hiding_every_skipped_reading_preserves_payload_and_forecast(self):
        first, state = encode_telemetry(self.context, self.voltage, 'adaptive')
        hidden = ~state['sent_mask']
        self.assertTrue(np.any(hidden))
        unavailable = np.where(hidden, np.nan, self.voltage)
        second, second_state = encode_telemetry(self.context, unavailable, 'adaptive')
        self.assertEqual(first, second)
        np.testing.assert_array_equal(state['prior_mean_V'], second_state['prior_mean_V'])
        np.testing.assert_array_equal(state['sent_mask'], second_state['sent_mask'])

    def test_voltage_is_seen_after_the_transmission_decision(self):
        _, state = encode_telemetry(self.context, self.voltage, 'adaptive')
        index = np.flatnonzero(state['sent_mask'])[2]
        modified = self.voltage.copy()
        modified[index] += .12
        _, changed = encode_telemetry(self.context, modified, 'adaptive')
        np.testing.assert_array_equal(state['prior_mean_V'][:index + 1], changed['prior_mean_V'][:index + 1])
        np.testing.assert_array_equal(state['sent_mask'][:index + 1], changed['sent_mask'][:index + 1])

    def test_equal_count_and_independent_wire_reconstruction(self):
        report, payloads = compare_policies(self.context, self.voltage)
        methods = report['methods']
        self.assertEqual(methods['uniform']['observations_sent'], methods['adaptive']['observations_sent'])
        self.assertEqual(methods['full']['full_reconstruction_rmse_mV'], 0.)
        for name, payload in payloads.items():
            self.assertEqual(methods[name]['total_wire_bytes'], len(payload))
            actual = reconstruct_cloud(payload)
            np.testing.assert_array_equal(actual['reconstructed_V'], methods[name]['reconstructed_V'])
            rows = [json.loads(line) for line in payload.splitlines()[1:]]
            self.assertEqual(sum(len(row) == 5 for row in rows), methods[name]['observations_sent'])


if __name__ == '__main__':
    unittest.main()
