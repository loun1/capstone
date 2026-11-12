"""Independent, bounded lifetime engine using public PyBaMM APIs.

This is an uncalibrated virtual-cell engine, never an experimental truth source.
No import of battery.py, old controllers, or solver monkey patches is permitted.
Operating SOC and diagnostic capacity are deliberately separate measurements.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import time

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
os.environ.setdefault('PYBAMM_DISABLE_TELEMETRY', 'true')
import numpy as np
import pybamm

ROOT = Path(__file__).resolve().parents[1]
FORMAT = 'independent-lifetime-v2'
F = float(pybamm.constants.F.evaluate())
STO = 'Average negative particle stoichiometry'
RAW_VARIABLES = ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]')


class StepAuditError(ValueError):
    def __init__(self, message, details, raw):
        super().__init__(message)
        self.details, self.raw = details, raw


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def scalar(sol, name, index=-1):
    a = np.asarray(sol[name].entries, dtype=float)
    value = float(a.reshape(-1, a.shape[-1])[0, index])
    if not math.isfinite(value):
        raise ValueError('Nonfinite variable: ' + name)
    return value


def fingerprint(sol):
    h = hashlib.sha256()
    for t, y in zip(sol.all_ts, sol.all_ys):
        h.update(np.asarray(t, dtype=float).tobytes())
        h.update(np.asarray(y, dtype=float).tobytes())
    return h.hexdigest()


class Engine:
    def __init__(self, *, family='SPMe', mesh=1, rtol=1e-5, atol=1e-7,
                 temperature_K=298.15, overrides=None, physics='baseline',
                 reference_output_only=True):
        if family not in ('SPMe', 'DFN') or mesh not in (1, 2, 4, 8):
            raise ValueError('Unsupported model/mesh')
        if not all(math.isfinite(v) and v > 0 for v in (rtol, atol, temperature_K)):
            raise ValueError('Tolerances and temperature must be finite and positive')
        cfg = json.loads((ROOT/'configs/model.json').read_text(encoding='utf-8'))
        self.diag = json.loads((ROOT/'configs/diagnostics.json').read_text(encoding='utf-8'))['reference_test']
        # Versioned controller is intentionally restricted to the reviewed protocol.
        # Fail on an incompatible config instead of silently running hardcoded defaults.
        for actual, expected in ((cfg['voltage_max_V'], 4.2), (cfg['voltage_min_V'], 2.5),
                                 (cfg['soc_window'], [.2, .8]), (cfg['rest_s'], 300),
                                 (cfg['currents_A']['discharge'], 5.)):
            if actual != expected:
                raise ValueError('Controller protocol differs from reviewed configuration')
        self.options = cfg['options'].copy()
        if physics == 'coupled':
            self.options.update({'particle mechanics': ('swelling and cracking', 'swelling only'),
                                 'SEI on cracks': 'true', 'loss of active material': 'stress-driven'})
        elif physics != 'baseline':
            raise ValueError('Unknown physics profile')
        self.physics = physics
        self.family, self.mesh, self.rtol, self.atol = family, mesh, rtol, atol
        self.param = pybamm.ParameterValues(cfg['parameter_set'])
        self.param.update({'Ambient temperature [K]': temperature_K,
                           'Initial temperature [K]': temperature_K})
        self.param.update(overrides or {})
        self.temperature_K = temperature_K
        self.overrides = overrides or {}
        self.reference_output_only = reference_output_only
        self.sims = {}
        self.esoh = pybamm.lithium_ion.ElectrodeSOHSolver(
            self.param, param=self.model().param, options=self.model().options)

    def model(self):
        cls = getattr(pybamm.lithium_ion, self.family)
        return cls(options=copy.deepcopy(self.options))

    def sim(self, key, steps):
        if key not in self.sims:
            m = self.model()
            pts = {k: int(v*self.mesh) for k, v in m.default_var_pts.items()}
            if self.physics == 'coupled':
                for k in ('r_n', 'r_p'):
                    pts[k] = max(30, pts[k])
            diagnostic = isinstance(key, tuple) and key[0] == 'reference'
            # IDAKLU retains the full terminal state between experiment phases.
            # RPT needs only these observations, not every spatial state at 1 Hz.
            outputs = list(RAW_VARIABLES) if diagnostic and self.reference_output_only else []
            self.sims[key] = pybamm.Simulation(m, parameter_values=self.param.copy(),
                experiment=pybamm.Experiment([tuple(steps)]), var_pts=pts,
                solver=pybamm.IDAKLUSolver(rtol=self.rtol, atol=self.atol,
                                         output_variables=outputs))
        return self.sims[key]

    def start(self):
        return self.sim('initial', [pybamm.step.rest(duration=1)]).solve(
            initial_soc=0.2, calc_esoh=False).last_state.copy()

    def bounds(self, sol):
        args = {'Q_n': scalar(sol, 'Negative electrode capacity [A.h]'),
                'Q_p': scalar(sol, 'Positive electrode capacity [A.h]'),
                'Q_Li': scalar(sol, 'Total lithium in particles [mol]')*F/3600}
        r = self.esoh.solve(args)
        x0, x100 = (float(np.ravel(r[n])[0]) for n in ('x_0', 'x_100'))
        if not (math.isfinite(x0) and math.isfinite(x100) and 0 <= x0 < x100 <= 1):
            raise ValueError('Invalid eSOH boundaries')
        return x0, x100

    def soc(self, sol, bounds=None):
        lo, hi = bounds if bounds is not None else self.bounds(sol)
        return (scalar(sol, STO)-lo)/(hi-lo)

    def readout(self, sol):
        names = {
            't_s': 'Time [s]', 'V': 'Voltage [V]', 'I_A': 'Current [A]',
            'Q_signed_Ah': 'Discharge capacity [A.h]',
            'SEI_m': 'X-averaged negative SEI thickness [m]',
            'SEI_loss_Ah': 'Loss of capacity to negative SEI [A.h]',
            'crack_SEI_loss_Ah': 'Loss of capacity to negative SEI on cracks [A.h]',
            'plating_loss_Ah': 'Loss of capacity to negative lithium plating [A.h]',
            'dead_Li_mol_m3': 'X-averaged negative dead lithium concentration [mol.m-3]',
            'plated_Li_mol_m3': 'X-averaged negative lithium plating concentration [mol.m-3]',
            'Li_total_mol': 'Total lithium [mol]',
            'Li_side_loss_mol': 'Total lithium lost to side reactions [mol]',
            'LLI_percent': 'Loss of lithium inventory [%]',
            'neg_porosity': 'X-averaged negative electrode porosity',
            'temperature_K': 'X-averaged cell temperature [K]',
            'LAM_neg_percent': 'Loss of active material in negative electrode [%]',
            'LAM_pos_percent': 'Loss of active material in positive electrode [%]'}
        return {k: scalar(sol, v) for k, v in names.items()}

    def advance(self, sol, phase, duration_s, current_A, bounds):
        """Use duration directly; no large absolute-clock termination expression."""
        if not math.isfinite(duration_s) or duration_s <= 1e-7:
            raise ValueError('No positive integration interval')
        charge = phase in ('CC', 'CV')
        target = bounds[0] + (0.8 if charge else 0.2)*(bounds[1]-bounds[0])
        term = pybamm.step.CustomTermination('operating SOC',
            (lambda v: pybamm.InputParameter('target')-v[STO]) if charge else
            (lambda v: v[STO]-pybamm.InputParameter('target')))
        kw = {'duration': duration_s, 'period': 1}
        if phase == 'CC':
            step = pybamm.step.current(-current_A, termination=['4.2 V', term], **kw)
        elif phase == 'CV':
            step = pybamm.step.voltage(4.2, termination=[term], **kw)
        elif phase == 'DIS':
            step = pybamm.step.current(5., termination=['2.5 V', term], **kw)
        else:
            step = pybamm.step.rest(**kw)
        # Partial duration simulations are transient, so diagnostic targets do not
        # grow a cache unboundedly over a full lifetime.
        key = (phase, duration_s, current_A)
        sim = self.sim(key, [step])
        result = sim.solve(starting_solution=sol.copy(), calc_esoh=False,
            inputs={'target': target} if phase in ('CC', 'CV', 'DIS') else {})
        st = result.cycles[-1].steps[-1]
        if duration_s != 60.:
            self.sims.pop(key, None)
        if not np.all(np.isfinite(np.asarray(st.all_ys[-1]))):
            raise ValueError('Nonfinite differential/algebraic state')
        t = np.asarray(st['Time [s]'].entries)
        if len(t) < 2 or not np.all(np.diff(t) > 0):
            raise ValueError('Non-monotonic or zero-length output')
        i = np.asarray(st['Current [A]'].entries)
        q = np.asarray(st['Discharge capacity [A.h]'].entries)
        dq = float(q[-1]-q[0])
        dq_int = float(np.trapezoid(i, t)/3600.)
        raw = {n: np.asarray(st[n].entries) for n in RAW_VARIABLES}
        if abs(dq-dq_int) > 1e-7 + 1e-4*abs(dq):
            raise StepAuditError('Current integral and cumulative capacity disagree',
                {'phase': phase, 't_start_s': float(t[0]), 't_end_s': float(t[-1]),
                 'Q_cumulative_Ah': dq, 'Q_integral_Ah': dq_int,
                 'absolute_difference_Ah': abs(dq-dq_int),
                 'limit_Ah': 1e-7+1e-4*abs(dq), 'rtol': self.rtol, 'atol': self.atol,
                 'termination': str(st.termination)}, raw)
        if phase == 'DIS' and (dq <= 0 or np.min(i) < 0):
            raise ValueError('Discharge direction/throughput error')
        return result.last_state.copy(), st, dq, raw

    def reference(self, main, *, rest_s=None):
        before = fingerprint(main)
        r = self.diag
        rest_s = r['rest_s'] if rest_s is None else rest_s
        period = r['output_period_s']
        steps = [pybamm.step.current(-r['charge_current_A'], duration=21600,
                                    termination=f"{r['charge_voltage_V']} V", period=period),
                 pybamm.step.voltage(r['charge_voltage_V'], duration=21600,
                                    termination=f"{r['cv_cutoff_current_A']} A", period=period),
                 pybamm.step.rest(duration=rest_s, period=period),
                 pybamm.step.current(r['discharge_current_A'], duration=86400,
                                    termination=f"{r['discharge_cutoff_V']} V", period=period)]
        result = self.sim(('reference', rest_s), steps).solve(
            starting_solution=main.copy(), calc_esoh=False)
        pieces = result.cycles[-1].steps
        if len(pieces) != 4:
            raise ValueError('Incomplete reference protocol')
        terms = [str(s.termination) for s in pieces]
        expected_terms = [f"Voltage > {r['charge_voltage_V']} [V]",
                          f"abs(Current [A]) < {r['cv_cutoff_current_A']} [A]",
                          'final time', f"Voltage < {r['discharge_cutoff_V']} [V]"]
        if any(expected not in actual for expected, actual in zip(expected_terms, terms)):
            raise ValueError('Reference ended without prescribed events: '+repr(terms))
        for piece in pieces:
            for name in ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]'):
                if not np.all(np.isfinite(np.asarray(piece[name].entries))):
                    raise ValueError('Nonfinite reference output: '+name)
            times = np.asarray(piece['Time [s]'].entries)
            if len(times) < 2 or not np.all(np.diff(times) > 0):
                raise ValueError('Invalid reference time grid')
            if np.max(np.diff(times)) > period + 1e-7:
                raise ValueError('Reference output period not honored')
        rest_times = np.asarray(pieces[2]['Time [s]'].entries)
        if abs(float(rest_times[-1]-rest_times[0])-rest_s) > 1e-6:
            raise ValueError('Reference rest duration incomplete')
        dis = pieces[-1]
        q = np.asarray(dis['Discharge capacity [A.h]'].entries)
        t = np.asarray(dis['Time [s]'].entries)
        i = np.asarray(dis['Current [A]'].entries)
        capacity, integral = float(q[-1]-q[0]), float(np.trapezoid(i, t)/3600)
        if not (capacity > 0 and abs(capacity-integral) < 1e-6):
            raise ValueError('Invalid reference integral')
        if before != fingerprint(main):
            raise ValueError('Reference mutated main trajectory')
        raw = {f'{j}_{n}': np.asarray(s[n].entries) for j, s in enumerate(pieces)
               for n in ('Time [s]', 'Current [A]', 'Voltage [V]', 'Discharge capacity [A.h]')}
        return {'Qref_Ah': capacity, 'integral_Ah': integral, 'terminations': terms,
                'main_unchanged': True, 'diagnostic_ages_its_own_copy': True}, raw


def identity(engine, args):
    data = {'format': FORMAT, 'code_sha256': sha(__file__),
        'config_sha256': {n: sha(ROOT/'configs'/n) for n in
                          ('model.json', 'diagnostics.json', 'cells_pilot9.json')},
        'family': engine.family, 'mesh': engine.mesh, 'rtol': engine.rtol, 'atol': engine.atol,
        'physics': engine.physics, 'options': engine.options, 'temperature_K': engine.temperature_K,
        'overrides': engine.overrides, 'current_A': args.current,
        'boundary_s': args.boundary_s, 'diagnostic_interval_EFC': args.rpt_every,
        'initial_solver': {'rtol': args.initial_rtol, 'atol': args.initial_atol},
        'reference_solver': {'rtol': args.rpt_rtol, 'atol': args.rpt_atol},
        'reference_output_only': True,
        'EFC_reference_Ah_override': args.efc_reference_ah,
        'python': platform.python_version(), 'platform': platform.system(),
        'versions': {n: importlib.metadata.version(n) for n in
                     ('pybamm', 'pybammsolvers', 'numpy', 'casadi')},
        'experimental_validation': False, 'label_status': 'uncalibrated_virtual_cell'}
    return json.loads(json.dumps(data, allow_nan=False))


def run(args):
    import casadi
    import pickle
    # Defaults keep old programmatic callers usable; the v2 identity prevents
    # resuming a v1 checkpoint with different initialization or measurement rules.
    for key, value in {'rtol': 1e-5, 'atol': 1e-7, 'initial_rtol': 1e-8,
                       'initial_atol': 1e-10, 'rpt_rtol': 1e-8, 'rpt_atol': 1e-10,
                       'efc_reference_ah': None}.items():
        if not hasattr(args, key):
            setattr(args, key, value)
    out = Path(args.output)
    if not (0 < args.max_efc <= 3000 and 0 < args.wall_s <= 43200 and
            0 < args.boundary_s <= 60 and args.rpt_every > 0):
        raise ValueError('Explicit bounded run required')
    if not args.resume and out.exists():
        raise ValueError('New run output must not exist')
    if args.efc_reference_ah is not None and not (
            math.isfinite(args.efc_reference_ah) and args.efc_reference_ah > 0):
        raise ValueError('Invalid common throughput reference')
    engine = Engine(family=args.family, mesh=args.mesh, physics=args.physics,
                    rtol=args.rtol, atol=args.atol)
    initial_engine = Engine(family=args.family, mesh=args.mesh, physics=args.physics,
                           rtol=args.initial_rtol, atol=args.initial_atol)
    reference_engine = Engine(family=args.family, mesh=args.mesh, physics=args.physics,
                             rtol=args.rpt_rtol, atol=args.rpt_atol)
    ident = identity(engine, args)
    began = time.monotonic()
    interrupted = [False]
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: interrupted.__setitem__(0, True))
    if args.resume:
        # Only this engine's trusted local checkpoint is accepted. No old Linux
        # checkpoint is migrated to a different OS/model/config.
        commit = json.loads((out/'commit.json').read_text(encoding='utf-8'))
        if commit['identity'] != ident or sha(out/commit['checkpoint']) != commit['sha256']:
            raise ValueError('Checkpoint identity/hash mismatch')
        with casadi.global_unpickle_context(), (out/commit['checkpoint']).open('rb') as f:
            state = pickle.load(f)
        if state['identity'] != ident or state['status'] not in ('ready', 'wall_limit', 'interrupted'):
            raise ValueError('Checkpoint is not a resumable healthy state')
    else:
        out.mkdir(parents=True)
        (out/'raw').mkdir()
        sol = initial_engine.start()
        rpt, raw = reference_engine.reference(sol)
        np.savez_compressed(out/'raw/rpt_000000.npz', **raw)
        state = {'identity': ident, 'solution': sol, 'Q0_Ah': rpt['Qref_Ah'],
                 'initial_state_fingerprint': fingerprint(sol),
                 'EFC_reference_Ah': (args.efc_reference_ah if args.efc_reference_ah is not None
                                      else rpt['Qref_Ah']),
                 'EFC_definition': ('common_reference_discharge_Ah_for_convergence_audit'
                                    if args.efc_reference_ah is not None else 'discharge_Ah/fixed_Q0_Ah'),
                 'discharge_Ah': 0., 'charge_Ah': 0., 'efc': 0., 'phase': 'CC',
                 'rest_until_s': None, 'charge_start_s': scalar(sol, 'Time [s]'),
                 'cycles': 0, 'pieces': 0, 'next_rpt_EFC': 5., 'elapsed_wall_s': 0.,
                 'status': 'ready', 'rpts': [{'efc': 0., 'SOH_percent': 100., **rpt}],
                 'initial_inventory_mol': sum(engine.readout(sol)[k] for k in
                     ('Li_total_mol', 'Li_side_loss_mol')), 'charge_violations': 0}

    prior_wall = state['elapsed_wall_s']
    base_piece = state['pieces']
    def checkpoint():
        # Three checkpoint slots + one atomic commit pointer. Two previous slots
        # survive; raw observations are retained. Do not grow checkpoints forever.
        state['elapsed_wall_s'] = prior_wall+time.monotonic()-began
        state['checkpoint_generation'] = state.get('checkpoint_generation', -1)+1
        name = f"checkpoint_{state['checkpoint_generation']%3}.pkl"
        tmp = out/(name+'.tmp')
        with casadi.global_pickle_context(), tmp.open('wb') as handle:
            pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush(); os.fsync(handle.fileno())
        tmp.replace(out/name)
        write_json(out/'commit.json', {'identity': ident, 'checkpoint': name, 'sha256': sha(out/name)})
        write_json(out/'summary.json', {k: v for k, v in state.items() if k != 'solution'})

    checkpoint()
    try:
        while True:
            if interrupted[0]:
                state['status'] = 'interrupted'; break
            if prior_wall+time.monotonic()-began >= args.wall_s:
                state['status'] = 'wall_limit'; break
            if state['efc'] >= args.max_efc-1e-9:
                state['status'] = 'max_efc_censored'; break
            if args.max_pieces and state['pieces']-base_piece >= args.max_pieces:
                state['status'] = 'ready'; break
            sol, phase = state['solution'], state['phase']
            bounds = engine.bounds(sol)
            soc = engine.soc(sol, bounds)
            t0 = scalar(sol, 'Time [s]')
            def finish_phase():
                if phase in ('CC', 'CV', 'DIS'):
                    target = .2 if phase == 'DIS' else .8
                    if abs(engine.soc(state['solution'])-target) > .001:
                        raise ValueError('Operating SOC target missed')
                if phase in ('CC', 'CV'):
                    if scalar(state['solution'], 'Time [s]')-state['charge_start_s'] > 3600:
                        state['charge_violations'] += 1
                    state['phase'] = 'REST_CHARGE'
                    state['rest_until_s'] = scalar(state['solution'], 'Time [s]')+300
                elif phase == 'DIS':
                    state['phase'] = 'REST_DISCHARGE'
                    state['rest_until_s'] = scalar(state['solution'], 'Time [s]')+300
                elif phase == 'REST_CHARGE':
                    state['phase'] = 'DIS'
                else:
                    state['phase'] = 'CC'; state['cycles'] += 1
                    state['charge_start_s'] = scalar(state['solution'], 'Time [s]')
            if ((phase in ('CC', 'CV') and soc >= .8) or (phase == 'DIS' and soc <= .2) or
                    (phase.startswith('REST') and t0 >= state['rest_until_s']-1e-7)):
                finish_phase(); continue
            duration = float(args.boundary_s)
            if phase.startswith('REST'):
                duration = min(duration, state['rest_until_s']-t0)
            if phase == 'DIS':
                target = min(args.max_efc, state['next_rpt_EFC'])
                duration = min(duration, (target*state['EFC_reference_Ah']-state['discharge_Ah'])*720)
            next_sol, step, dq, raw = engine.advance(sol, phase, duration, args.current, bounds)
            values = engine.readout(next_sol)
            for variable in ('Negative particle concentration [mol.m-3]',
                             'Positive particle concentration [mol.m-3]', 'Electrolyte concentration [mol.m-3]',
                             'Negative electrode porosity', 'Positive electrode porosity',
                             'X-averaged negative SEI thickness [m]'):
                a = np.asarray(step[variable].entries, dtype=float)
                if not np.all(np.isfinite(a)) or np.min(a) < 0:
                    raise ValueError('Nonphysical spatial field: '+variable)
                if 'porosity' in variable and (np.min(a) <= 0 or np.max(a) >= 1):
                    raise ValueError('Spatial porosity outside (0,1)')
            if abs(values['t_s']-t0-float(step.t[-1]-step.t[0])) > 1e-6:
                raise ValueError('Time connection changed')
            residual = values['Li_total_mol']+values['Li_side_loss_mol']-state['initial_inventory_mol']
            if abs(residual) > 1e-7*state['initial_inventory_mol']:
                raise ValueError('Lithium conservation failed')
            if not (0 < values['neg_porosity'] < 1 and values['SEI_m'] >= 0):
                raise ValueError('Nonphysical porosity/SEI')
            if abs(values['temperature_K']-engine.temperature_K) > 1e-6:
                raise ValueError('Isothermal temperature changed')
            if phase == 'DIS':
                state['discharge_Ah'] += dq
            elif phase in ('CC', 'CV'):
                state['charge_Ah'] -= dq
            elif abs(dq) > 1e-8:
                raise ValueError('Rest added throughput')
            state.update(solution=next_sol, efc=state['discharge_Ah']/state['EFC_reference_Ah'],
                         pieces=state['pieces']+1, latest_physics=values,
                         inventory_residual_mol=residual)
            raw_name = f"piece_{state['pieces']:09d}.npz"
            raw_path = out/'raw'/raw_name
            if raw_path.exists():
                # Preserve orphan rather than overwrite evidence after interruption.
                raw_name = f"piece_{state['pieces']:09d}_{time.time_ns()}.npz"
            np.savez_compressed(out/'raw'/raw_name, **raw)
            term = str(step.termination)
            if 'operating SOC' in term:
                finish_phase()
            elif phase == 'CC' and 'Voltage' in term:
                state['phase'] = 'CV'
            elif phase == 'DIS' and 'Voltage' in term:
                state['status'] = 'operating_condition_failed'; break
            elif phase == 'CV' and abs(values['I_A']) < .001:
                state['status'] = 'operating_condition_failed'; break
            if state['efc'] >= min(args.max_efc, state['next_rpt_EFC'])-1e-8:
                rpt, raw = reference_engine.reference(next_sol)
                number = len(state['rpts'])
                path = out/'raw'/f'rpt_{number:06d}_{time.time_ns()}.npz'
                np.savez_compressed(path, **raw)
                point = {'efc': state['efc'], 'SOH_percent': 100*rpt['Qref_Ah']/state['Q0_Ah'],
                         'raw': path.name, 'sha256': sha(path), **rpt}
                state['rpts'].append(point)
                if point['SOH_percent'] < 80:
                    state['status'] = 'virtual_eol_observed'; break
                state['next_rpt_EFC'] = (10. if state['next_rpt_EFC'] == 5. else
                                         state['next_rpt_EFC']+args.rpt_every)
            if state['pieces'] % 50 == 0:
                checkpoint()
    except BaseException as exc:
        state['status'] = 'failed'
        state['error'] = repr(exc)
        if isinstance(exc, StepAuditError):
            raw_path = out/'raw'/f'failed_step_{time.time_ns()}.npz'
            np.savez_compressed(raw_path, **exc.raw)
            state['failed_step'] = {**exc.details, 'raw': raw_path.name,
                                    'sha256': sha(raw_path)}
        checkpoint()
        raise
    checkpoint()
    return {k: v for k, v in state.items() if k != 'solution'}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--family', choices=('SPMe', 'DFN'), default='SPMe')
    p.add_argument('--physics', choices=('baseline', 'coupled'), default='baseline')
    p.add_argument('--mesh', type=int, choices=(1, 2, 4, 8), default=1)
    p.add_argument('--current', type=float, choices=(3., 3.1, 3.2, 3.3), default=3.2)
    p.add_argument('--max-efc', type=float, required=True)
    p.add_argument('--wall-s', type=float, required=True)
    p.add_argument('--boundary-s', type=float, default=60.)
    p.add_argument('--rpt-every', type=float, default=10.)
    p.add_argument('--max-pieces', type=int, default=0)
    p.add_argument('--rtol', type=float, default=1e-5)
    p.add_argument('--atol', type=float, default=1e-7)
    p.add_argument('--initial-rtol', type=float, default=1e-8)
    p.add_argument('--initial-atol', type=float, default=1e-10)
    p.add_argument('--rpt-rtol', type=float, default=1e-8)
    p.add_argument('--rpt-atol', type=float, default=1e-10)
    p.add_argument('--efc-reference-ah', type=float,
                   help='Audit only: common physical throughput denominator; otherwise fixed Q0')
    return p


def main():
    args = parser().parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ('status', 'efc', 'Q0_Ah', 'rpts')}, indent=2))
    return 0 if result['status'] in ('ready', 'max_efc_censored', 'virtual_eol_observed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
