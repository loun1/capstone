"""Bounded acceleration of the one-step/day BLAST-Lite 1.1.0 M50 projection.

The fixed daily Ah waveform makes EFC constant. Calendar rates are sampled
using the pinned model and interpolated in SOH. Cycling keeps the model's
SOH-dependent 0.01C exclusion mask and official rate/state equations.
"""

from bisect import bisect_left
from importlib.metadata import version

import numpy as np
from scipy.interpolate import CubicSpline

from .habit_life import BLAST_VERSION, DAY_S, DEPARTURE_SOC, MAX_DAYS, TEMPERATURE_C, THRESHOLD, YEAR_DAYS


_DISCHARGE_RATE_CUTOFF = .01
_CALENDAR_RATE_CHECK_TOLERANCE = 1e-8


def _model():
    from blast.models import Nmc811_GrSi_LGM50_5Ah_Battery

    if version('blast-lite') != BLAST_VERSION or not hasattr(np, 'trapz'):
        raise ValueError('The fast projection requires BLAST-Lite 1.1.0 and NumPy with np.trapz')
    return Nmc811_GrSi_LGM50_5Ah_Battery()


def _calendar_curve(model, time, charge, temperature_kelvin):
    def calendar_rate(soh):
        soc = DEPARTURE_SOC - charge / (model.cap * soh)
        model.update_rates({'t_secs': time, 'soc': soc, 'Ua': model._get_Ua(soc),
                            'TdegK': temperature_kelvin, 'Cdis': 0.})
        return float(model.rates['k_cal'][-1])

    # Check between knots using official rates independently of the fit.
    for knot_count in (21, 41, 81):
        knots = np.linspace(THRESHOLD, 1., knot_count)
        values = [calendar_rate(soh) for soh in knots]
        curve = CubicSpline(knots, values, extrapolate=False)
        midpoints = (knots[1:] + knots[:-1]) / 2
        check_error = max(abs(float(curve(soh)) - calendar_rate(soh)) for soh in midpoints)
        if check_error <= _CALENDAR_RATE_CHECK_TOLERANCE:
            return curve, knot_count, check_error
    raise ValueError('Calendar-rate approximation exceeded checked tolerance')


def _cycling_rate_function(model, time, charge, temperature_kelvin):
    dt = np.diff(time)
    nominal_c = np.diff(charge) / (dt / 3600) / model.cap
    breakpoint_soh = np.unique(nominal_c[(nominal_c >= THRESHOLD * _DISCHARGE_RATE_CUTOFF)
                                         & (nominal_c <= _DISCHARGE_RATE_CUTOFF)] / _DISCHARGE_RATE_CUTOFF).tolist()
    # Cdis itself is nominal-C, but BLAST's active mask is defined before the
    # multiplication by SOH. Retain each distinct mask across the SOH domain.
    rates = {}
    auxiliary_time = np.array([0., DAY_S])
    auxiliary_soc = np.full(2, DEPARTURE_SOC)
    auxiliary_ua = model._get_Ua(auxiliary_soc)
    auxiliary_temp = np.full(2, temperature_kelvin[0])

    def rate(soh):
        index = bisect_left(breakpoint_soh, soh)
        if index not in rates:
            mask = nominal_c / soh >= _DISCHARGE_RATE_CUTOFF
            discharging = nominal_c[mask]
            if len(discharging) > 1:
                elapsed = np.cumsum(dt[mask])
                cdis = float(np.trapezoid(discharging, elapsed) / (elapsed[-1] - elapsed[0]))
            elif len(discharging) == 1:
                cdis = float(discharging[0])
            else:
                cdis = 0.
            model.update_rates({'t_secs': auxiliary_time, 'soc': auxiliary_soc, 'Ua': auxiliary_ua,
                                'TdegK': auxiliary_temp, 'Cdis': cdis})
            rates[index] = float(model.rates['k_cyc'][-1])
        return rates[index]

    return rate, len(breakpoint_soh)


def fast_lifetime(grid, cumulative, *, max_days=MAX_DAYS, include_history=False):
    """Return the checked one-step/day projection for a closed daily duty.

    ``crossing_day`` interpolates within the model's observed daily bracket.
    The bracket remains the primary timing resolution; results are virtual.
    """
    time = np.asarray(grid, dtype=float)
    charge = np.asarray(cumulative, dtype=float)
    if (time.ndim != 1 or len(time) < 3 or time.shape != charge.shape or time[0] != 0 or time[-1] != DAY_S
            or charge[0] != 0 or charge[-1] != 0 or np.any(np.diff(time) <= 0)
            or not np.all(np.isfinite(time)) or not np.all(np.isfinite(charge))):
        raise ValueError('Expected a finite, closed daily charge waveform on increasing times')
    if isinstance(max_days, bool) or not isinstance(max_days, int) or not 1 <= max_days <= MAX_DAYS:
        raise ValueError('max_days must be an integer inside the reviewed projection horizon')

    model = _model()
    soh_limit = DEPARTURE_SOC - charge / (model.cap * THRESHOLD)
    if soh_limit.min() < 0 or soh_limit.max() > 1:
        raise ValueError('Daily charge demand exceeds usable SOC at the evaluation threshold')
    temperature_kelvin = np.full_like(time, TEMPERATURE_C + 273.15)
    calendar_rate, knot_count, rate_error = _calendar_curve(model, time, charge, temperature_kelvin)
    cycling_rate, n_breakpoints = _cycling_rate_function(model, time, charge, temperature_kelvin)
    daily_efc = float(np.abs(np.diff(charge)).sum() / (2 * model.cap))
    dt_calendar, dt_cycle = 1e-4, daily_efc / 1e4
    parameters = model._params_life
    power_calendar, power_cycle = parameters['p_cal'], parameters['p_cyc']
    advance = model._update_power_state
    q, loss_calendar, loss_cycle = 1., 0., 0.
    crossing, crossing_day, five_year_soh = None, None, None
    if include_history:
        history = [{'day': 0, 'soh_pct': 100., 'calendar_loss_pp': 0., 'cycle_loss_pp': 0., 'efc': 0.}]

    for day in range(1, max_days + 1):
        previous = q
        loss_calendar += advance(loss_calendar, dt_calendar, float(calendar_rate(q)), power_calendar)
        loss_cycle += advance(loss_cycle, dt_cycle, cycling_rate(q), power_cycle)
        q = 1 - loss_calendar - loss_cycle
        if not 0 < q <= previous:
            raise ValueError('Projection did not produce a positive monotone SOH')
        row = {'day': day, 'soh_pct': 100 * q, 'calendar_loss_pp': 100 * loss_calendar,
               'cycle_loss_pp': 100 * loss_cycle, 'efc': day * daily_efc}
        if include_history:
            history.append(row)
        if day == 5 * YEAR_DAYS:
            five_year_soh = row['soh_pct']
        if q <= THRESHOLD:
            crossing = [day - 1, day]
            crossing_day = day - 1 + (previous - THRESHOLD) / (previous - q)
            break

    result = {'crossing_day': crossing_day, 'soh80_crossing_day_bracket': crossing, 'right_censored': crossing is None,
              'daily_efc': daily_efc, 'steps_per_day': 1, 'end': row, 'soh_pct_at_5y': five_year_soh,
              'numerics': {'calendar_rate_method': 'cubic interpolation of pinned official rates in SOH',
                           'calendar_rate_knots': knot_count, 'calendar_midpoint_max_abs_error': rate_error,
                           'calendar_midpoint_error_limit': _CALENDAR_RATE_CHECK_TOLERANCE,
                           'cycling_threshold_breakpoint_count': n_breakpoints}}
    if include_history:
        result['daily_history'] = history
    return result
