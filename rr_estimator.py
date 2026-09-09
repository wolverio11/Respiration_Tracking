"""
rr_estimator.py
===============
Practical respiration-rate estimator for the PLIM pipeline.

This version uses real timestamps instead of assuming a fixed frame rate.
That makes BPM much more reliable when the live loop does not actually run
at 30 fps.
"""

from collections import deque
import statistics
import time

from config import MIN_PRANAYAMA_BPM, MAX_PRANAYAMA_BPM


class RREstimator:
    def __init__(self):
        self.signal_history = deque(maxlen=300)
        self.time_history = deque(maxlen=300)
        self.peak_times = deque(maxlen=8)

        self.smoothed_signal = 0.0
        self.rf = 0.0
        self.rho = 0          # 1 while inside active breath peak region
        self.beta = 0         # cumulative detected breath count
        self.upsilon = 0.0    # dynamic signal scale
        self.zeta_u = 0.0     # upper threshold
        self.zeta_l = 0.0     # lower threshold
        self.tau = 0.0        # last inter-breath interval (s)

        self._last_peak_time = 0.0
        self._last_valley_time = 0.0

    def update(self, ef_prime: int, timestamp: float | None = None) -> dict:
        if timestamp is None:
            timestamp = time.time()

        x = float(ef_prime)

        # Exponential smoothing to suppress frame-level jitter.
        alpha = 0.20
        self.smoothed_signal = (alpha * x) + ((1.0 - alpha) * self.smoothed_signal)

        self.signal_history.append(self.smoothed_signal)
        self.time_history.append(timestamp)

        if len(self.signal_history) < 20:
            return self._get_state()

        recent = list(self.signal_history)
        baseline = statistics.median(recent)
        dynamic_span = max(recent) - min(recent)

        # Dynamic thresholds. These are intentionally conservative.
        self.upsilon = max(1.0, dynamic_span)
        self.zeta_u = baseline + max(2.0, 0.35 * self.upsilon)
        self.zeta_l = baseline + max(1.0, 0.15 * self.upsilon)

        current = self.signal_history[-1]
        prev = self.signal_history[-2]
        now = self.time_history[-1]

        # Plausibility guard: avoid counting multiple peaks inside one breath.
        min_interval_s = 60.0 / max(MAX_PRANAYAMA_BPM, 1)
        max_interval_s = 60.0 / max(MIN_PRANAYAMA_BPM, 1)

        if self.rho == 0:
            # Enter peak state only on a meaningful upward crossing.
            if current >= self.zeta_u and current > prev:
                if (now - self._last_peak_time) >= min_interval_s:
                    self.rho = 1
                    self._last_peak_time = now
                    self.peak_times.append(now)
                    self.beta += 1
        else:
            # Leave peak state once the signal relaxes below the lower threshold.
            if current <= self.zeta_l:
                self.rho = 0
                self._last_valley_time = now

        # Estimate RR from recent accepted peak times.
        if len(self.peak_times) >= 2:
            intervals = [
                self.peak_times[i] - self.peak_times[i - 1]
                for i in range(1, len(self.peak_times))
            ]
            valid_intervals = [dt for dt in intervals if min_interval_s <= dt <= max_interval_s]

            if valid_intervals:
                self.tau = valid_intervals[-1]
                mean_interval = sum(valid_intervals[-3:]) / min(len(valid_intervals), 3)
                if mean_interval > 0:
                    self.rf = 60.0 / mean_interval
                    self.rf = max(float(MIN_PRANAYAMA_BPM), min(float(MAX_PRANAYAMA_BPM), self.rf))

        # If the signal has become nearly flat for a while, drop RR to zero.
        if self.upsilon < 2.0:
            if len(self.time_history) >= 2 and (self.time_history[-1] - self.time_history[0]) >= 3.0:
                self.rf = 0.0

        return self._get_state()

    def _get_state(self) -> dict:
        return {
            "rf": round(self.rf, 1),
            "upsilon": float(round(self.upsilon, 2)),
            "zeta_u": float(round(self.zeta_u, 2)),
            "zeta_l": float(round(self.zeta_l, 2)),
            "rho": int(self.rho),
            "beta": int(self.beta),
            "tau": float(round(self.tau, 3)),
            "af": 0,
            "bf": 0,
        }

    def reset(self) -> None:
        self.signal_history.clear()
        self.time_history.clear()
        self.peak_times.clear()
        self.smoothed_signal = 0.0
        self.rf = 0.0
        self.rho = 0
        self.beta = 0
        self.upsilon = 0.0
        self.zeta_u = 0.0
        self.zeta_l = 0.0
        self.tau = 0.0
        self._last_peak_time = 0.0
        self._last_valley_time = 0.0
