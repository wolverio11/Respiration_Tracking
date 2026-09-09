"""
change_monitor.py
=================
Implements Section 2.5 of Lee et al. (2021):
"Monitor change in respiration rate" — Equations 23 to 28.

WHAT THIS MODULE DOES:
    Two jobs:

    JOB 1 — RATE OF CHANGE MONITORING (Equations 23-26):
        Every 10 seconds, compares current RR to RR from 10 seconds ago.
        If RR changed significantly → generate an alert.
        Three alert levels: minor (>1%), moderate (>25%), critical (>50%).

    JOB 2 — RESPIRATORY ARREST / KUMBHAKA DETECTION (Equations 27-28):
        If RR = 0 for a prolonged period → detect as no-breathing event.
        The waiting time is ADAPTIVE — it grows each time arrest is
        detected, up to a maximum of 30 seconds.
        This prevents constant false alarms for slow breathers.

PAPER EQUATIONS IMPLEMENTED:
    Equation 23: Alf — fixed threshold alarm (upper/lower RR limits)
    Equation 24: Δrf = |rf − rf−k|
    Equation 25: δrf = Δrf / rf × 100
    Equation 26: Alc — change classification (minor/moderate/critical)
    Equation 27: Aln — no-breathing alarm flag
    Equation 28: qf — adaptive inactivity threshold

PRANAYAMA CONTEXT:
    For pranayama, "respiratory arrest" = KUMBHAKA (breath retention).
    This is intentional and healthy. The system detects it correctly
    using the same mechanism as arrest detection — it just has a
    different meaning in the pranayama context.

    The adaptive qf threshold is very helpful here:
    When you do repeated Kumbhaka holds, the system learns to wait
    longer before raising an alarm, accommodating extended holds.
"""

from collections import deque
from config import (
    FRAME_RATE,
    CHANGE_CALC_FRAMES,
    ALERT_MINOR_THRESH,
    ALERT_MODERATE_THRESH,
    INACTIVITY_INITIAL_FRAMES,
    INACTIVITY_MAX_FRAMES,
    MIN_PRANAYAMA_BPM,
    MAX_PRANAYAMA_BPM,
)


class ChangeMonitor:
    """
    Monitors RR changes and detects respiratory arrest / Kumbhaka.

    Internal variables:
        rr_history          — rolling buffer of rf values (last 10 sec)
        alert_level         — current alert: "none","minor","moderate","critical"
        aln                 — Aln: no-breathing alarm (0 or 1)
        q_f                 — qf: adaptive inactivity threshold (frames)
        consecutive_zero    — frames with rf=0 continuously
        is_kumbhaka         — True when no-breathing detected
        kumbhaka_duration   — seconds of current Kumbhaka hold
    """

    def __init__(self):
        # Rolling buffer of rf values.
        # We store 2 × CHANGE_CALC_FRAMES (20 seconds) so that when
        # we compare current RF to RF from 10s ago, the older value
        # is still in the buffer at index [0].
        # (With only 1× capacity, old values are overwritten before
        # the comparison fires, causing delta = 0 incorrectly.)
        self._rr_history: deque = deque(maxlen=CHANGE_CALC_FRAMES * 2)

        # Alert state (Equation 26)
        self.alert_level: str = "none"

        # Fixed threshold alarm state (Equation 23)
        # Upper and lower RR limits for fixed alarm
        # [PRANAYAMA] Normal pranayama range: 2-60 bpm
        self._upper_fixed_limit: float = MAX_PRANAYAMA_BPM   # εu
        self._lower_fixed_limit: float = MIN_PRANAYAMA_BPM   # εl
        self.fixed_alarm: str = "none"   # "upper", "lower", "none"

        # Respiratory arrest / Kumbhaka (Equations 27-28)
        self.aln: int            = 0      # Aln: no-breathing alarm flag
        self.q_f: int            = INACTIVITY_INITIAL_FRAMES  # adaptive threshold
        self._consecutive_zero: int = 0   # frames with rf=0
        self.is_kumbhaka: bool   = False
        self._kumbhaka_start_frame: int = 0
        self.kumbhaka_duration_s: float = 0.0

        # Frame counter
        self._frame_count: int = 0
        self._last_change_calc_frame: int = 0

    def update(self, rf: float) -> dict:
        """
        Updates the monitor with the current respiration rate.
        Called every frame.

        Implements Equations 23-28.

        Args:
            rf: current respiration rate from RREstimator (bpm)

        Returns:
            dict with alert_level, is_kumbhaka, kumbhaka_duration_s, etc.
        """
        self._frame_count += 1
        self._rr_history.append(rf)

        # ── Equation 23: Fixed threshold alarm ───────────────────────
        # Alf = NU if rf > εu  (RR too high)
        #     = NL if rf < εl  (RR too low)
        #     = NG otherwise   (normal)
        if rf > 0:   # Only check when we have a valid RR
            if rf > self._upper_fixed_limit:
                self.fixed_alarm = "upper"
            elif rf < self._lower_fixed_limit:
                self.fixed_alarm = "lower"
            else:
                self.fixed_alarm = "none"

        # ── Equations 27-28: Respiratory arrest / Kumbhaka ──────────
        # Aln = 1 if rf = 0 for ℓ continuous frames
        # qf grows by ℓ each detection (up to Z max)
        self._update_arrest_detection(rf)

        # ── Equations 24-26: Rate of change every 10 seconds ────────
        if (self._frame_count - self._last_change_calc_frame
                >= CHANGE_CALC_FRAMES):
            self._calculate_change()
            self._last_change_calc_frame = self._frame_count

        return self._get_state()

    def _update_arrest_detection(self, rf: float) -> None:
        """
        Implements Equations 27 and 28.

        Equation 27:
            Aln = 1 if Σ rf = 0 for ℓ frames continuously
                = 0 otherwise

        Equation 28:
            qf = qf−1 + ℓ  if Aln = 1 AND qf−1 < Z
               = ℓ          if qf−1 > Z
               = qf−1       otherwise

        Note: the paper's "Σ rf = 0" means rf has been zero
        for qf continuous frames, not cumulative sum.
        We track consecutive zero-RR frames.
        """
        if rf == 0.0:
            self._consecutive_zero += 1
        else:
            # RR is non-zero → person is breathing → reset everything
            self._consecutive_zero = 0
            self.aln = 0
            self.is_kumbhaka = False
            self.kumbhaka_duration_s = 0.0
            self._kumbhaka_start_frame = 0
            # Reset qf to initial value when breathing resumes
            self.q_f = INACTIVITY_INITIAL_FRAMES

        # Check if we have reached the inactivity threshold
        if self._consecutive_zero >= self.q_f:
            # Trigger arrest / Kumbhaka detection
            self.aln = 1
            self.is_kumbhaka = True

            # Record start frame for duration calculation
            if self._kumbhaka_start_frame == 0:
                # Mark start as when consecutive zeros began
                self._kumbhaka_start_frame = (
                    self._frame_count - self._consecutive_zero
                )

            # Calculate duration in seconds
            self.kumbhaka_duration_s = (
                self._frame_count - self._kumbhaka_start_frame
            ) / FRAME_RATE

            # Equation 28: update adaptive threshold qf
            if self.q_f < INACTIVITY_MAX_FRAMES:
                self.q_f = self.q_f + INACTIVITY_INITIAL_FRAMES
            else:
                # Wrap back to initial when maximum is reached
                self.q_f = INACTIVITY_INITIAL_FRAMES

    def _calculate_change(self) -> None:
        """
        Calculates rate of change of RR every 10 seconds.
        Implements Equations 24, 25, 26.
        """
        if len(self._rr_history) < CHANGE_CALC_FRAMES:
            return   # Not enough data

        # Get current and past RR values
        rf_current = self._rr_history[-1]   # rf
        rf_past    = self._rr_history[0]    # rf−k (10 seconds ago)

        # Skip if no valid current RR
        if rf_current == 0:
            return

        # Equation 24: Δrf = |rf − rf−k|
        delta_rf = abs(rf_current - rf_past)

        # Equation 25: δrf = (Δrf / rf) × 100
        delta_rf_percent = (delta_rf / rf_current) * 100.0

        # Equation 26: Classify alert level
        # Paper: 1 = 1%, 2 = 25%, 3 = 50%
        if delta_rf_percent > ALERT_MODERATE_THRESH:
            self.alert_level = "critical"
        elif delta_rf_percent > ALERT_MINOR_THRESH:
            self.alert_level = "moderate"
        elif delta_rf_percent > 1.0:
            self.alert_level = "minor"
        else:
            self.alert_level = "none"

    def _get_state(self) -> dict:
        """Returns current monitor state as dictionary."""
        return {
            "alert_level":        self.alert_level,
            "fixed_alarm":        self.fixed_alarm,
            "is_kumbhaka":        self.is_kumbhaka,
            "kumbhaka_duration":  round(self.kumbhaka_duration_s, 1),
            "aln":                self.aln,
            "q_f_seconds":        round(self.q_f / FRAME_RATE, 1),
        }

    def reset(self) -> None:
        """Resets all state — call when user restarts session."""
        self._rr_history.clear()
        self.alert_level           = "none"
        self.fixed_alarm           = "none"
        self.aln                   = 0
        self.q_f                   = INACTIVITY_INITIAL_FRAMES
        self._consecutive_zero     = 0
        self.is_kumbhaka           = False
        self._kumbhaka_start_frame = 0
        self.kumbhaka_duration_s   = 0.0
        self._frame_count          = 0
        self._last_change_calc_frame = 0
