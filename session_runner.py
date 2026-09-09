"""
session_runner.py
=================
The coordinator — connects all modules together and runs
the main camera processing loop.

This module does NOT implement any paper algorithms itself.
It orchestrates the correct sequence of calls to each module
every frame, collects the results, annotates the video frame,
and stores the system state for the Flask server to serve.

PROCESSING ORDER EVERY FRAME (strictly follows paper sections):
    1. Capture frame from camera
    2. LightingAdapter.update()         → α, ν      (Section 2.6)
    3. Get ROI (default or LABB)        → roi        (Section 2.2)
    4. PLIMCore.process()               → Af, ef     (Section 2.1)
    5a. TemplateManager.calibrate()     → template   (calibration phase)
    5b. TemplateManager.filter()        → Mf, e'f    (running phase)
    6. Body movement classification                  (Section 2.3)
    7. RREstimator.update()             → rf etc.    (Section 2.4)
    8. ChangeMonitor.update()           → alerts     (Section 2.5)
    9. Annotate frame for display
    10. Store state

THREAD SAFETY:
    The camera loop runs in a background thread.
    The Flask server reads state from the main thread.
    threading.Lock() protects all shared state.
"""

import cv2
import numpy as np
import threading
import time
from collections import deque

from config import (
    CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT, FRAME_RATE,
    NOTABLE_EF_FRACTION, VERY_LOW_EF_PRIME,
)
from lighting_adapter   import LightingAdapter
from plim_core          import PLIMCore
from template_manager   import TemplateManager
from labb_roi           import LABBRoi
from rr_estimator       import RREstimator
from change_monitor     import ChangeMonitor


class SessionRunner:
    """
    Runs the complete PLIM pipeline in a background thread.
    Provides thread-safe access to the latest frame and state.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self._camera_index = camera_index

        # ── Instantiate all modules ──────────────────────────────────
        self.lighting_adapter   = LightingAdapter()
        self.plim_core          = PLIMCore()
        self.template_manager   = TemplateManager()
        self.labb_roi           = LABBRoi()
        self.rr_estimator       = RREstimator()
        self.change_monitor     = ChangeMonitor()

        # ── Thread control ───────────────────────────────────────────
        self._is_running: bool = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # ── Shared state (protected by lock) ────────────────────────
        self._latest_jpeg: bytes = b""
        self._state: dict = self._empty_state()

        # ── Internal tracking ────────────────────────────────────────
        self._session_start: float = 0.0
        self._frame_count: int = 0
        self._fps_deque: deque = deque(maxlen=30)   # for FPS display
        self._current_roi: tuple = (
            FRAME_WIDTH // 4,
            int(FRAME_HEIGHT * 0.35),
            FRAME_WIDTH // 2,
            int(FRAME_HEIGHT * 0.30),
        )

        # e'f signal history for the activity chart (last 10 seconds)
        self._ef_prime_history: deque = deque(maxlen=FRAME_RATE * 10)
        self._ef_history:       deque = deque(maxlen=FRAME_RATE * 10)

    # ──────────────────────────────────────────────────────────────
    #  PUBLIC INTERFACE
    # ──────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Opens camera and starts the processing loop."""
        if self._is_running:
            return
        self._is_running = True
        self._session_start = time.time()
        self._thread = threading.Thread(
            target=self._loop, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stops the processing loop and releases camera."""
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def reset(self) -> None:
        """
        Resets all modules — recalibrates from scratch.
        Called when user clicks "Restart Session".
        """
        self.plim_core.reset()
        self.template_manager.full_reset()
        self.labb_roi.reset_roi()
        self.rr_estimator.reset()
        self.change_monitor.reset()
        self._ef_prime_history.clear()
        self._ef_history.clear()
        self._session_start = time.time()
        self._frame_count = 0
        with self._lock:
            self._state = self._empty_state()

    def get_frame(self) -> bytes:
        """Returns latest annotated JPEG frame bytes."""
        with self._lock:
            return self._latest_jpeg

    def get_state(self) -> dict:
        """Returns a copy of the current system state."""
        with self._lock:
            return dict(self._state)

    # ──────────────────────────────────────────────────────────────
    #  MAIN PROCESSING LOOP
    # ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """
        Background thread: captures frames and runs PLIM pipeline.
        Runs at camera frame rate (30 fps).
        """
        cap = cv2.VideoCapture(self._camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS,          FRAME_RATE)

        if not cap.isOpened():
            with self._lock:
                self._state["error"] = (
                    "Camera not found. Check camera index in config.py"
                )
            return

        while self._is_running:
            t_frame_start = time.time()

            # ── STEP 1: Capture frame ────────────────────────────────
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            self._frame_count += 1
            h_frame, w_frame = frame.shape[:2]

            # ── STEP 2: Lighting adaptation (Section 2.6) ────────────
            alpha, nu = self.lighting_adapter.update(frame)

            # ── STEP 3: Get ROI ──────────────────────────────────────
            # Keep ROI stable. Dynamic LABB refinement was causing shape
            # changes mid-session, which corrupted background/template
            # state and repeatedly crashed the pipeline. A fixed ROI is
            # less fancy but much more reliable for now.
            roi = self.labb_roi.get_default_roi(w_frame, h_frame)

            self._current_roi = roi
            rx, ry, rw, rh = roi

            # Extract ROI and convert to grayscale
            roi_bgr  = frame[ry:ry+rh, rx:rx+rw]
            if roi_bgr.size == 0:
                continue
            roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)

            # ── STEP 4: PLIM Core (Section 2.1, Eq. 1-5) ────────────
            if not self.plim_core.is_initialised:
                self.plim_core.initialise(roi_gray)
                # Set offset for LABB coordinate mapping
                self.labb_roi.set_initial_crop(rx, ry, rw, rh)
                continue

            roi_shape_changed = (
                self.plim_core.background_model is not None
                and self.plim_core.background_model.shape != roi_gray.shape
            )
            if roi_shape_changed:
                # LABB refined the ROI. Start a fresh local background for the
                # new crop and clear rate state so old peak history does not
                # pollute the new signal.
                self.plim_core.initialise(roi_gray)
                self.rr_estimator.reset()
                self.change_monitor.reset()
                activity_map = np.zeros_like(roi_gray, dtype=np.uint8)
                ef = 0
            else:
                activity_map, ef = self.plim_core.process(
                    roi_gray, alpha, nu
                )

            # ── STEP 5: Calibration or Filtering ────────────────────
            if not self.template_manager.is_calibrated:
                # Calibration phase: collect activity maps for template
                done = self.template_manager.calibrate(activity_map)
                ef_prime = 0
                rr_state = {"rf": 0.0, "upsilon": 0.0,
                            "zeta_u": 0.0, "zeta_l": 0.0,
                            "rho": 0, "beta": 0, "tau": 0.0,
                            "af": 0, "bf": 0}
                monitor_state = {"alert_level": "none",
                                 "is_kumbhaka": False,
                                 "kumbhaka_duration": 0.0,
                                 "fixed_alarm": "none",
                                 "aln": 0,
                                 "q_f_seconds": 10.0}
            else:
                # Running phase: filter activity map with template

                # ── STEP 5b: Template filtering (Eq. 6-7) ───────────
                filtered_map, ef_prime = self.template_manager.filter(
                    activity_map, ef
                )

                # ── STEP 6: Movement classification (Section 2.3) ────
                # ef high but e'f low for sustained period
                # → body movement → reset ROI
                # (auto-reset handled inside TemplateManager,
                #  here we reset ROI when template resets)
                if not self.template_manager.is_calibrated:
                    self.labb_roi.reset_roi()

                # ── STEP 7: RR Estimation (Section 2.4, Eq. 14-22) ──
                rr_state = self.rr_estimator.update(ef_prime, timestamp=time.time())

                # ── STEP 8: Change Monitoring (Section 2.5, Eq.23-28)
                monitor_state = self.change_monitor.update(
                    rr_state["rf"]
                )

            # Store signal history for chart
            self._ef_prime_history.append(ef_prime)
            self._ef_history.append(ef)

            # ── STEP 9: Annotate frame ───────────────────────────────
            annotated = self._draw_overlay(
                frame.copy(), roi, ef, ef_prime,
                rr_state, monitor_state,
                self.template_manager.is_calibrated,
                self.template_manager.calibration_progress,
                self.lighting_adapter.get_lighting_label(),
                alpha
            )

            # ── STEP 10: Encode JPEG ─────────────────────────────────
            _, jpeg_buf = cv2.imencode(
                ".jpg", annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 75]
            )

            # ── STEP 11: Update shared state (thread-safe) ───────────
            self._fps_deque.append(1.0 / max(
                time.time() - t_frame_start, 1e-6
            ))
            actual_fps = round(
                sum(self._fps_deque) / len(self._fps_deque), 1
            )

            session_secs = time.time() - self._session_start

            new_state = {
                # Core RR output
                "rf":            rr_state.get("rf", 0.0),
                # PLIM internals for display
                "ef":            ef,
                "ef_prime":      ef_prime,
                "upsilon":       rr_state.get("upsilon", 0.0),
                "zeta_u":        rr_state.get("zeta_u", 0.0),
                "zeta_l":        rr_state.get("zeta_l", 0.0),
                "rho":           rr_state.get("rho", 0),
                "beta":          rr_state.get("beta", 0),
                "tau":           rr_state.get("tau", 0.0),
                # Alert and Kumbhaka
                "alert_level":   monitor_state.get("alert_level", "none"),
                "is_kumbhaka":   monitor_state.get("is_kumbhaka", False),
                "kumbhaka_s":    monitor_state.get("kumbhaka_duration", 0.0),
                "fixed_alarm":   monitor_state.get("fixed_alarm", "none"),
                # Session info
                "is_calibrated": self.template_manager.is_calibrated,
                "calib_pct":     round(
                    self.template_manager.calibration_progress * 100, 0
                ),
                "session_s":     round(session_secs, 0),
                "fps":           actual_fps,
                "lighting":      self.lighting_adapter.get_lighting_label(),
                "alpha":         alpha,
                "nu":            nu,
                # Signal chart data (ef and ef_prime histories)
                "ef_history":    list(self._ef_history)[-150:],
                "ef_prime_history": list(self._ef_prime_history)[-150:],
                "error":         "",
            }

            with self._lock:
                self._latest_jpeg = jpeg_buf.tobytes()
                self._state = new_state

        cap.release()

    # ──────────────────────────────────────────────────────────────
    #  FRAME ANNOTATION
    # ──────────────────────────────────────────────────────────────

    def _draw_overlay(
        self, frame, roi, ef, ef_prime,
        rr_state, monitor_state,
        is_calibrated, calib_pct,
        lighting_label, alpha
    ) -> np.ndarray:
        """
        Draws informative overlays on the video frame.
        Shows ROI rectangle, RR, ef, e'f, thresholds, alerts.
        """
        rx, ry, rw, rh = roi
        rf      = rr_state.get("rf", 0.0)
        rho     = rr_state.get("rho", 0)
        upsilon = rr_state.get("upsilon", 0.0)
        zeta_u  = rr_state.get("zeta_u",  0.0)
        zeta_l  = rr_state.get("zeta_l",  0.0)
        is_kumb = monitor_state.get("is_kumbhaka", False)
        alert   = monitor_state.get("alert_level", "none")

        # Colour for ROI rectangle based on detection state
        if is_kumb:
            roi_colour = (0, 215, 255)    # gold = Kumbhaka
        elif rho == 1:
            roi_colour = (0, 200, 80)     # green = in breath peak
        elif is_calibrated:
            roi_colour = (200, 100, 0)    # blue = monitoring
        else:
            roi_colour = (120, 120, 120)  # grey = calibrating

        # Draw ROI rectangle
        cv2.rectangle(frame, (rx, ry), (rx+rw, ry+rh), roi_colour, 2)

        # Semi-transparent dark panel at top left for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (340, 180), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # ── Text overlays ────────────────────────────────────────────
        font  = cv2.FONT_HERSHEY_SIMPLEX
        white = (240, 240, 240)
        grey  = (160, 160, 160)
        gold  = (0, 200, 220)

        # RR value
        rr_text = f"{rf:.1f} bpm" if rf > 0 else "-- bpm"
        cv2.putText(frame, rr_text, (10, 38),
                    font, 1.1, gold, 2)

        # Calibration or live monitoring status
        if not is_calibrated:
            bar_w = int(290 * calib_pct)
            cv2.rectangle(frame, (10, 48), (300, 60), (40,40,40), -1)
            cv2.rectangle(frame, (10, 48), (10+bar_w, 60), (0,180,100), -1)
            cv2.putText(frame,
                        f"Calibrating... {int(calib_pct*100)}%",
                        (10, 78), font, 0.55, white, 1)
        else:
            status = "KUMBHAKA" if is_kumb else (
                "breath peak" if rho == 1 else "monitoring"
            )
            cv2.putText(frame, status, (10, 68), font, 0.6, roi_colour, 2)

        # PLIM internals
        cv2.putText(frame,
                    f"ef={ef}  e'f={ef_prime}  Y={upsilon:.0f}",
                    (10, 100), font, 0.5, grey, 1)
        cv2.putText(frame,
                    f"zu={zeta_u:.0f}  zl={zeta_l:.0f}  "
                    f"b={rr_state.get('beta',0)}  p={rho}",
                    (10, 120), font, 0.5, grey, 1)
        cv2.putText(frame,
                    f"light={lighting_label}  a={alpha}",
                    (10, 140), font, 0.5, grey, 1)

        # Alert badge
        alert_colours = {
            "minor":    (0, 200, 255),
            "moderate": (0, 140, 255),
            "critical": (0, 0, 255),
        }
        if alert in alert_colours:
            cv2.putText(frame, f"ALERT: {alert.upper()}",
                        (10, 165), font, 0.6,
                        alert_colours[alert], 2)

        # Kumbhaka duration overlay
        if is_kumb:
            kumb_s = monitor_state.get("kumbhaka_duration", 0.0)
            cv2.putText(frame,
                        f"KUMBHAKA  {kumb_s:.1f}s",
                        (frame.shape[1]//2 - 120, 50),
                        font, 1.0, (0, 215, 255), 2)

        return frame

    # ──────────────────────────────────────────────────────────────
    #  HELPERS
    # ──────────────────────────────────────────────────────────────

    @staticmethod
    def _empty_state() -> dict:
        return {
            "rf": 0.0, "ef": 0, "ef_prime": 0,
            "upsilon": 0.0, "zeta_u": 0.0, "zeta_l": 0.0,
            "rho": 0, "beta": 0, "tau": 0.0,
            "alert_level": "none", "is_kumbhaka": False,
            "kumbhaka_s": 0.0, "fixed_alarm": "none",
            "is_calibrated": False, "calib_pct": 0.0,
            "session_s": 0.0, "fps": 0.0,
            "lighting": "unknown", "alpha": 12, "nu": 2,
            "ef_history": [], "ef_prime_history": [],
            "error": "",
        }
