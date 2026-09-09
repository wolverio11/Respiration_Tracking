"""
lighting_adapter.py
===================
Implements Section 2.6 of Lee et al. (2021):
"Online adaptation to lighting"

WHAT THIS MODULE DOES:
    Every frame, the camera image has a certain average brightness.
    A dark meditation room needs MORE sensitivity (lower threshold α)
    so even tiny belly movements are detected.
    A bright room needs LESS sensitivity (higher α) to avoid
    false detections from light reflections.

    This module looks at average frame brightness (φf) and
    automatically sets α (motion threshold) and ν (background
    update frequency) for the PLIM core.

PAPER EQUATIONS IMPLEMENTED:
    Equation 29: φf = Σ Vf(x,y) / (w × h)
    Equation 30: α = ω1 if φf < β1, ω2 if β1 ≤ φf ≤ β2, ω3 if φf > β2
    Equation 31: ν = γ1 if α = ω1, else γ2

PRANAYAMA CONTEXT:
    Many people do pranayama in dim, calm environments.
    A candle-lit room might have φf around 40-60.
    A well-lit yoga studio might have φf around 120-150.
    This module handles both automatically.
"""

import cv2
import numpy as np
from config import (
    MOTION_THRESHOLD_DIM,
    MOTION_THRESHOLD_MED,
    MOTION_THRESHOLD_BRIGHT,
    BG_UPDATE_FREQ_DIM,
    BG_UPDATE_FREQ_OTHER,
    LIGHT_THRESHOLD_LOW,
    LIGHT_THRESHOLD_HIGH,
)


class LightingAdapter:
    """
    Measures current lighting and returns appropriate α and ν values.

    Why α and ν matter:
    ─────────────────
    α (alpha) = motion detection threshold
        A pixel must change by MORE than α to be counted as "active"
        in the activity map Af(bx,by).
        In dim light: camera noise is relative larger, so we need
        a LOWER α to still catch real breathing signals.
        In bright light: more noise from reflections, so HIGHER α
        prevents false positives.

    ν (nu) = background update frequency
        The background model Pf updates every ν frames.
        In dim light (α=8, more sensitive), we update slower (ν=4)
        so the background is more stable and breathing changes
        stand out more clearly.
        In normal/bright light, ν=2 keeps background fresh.
    """

    def __init__(self):
        # Store current values so other modules can read them
        self.current_alpha = MOTION_THRESHOLD_MED   # default: medium
        self.current_nu    = BG_UPDATE_FREQ_OTHER    # default: medium
        self.current_phi   = 0.0   # current illumination level φf

    def compute_illumination(self, frame: np.ndarray) -> float:
        """
        Calculates φf — average brightness of the frame.

        Implements Equation 29:
            φf = Σ Vf(x,y) / (w × h)

        Steps:
            1. Convert colour frame to grayscale Vf(x,y)
               (Grayscale = weighted average of R,G,B channels)
            2. Sum all pixel values
            3. Divide by total number of pixels

        Args:
            frame: BGR colour image from camera (numpy array)

        Returns:
            phi: float between 0 (pure black) and 255 (pure white)
        """
        # Step 1: Convert to grayscale
        # This produces Vf(x,y) from the paper
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape

        # Step 2+3: Equation 29 — sum and divide
        # numpy.mean does sum/count in one efficient operation
        phi = float(np.mean(gray))   # equivalent to Σ Vf(x,y) / (w×h)

        return phi

    def get_alpha_and_nu(self, phi: float):
        """
        Applies Equations 30 and 31 to get α and ν from φ.

        Equation 30 (choosing α):
            α = ω1 = 8  if φ < β1 (= 83)    → dim room
            α = ω2 = 12 if β1 ≤ φ ≤ β2       → medium room
            α = ω3 = 24 if φ > β2 (= 100)    → bright room

        Equation 31 (choosing ν):
            ν = γ1 = 4  if α = ω1  → slow update in dim room
            ν = γ2 = 2  otherwise  → fast update in normal/bright

        Args:
            phi: illumination level from compute_illumination()

        Returns:
            alpha: motion detection threshold
            nu: background update frequency
        """
        # Equation 30
        if phi < LIGHT_THRESHOLD_LOW:
            alpha = MOTION_THRESHOLD_DIM      # ω1 = 8, dim room
        elif phi <= LIGHT_THRESHOLD_HIGH:
            alpha = MOTION_THRESHOLD_MED      # ω2 = 12, medium
        else:
            alpha = MOTION_THRESHOLD_BRIGHT   # ω3 = 24, bright

        # Equation 31
        if alpha == MOTION_THRESHOLD_DIM:
            nu = BG_UPDATE_FREQ_DIM     # γ1 = 4
        else:
            nu = BG_UPDATE_FREQ_OTHER   # γ2 = 2

        return alpha, nu

    def update(self, frame: np.ndarray):
        """
        Main method: call every frame to get current α and ν.

        Updates internal state and returns current values.

        Args:
            frame: current BGR camera frame

        Returns:
            alpha: int, current motion threshold
            nu:    int, current background update frequency
        """
        phi = self.compute_illumination(frame)
        alpha, nu = self.get_alpha_and_nu(phi)

        # Store for external inspection
        self.current_alpha = alpha
        self.current_nu    = nu
        self.current_phi   = phi

        return alpha, nu

    def get_lighting_label(self) -> str:
        """Returns human-readable lighting description for UI display."""
        if self.current_phi < LIGHT_THRESHOLD_LOW:
            return "dim"
        elif self.current_phi <= LIGHT_THRESHOLD_HIGH:
            return "medium"
        else:
            return "bright"
