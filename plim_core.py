"""
plim_core.py
============
Implements Section 2.1 of Lee et al. (2021):
"Adaptive breathing motion detection" — the PLIM algorithm core.

This version uses a persistent background model and also handles
ROI shape changes safely. When the ROI changes size after LABB
refinement, the background model is reinitialised instead of
crashing on shape mismatch.
"""

import numpy as np
from config import BG_MODEL_DTYPE


class PLIMCore:
    def __init__(self):
        self.background_model: np.ndarray | None = None
        self.is_initialised: bool = False

        self.last_activity_map: np.ndarray | None = None
        self.last_ef: int = 0
        self.last_delta: np.ndarray | None = None
        self.frame_count: int = 0

    def initialise(self, first_roi_gray: np.ndarray) -> None:
        self.background_model = first_roi_gray.astype(BG_MODEL_DTYPE)
        self.is_initialised = True
        self.frame_count = 0
        self.last_activity_map = np.zeros_like(first_roi_gray, dtype=np.uint8)
        self.last_delta = np.zeros_like(self.background_model, dtype=BG_MODEL_DTYPE)
        self.last_ef = 0

    def process(self, current_roi_gray: np.ndarray, alpha: int, nu: int):
        if not self.is_initialised:
            raise RuntimeError(
                "PLIMCore.initialise() must be called before process()"
            )

        current_gray_f32 = current_roi_gray.astype(BG_MODEL_DTYPE)

        # LABB can shrink or expand the ROI after calibration.
        # If that happens, the stored background model no longer matches
        # the new ROI shape. Reinitialise cleanly instead of crashing.
        if (
            self.background_model is None
            or self.background_model.shape != current_gray_f32.shape
        ):
            self.initialise(current_roi_gray)
            return self.last_activity_map.copy(), 0

        delta = current_gray_f32 - self.background_model
        self.last_delta = delta

        activity_map = (np.abs(delta) > alpha).astype(np.uint8)
        ef = int(np.sum(activity_map))

        nu = max(1, int(nu))
        if self.frame_count % nu == 0:
            self.background_model = self.background_model + np.sign(delta)
            self.background_model = np.clip(self.background_model, 0, 255)

        self.last_activity_map = activity_map
        self.last_ef = ef
        self.frame_count += 1

        return activity_map, ef

    def reset(self) -> None:
        self.background_model = None
        self.is_initialised = False
        self.last_activity_map = None
        self.last_ef = 0
        self.last_delta = None
        self.frame_count = 0

    def get_background_as_uint8(self) -> np.ndarray | None:
        if self.background_model is None:
            return None
        return np.clip(self.background_model, 0, 255).astype(np.uint8)
