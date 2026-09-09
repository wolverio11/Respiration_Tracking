"""
template_manager.py
===================
Breathing template learning and filtering.

This version is deliberately defensive:
- Cleans activity maps before calibration/filtering
- Ignores useless near-empty frames during calibration
- Safely handles template/activity shape mismatches without crashing
"""

import numpy as np
import cv2
from config import (
    CALIBRATION_FRAMES,
    TEMPLATE_THRESHOLD,
    TEMPLATE_RESET_FRAMES,
    NOTABLE_EF_FRACTION,
    VERY_LOW_EF_PRIME,
)


class TemplateManager:
    def __init__(self):
        self.breathing_template: np.ndarray | None = None
        self.is_calibrated: bool = False
        self._calibration_maps: list = []
        self._frames_seen_in_calibration: int = 0
        self._mismatch_frame_count: int = 0
        self._notable_ef_threshold: float = 0.0
        self.last_filtered_map: np.ndarray | None = None
        self.last_ef_prime: int = 0
        self._roi_pixel_count: int = 0
        self.calibration_progress: float = 0.0
        self._kernel = np.ones((3, 3), dtype=np.uint8)

    def _clean_activity_map(self, activity_map: np.ndarray) -> np.ndarray:
        cleaned = activity_map.astype(np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, self._kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, self._kernel)
        return cleaned

    def calibrate(self, activity_map: np.ndarray) -> bool:
        if self.is_calibrated:
            return True

        cleaned_map = self._clean_activity_map(activity_map)
        ef = int(np.sum(cleaned_map))

        if self._roi_pixel_count == 0 or self._roi_pixel_count != cleaned_map.size:
            self._roi_pixel_count = cleaned_map.size

        min_informative_pixels = max(8, int(self._roi_pixel_count * 0.002))
        if ef >= min_informative_pixels:
            self._calibration_maps.append(cleaned_map.copy())

        self._frames_seen_in_calibration += 1
        self.calibration_progress = min(
            1.0, self._frames_seen_in_calibration / CALIBRATION_FRAMES
        )

        if self._frames_seen_in_calibration < CALIBRATION_FRAMES:
            return False

        if not self._calibration_maps:
            self.breathing_template = np.zeros_like(cleaned_map, dtype=np.uint8)
            self.is_calibrated = True
            self.calibration_progress = 1.0
            self._notable_ef_threshold = max(
                20, self._roi_pixel_count * NOTABLE_EF_FRACTION
            )
            return True

        stack = np.array(self._calibration_maps, dtype=np.float32)
        template_raw = np.mean(stack, axis=0)

        self.breathing_template = np.zeros_like(template_raw, dtype=np.uint8)
        self.breathing_template[template_raw >= TEMPLATE_THRESHOLD] = 1
        self.breathing_template = cv2.morphologyEx(
            self.breathing_template, cv2.MORPH_OPEN, self._kernel
        )
        self.breathing_template = cv2.morphologyEx(
            self.breathing_template, cv2.MORPH_CLOSE, self._kernel
        )

        avg_ef_calibration = float(np.mean([np.sum(m) for m in self._calibration_maps]))
        self._notable_ef_threshold = max(
            self._roi_pixel_count * NOTABLE_EF_FRACTION,
            avg_ef_calibration * 1.5,
        )

        self._calibration_maps.clear()
        self.is_calibrated = True
        self.calibration_progress = 1.0
        return True

    def filter(self, activity_map: np.ndarray, ef: int):
        if not self.is_calibrated or self.breathing_template is None:
            empty = np.zeros_like(activity_map, dtype=np.uint8)
            return empty, 0

        cleaned_map = self._clean_activity_map(activity_map)

        if cleaned_map.shape != self.breathing_template.shape:
            # Defensive recovery: this build should not crash because of a
            # template/ROI mismatch. Reset and force recalibration.
            self._reset_template()
            empty = np.zeros_like(activity_map, dtype=np.uint8)
            self.last_filtered_map = empty
            self.last_ef_prime = 0
            return empty, 0

        filtered_map = np.bitwise_and(cleaned_map, self.breathing_template)
        filtered_map = filtered_map.astype(np.uint8)
        ef_prime = int(np.sum(filtered_map))

        is_ef_notable = ef > self._notable_ef_threshold
        is_ef_prime_low = ef_prime < VERY_LOW_EF_PRIME

        if is_ef_notable and is_ef_prime_low:
            self._mismatch_frame_count += 1
        else:
            self._mismatch_frame_count = 0

        if self._mismatch_frame_count >= TEMPLATE_RESET_FRAMES:
            self._reset_template()

        self.last_filtered_map = filtered_map
        self.last_ef_prime = ef_prime
        return filtered_map, ef_prime

    def _reset_template(self) -> None:
        self.breathing_template = None
        self.is_calibrated = False
        self._calibration_maps.clear()
        self._frames_seen_in_calibration = 0
        self._mismatch_frame_count = 0
        self.calibration_progress = 0.0
        self._notable_ef_threshold = 0.0

    def full_reset(self) -> None:
        self._reset_template()
        self.last_filtered_map = None
        self.last_ef_prime = 0
        self._roi_pixel_count = 0

    def get_template_as_image(self) -> np.ndarray | None:
        if self.breathing_template is None:
            return None
        return (self.breathing_template * 255).astype(np.uint8)
