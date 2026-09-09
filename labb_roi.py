"""
labb_roi.py
===========
Implements Section 2.2 of Lee et al. (2021):
"Adaptive region of interest detection" — the LABB model.

LABB = Localisation of Active Breathing Behaviour.

WHAT THIS MODULE DOES:
    Once the breathing template Bf is known, we know exactly
    WHICH pixels in the ROI correspond to breathing movement.
    The LABB model uses this template to find the minimal
    bounding box around the breathing region.

    This bounding box becomes the refined ROI for all subsequent
    processing. It is tighter than the initial default ROI,
    which means less background noise and faster processing.

    If the refined ROI is too small (weak breathing signal),
    the system expands it by Ω = 15% on each side.

    If body movement is detected, the ROI is reset and will
    be recomputed from the new breathing template.

PAPER EQUATIONS IMPLEMENTED:
    Equation 8:   H = [[0,1,0],[1,1,1],[0,1,0]] — erosion kernel
    Equation 9:   R'f = Bf ⊕ H — erode template to remove noise
    Equation 10:  x1 expansion if ROI width < w/λ
    Equation 11:  x2 expansion if ROI width < w/λ
    Equation 12:  y1 expansion if ROI height < h/λ
    Equation 13:  y2 expansion if ROI height < h/λ

PRANAYAMA CONTEXT:
    In pranayama, the breathing region is usually the belly
    or chest. After calibration, the LABB model will find
    the exact bounding box of whichever area moved during
    the calibration breathing.
"""

import numpy as np
import cv2
from config import ROI_RESIZE_FACTOR, FRAME_DIVIDE_FACTOR


class LABBRoi:
    """
    Localisation of Active Breathing Behaviour.

    Produces a tight bounding box ROI around the detected
    breathing region, based on the breathing template.

    Coordinate system:
        (x1, y1) = top-left corner of ROI
        (x2, y2) = bottom-right corner of ROI
        width    = x2 - x1
        height   = y2 - y1
    """

    # Erosion structuring element H from Equation 8
    # Cross shape removes isolated single pixels (noise)
    # while preserving connected regions
    H = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ], dtype=np.uint8)

    def __init__(self):
        # Current detected ROI as (x1, y1, w, h) within the original frame
        # NOTE: these coordinates are WITHIN the full frame, not the
        # pre-cropped ROI passed to PLIM. They are set by
        # set_full_frame_offset() so the LABB output maps back correctly.
        self.current_roi: tuple | None = None

        # Offset of the initial crop within the full frame
        # (so we can map template coordinates back to full frame)
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._initial_roi_w: int = 0
        self._initial_roi_h: int = 0

    def set_initial_crop(self, x: int, y: int, w: int, h: int) -> None:
        """
        Records the offset of the initial ROI crop within the full frame.
        This is needed to map LABB coordinates back to full frame.

        Call this when the initial crop is first set up.

        Args:
            x, y: top-left of initial crop in full frame
            w, h: width and height of initial crop
        """
        self._offset_x = x
        self._offset_y = y
        self._initial_roi_w = w
        self._initial_roi_h = h

    def detect_from_template(self,
                              breathing_template: np.ndarray,
                              frame_width: int,
                              frame_height: int) -> tuple:
        """
        Finds the ROI bounding box from the breathing template.

        Implements Equations 8-13.

        Step by step:
            1. Erode template with H (Eq. 8-9) to remove noise pixels
            2. Find bounding box of remaining active pixels
            3. If bounding box is too small, expand it (Eq. 10-13)
            4. Clamp to frame boundaries
            5. Map back to full-frame coordinates using offset

        Args:
            breathing_template: binary 2D array Bf from TemplateManager
            frame_width:  full frame width (w in paper)
            frame_height: full frame height (h in paper)

        Returns:
            (x, y, w, h) — ROI as (left, top, width, height)
                            in FULL FRAME coordinates
        """
        if breathing_template is None or breathing_template.size == 0:
            return self.get_default_roi(frame_width, frame_height)

        # ── Equation 9: R'f = Bf ⊕ H (erosion) ─────────────────────
        # Erosion removes small isolated active pixels that are noise.
        # Only connected regions of active pixels survive.
        template_uint8 = breathing_template.astype(np.uint8)
        R_prime = cv2.erode(template_uint8, self.H, iterations=1)

        # Find all pixel locations where R_prime == 1
        active_pixels = np.argwhere(R_prime == 1)
        # argwhere returns array of (row, col) = (y, x) positions

        if len(active_pixels) == 0:
            # No active pixels after erosion — use default ROI
            return self.get_default_roi(frame_width, frame_height)

        # ── Find bounding box of active region ───────────────────────
        # These are coordinates WITHIN the template (ROI space)
        y1_prime = int(active_pixels[:, 0].min())
        y2_prime = int(active_pixels[:, 0].max())
        x1_prime = int(active_pixels[:, 1].min())
        x2_prime = int(active_pixels[:, 1].max())

        roi_width_in_template  = x2_prime - x1_prime
        roi_height_in_template = y2_prime - y1_prime

        template_h, template_w = R_prime.shape

        # ── Equations 10-11: Expand x if too narrow ─────────────────
        # "if x2 - x1 < w/λ" — using template dimensions here
        if roi_width_in_template < template_w / FRAME_DIVIDE_FACTOR:
            x1 = int((1 - ROI_RESIZE_FACTOR) * x1_prime)
            x2 = int((1 + ROI_RESIZE_FACTOR) * x2_prime)
        else:
            x1 = x1_prime
            x2 = x2_prime

        # ── Equations 12-13: Expand y if too short ──────────────────
        if roi_height_in_template < template_h / FRAME_DIVIDE_FACTOR:
            y1 = int((1 - ROI_RESIZE_FACTOR) * y1_prime)
            y2 = int((1 + ROI_RESIZE_FACTOR) * y2_prime)
        else:
            y1 = y1_prime
            y2 = y2_prime

        # ── Clamp to template boundaries ────────────────────────────
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(template_w, x2)
        y2 = min(template_h, y2)

        # ── Map back to full frame coordinates ───────────────────────
        # Template coordinates are relative to the initial crop.
        # Add offset to get full frame coordinates.
        x1_full = self._offset_x + x1
        y1_full = self._offset_y + y1
        x2_full = self._offset_x + x2
        y2_full = self._offset_y + y2

        # Clamp to full frame boundaries
        x1_full = max(0, x1_full)
        y1_full = max(0, y1_full)
        x2_full = min(frame_width,  x2_full)
        y2_full = min(frame_height, y2_full)

        final_w = max(10, x2_full - x1_full)
        final_h = max(10, y2_full - y1_full)

        roi = (x1_full, y1_full, final_w, final_h)
        self.current_roi = roi
        return roi

    def get_default_roi(self, frame_width: int, frame_height: int) -> tuple:
        """
        Default ROI used before calibration is complete.

        [PRANAYAMA ASSUMPTION]
        For pranayama in seated position (Padmasana / Sukhasana):
        - Person sits facing camera at navel/belly height
        - Belly region is approximately:
            - Horizontally: centre 50% of frame
            - Vertically: 35% to 65% of frame height
              (belly area when camera is at navel height)

        This is a practical assumption. The LABB model will refine
        this to the exact breathing region after calibration.

        Args:
            frame_width:  full frame width in pixels
            frame_height: full frame height in pixels

        Returns:
            (x, y, w, h) default ROI covering belly area
        """
        x = frame_width  // 4           # start at 25% from left
        y = int(frame_height * 0.35)    # start at 35% from top
        w = frame_width  // 2           # 50% of frame width
        h = int(frame_height * 0.30)    # 30% of frame height

        roi = (x, y, w, h)
        self.current_roi = roi
        return roi

    def reset_roi(self) -> None:
        """
        Resets the ROI — called when body movement is detected.
        Next call to detect_from_template will recompute from template.
        """
        self.current_roi = None
