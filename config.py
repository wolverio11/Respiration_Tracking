"""
config.py
=========
All constants used in this project.

SOURCE: Every value here comes from one of two places:
  1. Lee et al. (2021) "A real-time camera-based adaptive breathing
     monitoring system" - Med Biol Eng Comput 59:1285-1298
     → Marked with [PAPER] and the equation/section number

  2. Pranayama physiological assumptions
     → Marked with [PRANAYAMA] and the source reasoning

No values are invented. Every number has a reason.
"""

# ─────────────────────────────────────────────────────────────
#  CAMERA SETTINGS
# ─────────────────────────────────────────────────────────────

# [PAPER] Section 3.1 - paper used 1080p at 30fps on smartphones
# We use 720p for better real-time performance on laptops
CAMERA_INDEX    = 0       # 0 = built-in webcam, 1 = first external
FRAME_WIDTH     = 1280    # pixels
FRAME_HEIGHT    = 720     # pixels
FRAME_RATE      = 30      # F in the paper, frames per second

# ─────────────────────────────────────────────────────────────
#  LIGHTING ADAPTATION CONSTANTS
#  [PAPER] Section 2.6, Equations 30 and 31
# ─────────────────────────────────────────────────────────────

# Motion detection threshold α
# Controls how sensitive the system is to pixel changes
# Lower α = more sensitive = detects smaller movements
MOTION_THRESHOLD_DIM    = 8    # ω1 — dim room (φ < 83)
MOTION_THRESHOLD_MED    = 12   # ω2 — medium light (83 ≤ φ ≤ 100)
MOTION_THRESHOLD_BRIGHT = 18   # ω3 — bright room (φ > 100)

# Background update frequency ν
# How many frames between background model updates
# Lower ν = background updates faster = less "memory"
BG_UPDATE_FREQ_DIM   = 4   # γ1 — used when in dim lighting
BG_UPDATE_FREQ_OTHER = 2   # γ2 — used in medium and bright lighting

# Illumination thresholds for classifying lighting
# These are the β1 and β2 values from Equation 30
LIGHT_THRESHOLD_LOW  = 83   # β1 — below this = dim room
LIGHT_THRESHOLD_HIGH = 100  # β2 — above this = bright room

# ─────────────────────────────────────────────────────────────
#  PLIM CORE CONSTANTS
#  [PAPER] Section 2.1, Equations 1-5
# ─────────────────────────────────────────────────────────────

# The background model is stored as float32 so the ±1 updates
# (Equation 3) accumulate correctly without integer rounding.
# This is an implementation detail, not stated in paper but required
# for correctness.
BG_MODEL_DTYPE = "float32"

# ─────────────────────────────────────────────────────────────
#  TEMPLATE MANAGER CONSTANTS
#  [PAPER] Section 2.1 (template learning) and Section 2.3
# ─────────────────────────────────────────────────────────────

# How many seconds of breathing to observe before saving template
# [PRANAYAMA] A typical pranayama breath cycle (inhale+exhale) is
# 4-8 seconds. 10 seconds captures at least 1-2 complete cycles,
# enough to build a reliable template.
# [PAPER] Paper does not specify exact duration - uses "initial" frames
CALIBRATION_SECONDS = 10   # seconds of breathing for template learning
CALIBRATION_FRAMES  = CALIBRATION_SECONDS * FRAME_RATE  # = 300 frames

# What fraction of calibration frames must show a pixel active
# for it to become part of the breathing template
# [PAPER] Paper uses binary matching: Bf is from the activity map
# 0.3 means pixel must be active in ≥30% of calibration frames
# This filters out noise pixels that occasionally activate
TEMPLATE_THRESHOLD  = 0.3

# Auto-reset: if ef is "high" but e'f stays "low" for 10s,
# it means person shifted and template no longer matches
# [PAPER] Section 2.1: "automatically updated if the proposed system
# detects a notable raw activity level (ef) while detects none or
# very less filtered breathing activity level (e'f) for ten
# continuous seconds"
TEMPLATE_RESET_SECONDS = 10
TEMPLATE_RESET_FRAMES  = TEMPLATE_RESET_SECONDS * FRAME_RATE  # = 300

# "Notable" ef = more than this many active pixels
# "Very less" e'f = fewer than this many template-matching pixels
# These are practical thresholds derived from typical ROI sizes
NOTABLE_EF_FRACTION    = 0.10  # ef > 10% of ROI pixels = "notable"
VERY_LOW_EF_PRIME      = 5     # e'f < 5 pixels = "very less"

# ─────────────────────────────────────────────────────────────
#  LABB ROI CONSTANTS
#  [PAPER] Section 2.2, Equations 8-13
# ─────────────────────────────────────────────────────────────

# Erosion structuring element H (Equation 8)
# Cross-shaped, removes isolated noise pixels from template
# H = [[0,1,0],[1,1,1],[0,1,0]]
# Defined as numpy array in labb_roi.py

# ROI expansion parameters when detected region is too small
ROI_RESIZE_FACTOR   = 0.15   # Ω — expand by 15% on each side
FRAME_DIVIDE_FACTOR = 5      # λ — ROI must be > frame_size/5

# ─────────────────────────────────────────────────────────────
#  RR ESTIMATOR CONSTANTS
#  [PAPER] Section 2.4, Equations 14-22
# ─────────────────────────────────────────────────────────────

# Threshold scaling factor for peak detection
# [PAPER] "σ1 = 0.75 is empirically determined"
SIGMA_1 = 0.75   # Upper threshold = ϒf × 0.75, Lower = ϒf × 0.25

# Number of breaths per RR calculation
# [PAPER] "L = 2, which is empirically determined parameter"
# Using 2 breaths gives fresh RR reading quickly
BREATHS_PER_CALC = 2   # L

# Conversion constants
SECONDS_PER_MINUTE = 60   # χ in Equation 22

# ─────────────────────────────────────────────────────────────
#  PRANAYAMA BREATHING RANGE
#  [PRANAYAMA] These bounds define what is physiologically
#  possible and meaningful in pranayama practice.
#
#  Sources:
#  - Jerath R. et al. (2006) Physiology of long pranayamic breathing.
#    Medical Hypotheses 67(3):566-571
#    → Slow pranayama can reach 2-3 breaths/min
#  - Telles S. et al. (2011) Changes in autonomic variables following
#    two meditative states described in yoga texts.
#    J Altern Complement Med 17(7):645-650
#    → Normal pranayama range: 4-20 breaths/min
#  - Kapalabhati: 60+ rapid abdominal pumps/min
# ─────────────────────────────────────────────────────────────
MIN_PRANAYAMA_BPM = 2    # Slowest (deep Bhramari, long Kumbhaka)
MAX_PRANAYAMA_BPM = 60   # Fastest (Kapalabhati)

# ─────────────────────────────────────────────────────────────
#  CHANGE MONITOR CONSTANTS
#  [PAPER] Section 2.5, Equations 23-28
# ─────────────────────────────────────────────────────────────

# How often to calculate rate of change (in frames)
# [PAPER] "change in the RR is calculated every 10 s"
CHANGE_CALC_SECONDS = 10
CHANGE_CALC_FRAMES  = CHANGE_CALC_SECONDS * FRAME_RATE  # k = 10F = 300

# Alert thresholds (percentage change in RR)
# [PAPER] "1 = 1%, 2 = 25%, 3 = 50% are empirically determined"
ALERT_MINOR_THRESH    = 25   # δrf ≤ 25% → minor alert
ALERT_MODERATE_THRESH = 50   # 25% < δrf ≤ 50% → moderate alert
# δrf > 50% → critical alert

# Respiratory arrest / Kumbhaka detection
# [PAPER] Section 2.5: "ℓ = 10F, Z = 30F"
# [PRANAYAMA] Kumbhaka (breath retention) is intentional arrest.
# The system detects it exactly as the paper detects arrest —
# by noticing rf = 0 for ℓ seconds. For pranayama we label
# this "Kumbhaka" instead of "respiratory arrest".
INACTIVITY_INITIAL_SECONDS = 10   # ℓ — first arrest alarm after 10s
INACTIVITY_MAX_SECONDS     = 30   # Z — maximum adaptive threshold
INACTIVITY_INITIAL_FRAMES  = INACTIVITY_INITIAL_SECONDS * FRAME_RATE
INACTIVITY_MAX_FRAMES      = INACTIVITY_MAX_SECONDS * FRAME_RATE

# ─────────────────────────────────────────────────────────────
#  PRANAYAMA SESSION SETTINGS
#  [PRANAYAMA] Standard ratios from classical pranayama texts.
#
#  Source: Swami Niranjanananda Saraswati (2009).
#  Prana and Pranayama. Yoga Publications Trust, Bihar, India.
#  Also: Iyengar B.K.S. (1985). Light on Yoga. Schocken Books.
# ─────────────────────────────────────────────────────────────

# Available pranayama modes shown in UI
# Format: name → (inhale_parts, hold_parts, exhale_parts)
PRANAYAMA_MODES = {
    "Free Monitoring":  None,           # no target ratio
    "Sama Vritti":      (1, 0, 1),      # equal inhale/exhale
    "Anulom Vilom":     (1, 1, 2),      # 1:1:2 ratio
    "Box Breathing":    (1, 1, 1),      # 1:1:1:1 (simplified)
    "4-7-8":            (4, 7, 8),      # 4 inhale, 7 hold, 8 exhale
    "Kapalabhati":      None,           # fast pumping, no fixed ratio
}
