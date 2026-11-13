"""Converts real-time durations into frame counts at the processing rate.

The behavior detectors count consecutive frames. Those counts were tuned at
10 FPS, so changing --fps used to silently change how many seconds each alert
needed. Thresholds are now declared in seconds and converted here, so the
alert timing stays the same at any processing rate.
"""

from dataclasses import dataclass

REFERENCE_FPS = 10.0  # rate the original frame-count thresholds were tuned at

SMOKING_CHECK_SECS = 1.5       # smoking cadence (changes slowly)
# Seatbelt is the costliest model (full frame) and the slowest-changing state:
# checking it every 3 s instead of 1.5 s halves its CPU share and only delays
# the NO SEATBELT alert by up to 1.5 s.
SEATBELT_CHECK_SECS = 3.0
# Sustained-behavior durations, equal to the old frame counts at 10 FPS.
# Detectors that fire on `counter > threshold` get frames - 1 (see Cadence).
HEAD_TURN_SECS = 0.6
EYE_RUB_SECS = 0.6
EYE_RUB_HISTORY_SECS = 0.5     # movement-average window
EYE_RUB_MIN_MOVE_PX = 2.0      # per frame at REFERENCE_FPS (rescaled by per_frame)
EYE_RUB_MAX_MOVE_PX = 15.0
PHONE_CALL_SECS = 0.9
HEAD_PITCH_SECS = 0.6
HEAD_FACE_LOST_SECS = 2.8
RECORD_CONFIRM_SECS = 0.3      # sustained risk before a --record risk clip starts


def timing_fps(processing_fps: float) -> float:
    """Rate used for conversions. --fps 0 means an unthrottled loop, whose
    real rate is unknown up front, so it falls back to REFERENCE_FPS.

    Example: timing_fps(8) -> 8.0; timing_fps(0) -> 10.0
    """
    return float(processing_fps) if processing_fps > 0 else REFERENCE_FPS


def frames_for(seconds: float, fps: float) -> int:
    """Number of frames (at least 1) that spans `seconds` at `fps`.

    Example: frames_for(0.6, 8) -> 5; frames_for(1.5, 8) -> 12
    """
    if seconds <= 0 or fps <= 0:
        raise ValueError(f"seconds and fps must be > 0, got seconds={seconds}, fps={fps}")
    return max(1, round(seconds * fps))


def per_frame(value_per_reference_frame: float, fps: float) -> float:
    """Rescales a per-frame quantity (e.g. pixels of hand movement between
    frames) tuned at REFERENCE_FPS: fewer frames per second means more
    movement between two consecutive frames.

    Example: per_frame(2.0, 8) -> 2.5
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    return value_per_reference_frame * REFERENCE_FPS / fps


@dataclass(frozen=True)
class Cadence:
    """The *_SECS constants converted to frame counts for one processing rate.

    Example: Cadence.at_fps(8).head_turn_gt == 4  (fires on the 5th frame = 0.6 s)
    """
    fps: float
    smoking_every: int
    seatbelt_every: int
    head_turn_gt: int        # detectors that fire on `counter > n`
    eye_rub_gt: int
    phone_call_gt: int
    eye_rub_history: int
    head_pitch_frames: int   # detectors that fire on `counter >= n`
    head_face_lost_frames: int
    record_confirm_frames: int

    @classmethod
    def at_fps(cls, processing_fps: float) -> "Cadence":
        fps = timing_fps(processing_fps)
        return cls(
            fps=fps,
            smoking_every=frames_for(SMOKING_CHECK_SECS, fps),
            seatbelt_every=frames_for(SEATBELT_CHECK_SECS, fps),
            head_turn_gt=frames_for(HEAD_TURN_SECS, fps) - 1,
            eye_rub_gt=frames_for(EYE_RUB_SECS, fps) - 1,
            phone_call_gt=frames_for(PHONE_CALL_SECS, fps) - 1,
            eye_rub_history=frames_for(EYE_RUB_HISTORY_SECS, fps),
            head_pitch_frames=frames_for(HEAD_PITCH_SECS, fps),
            head_face_lost_frames=frames_for(HEAD_FACE_LOST_SECS, fps),
            record_confirm_frames=frames_for(RECORD_CONFIRM_SECS, fps),
        )
