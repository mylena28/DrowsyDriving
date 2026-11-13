import pytest

from utils.cadence import REFERENCE_FPS, Cadence, frames_for, per_frame, timing_fps


def test_frames_for_keeps_reference_counts_at_reference_fps():
    assert frames_for(0.6, REFERENCE_FPS) == 6
    assert frames_for(2.8, REFERENCE_FPS) == 28


def test_frames_for_scales_down_at_8_fps():
    assert frames_for(0.6, 8) == 5
    assert frames_for(1.5, 8) == 12


def test_frames_for_never_returns_zero():
    assert frames_for(0.01, 8) == 1


def test_frames_for_rejects_non_positive_input():
    with pytest.raises(ValueError, match="seconds=0"):
        frames_for(0, 8)


def test_per_frame_grows_when_fps_drops():
    assert per_frame(2.0, 8) == pytest.approx(2.5)


def test_timing_fps_falls_back_to_reference_when_unthrottled():
    assert timing_fps(0) == REFERENCE_FPS
    assert timing_fps(8) == 8.0


def test_cadence_at_10_fps_reproduces_the_old_frame_counts():
    c = Cadence.at_fps(10)
    assert (c.head_turn_gt, c.eye_rub_gt, c.phone_call_gt) == (5, 5, 8)
    assert (c.head_pitch_frames, c.head_face_lost_frames) == (6, 28)
    assert (c.smoking_every, c.seatbelt_every, c.eye_rub_history) == (15, 30, 5)
    assert c.record_confirm_frames == 3


def test_cadence_at_8_fps_keeps_alert_durations_in_seconds():
    c = Cadence.at_fps(8)
    assert (c.head_turn_gt + 1) / 8 == pytest.approx(0.6, abs=0.07)
    assert (c.phone_call_gt + 1) / 8 == pytest.approx(0.9, abs=0.07)
    assert c.head_face_lost_frames / 8 == pytest.approx(2.8, abs=0.07)
    assert (c.smoking_every, c.seatbelt_every) == (12, 24)
