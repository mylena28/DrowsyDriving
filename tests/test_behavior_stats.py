import pytest

from utils.behavior_stats import (
    FREQUENCY_WINDOW_SECS, TRACKED_BEHAVIORS, BehaviorStat, BehaviorStatsTracker, EpisodeTimer,
)


def test_episode_timer_accumulates_while_active():
    timer = EpisodeTimer()
    timer.update(True, dt=0.5, now=1.0)
    timer.update(True, dt=0.5, now=1.5)
    assert timer.stat() == BehaviorStat(active_secs=1.0, recent_count=1)


def test_episode_timer_resets_duration_but_keeps_count_when_inactive():
    timer = EpisodeTimer()
    timer.update(True, dt=1.0, now=1.0)
    timer.update(False, dt=1.0, now=2.0)
    assert timer.stat() == BehaviorStat(active_secs=0.0, recent_count=1)


def test_episode_timer_forgets_starts_outside_window():
    timer = EpisodeTimer()
    timer.update(True, dt=0.0, now=0.0)
    timer.update(False, dt=0.0, now=1.0)
    timer.update(False, dt=0.0, now=FREQUENCY_WINDOW_SECS + 1.0)
    assert timer.stat().recent_count == 0


def test_tracker_first_update_has_zero_duration():
    tracker = BehaviorStatsTracker()
    tracker.update({"no_seatbelt": True}, now=100.0)
    assert tracker.stats()["no_seatbelt"] == BehaviorStat(active_secs=0.0, recent_count=1)


def test_tracker_reports_every_tracked_behavior():
    tracker = BehaviorStatsTracker()
    tracker.update({}, now=0.0)
    assert set(tracker.stats()) == set(TRACKED_BEHAVIORS)


def test_tracker_rejects_unknown_behavior():
    with pytest.raises(ValueError, match="sleeping"):
        BehaviorStatsTracker().update({"sleeping": True}, now=0.0)
