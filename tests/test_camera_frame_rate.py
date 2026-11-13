from utils.camera import MAX_CAMERA_FPS, camera_frame_rate


def test_camera_runs_at_twice_the_processing_rate():
    assert camera_frame_rate(10) == 20.0


def test_camera_rate_is_capped_at_the_previous_fixed_30_fps():
    assert camera_frame_rate(25) == MAX_CAMERA_FPS


def test_unthrottled_loop_keeps_the_maximum_camera_rate():
    assert camera_frame_rate(0) == MAX_CAMERA_FPS
