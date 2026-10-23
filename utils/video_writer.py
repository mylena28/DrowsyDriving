"""Background-thread video writer plus two opt-in recording policies:
face-presence debounced, and risk-event fixed-length clips."""

import os
import queue
import threading
import time

import cv2
import numpy as np

from utils.daily_dir import daily_log_dir


class AsyncVideoWriter:
    """Encodes and writes frames in a background thread so the main loop
    never blocks on disk I/O."""

    def __init__(self):
        self._writer = None
        self._q: "queue.Queue" = queue.Queue(maxsize=10)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self) -> None:
        while True:
            writer, frame = self._q.get()
            writer.write(frame)
            self._q.task_done()

    def open(self, path: str, fourcc: int, fps: float, size: tuple[int, int]) -> bool:
        self._writer = cv2.VideoWriter(path, fourcc, fps, size)
        return self._writer.isOpened()

    def write(self, frame: np.ndarray) -> None:
        if self._writer is not None:
            try:
                self._q.put_nowait((self._writer, frame.copy()))
            except queue.Full:
                pass  # drop frame rather than stall the main loop

    def release(self) -> None:
        if self._writer is not None:
            self._q.join()  # flush all pending frames before closing
            self._writer.release()
            self._writer = None

    def isOpened(self) -> bool:
        return self._writer is not None and self._writer.isOpened()


class _ClipWriter:
    """Shared open/close plumbing for the recorders below: owns the
    AsyncVideoWriter, names each clip under the daily log dir, and prints the
    start/stop lines. Subclasses decide *when* to start and stop via update().
    """

    def __init__(self, snapshot_dir: str, fps: float, suffix: str):
        self._writer = AsyncVideoWriter()
        self._snapshot_dir = snapshot_dir
        self._fps = fps
        self._suffix = suffix
        self.video_path: "str | None" = None

    def _start(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(
            daily_log_dir(self._snapshot_dir), f"{ts}_{self._suffix}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if not self._writer.open(self.video_path, fourcc, self._fps, (w, h)):
            print(f"❌ Falha ao abrir VideoWriter: {self.video_path}")
            self.video_path = None
        else:
            print(f"🎥 Gravação iniciada: {self.video_path}")

    def _stop(self) -> None:
        self._writer.release()
        print(f"🎥 Gravação encerrada: {self.video_path}")
        self.video_path = None

    def release(self) -> None:
        if self._writer.isOpened():
            self._stop()

    def isOpened(self) -> bool:
        return self._writer.isOpened()


class DebouncedRecorder(_ClipWriter):
    """Starts/stops recording based on sustained face presence/absence, so a
    single missed detection doesn't flap recording.

    At ~10 FPS: start_frames=15 ~= 1.5s of presence, stop_frames=50 ~= 5s of
    absence, before toggling.
    """

    def __init__(self, snapshot_dir: str, fps: float = 10.0,
                 start_frames: int = 15, stop_frames: int = 50):
        super().__init__(snapshot_dir, fps, suffix="video")
        self._start_frames = start_frames
        self._stop_frames = stop_frames
        self._present_count = 0
        self._absent_count = 0

    def update(self, frame: np.ndarray, *, face_visible: bool,
               risk_active: bool = False) -> None:
        # risk_active is accepted for a uniform call site but unused here.
        if face_visible:
            self._present_count += 1
            self._absent_count = 0
        else:
            self._absent_count += 1
            self._present_count = 0

        if not self._writer.isOpened() and self._present_count >= self._start_frames:
            self._start(frame)
        elif self._writer.isOpened() and self._absent_count >= self._stop_frames:
            self._stop()

        if self._writer.isOpened():
            self._writer.write(frame)


class RiskClipRecorder(_ClipWriter):
    """Records a fixed-length clip around a confirmed risk event (opt-in via
    `--record risk`).

    To avoid filming on a single-frame false positive, it waits for
    `confirm_frames` consecutive risk frames before opening a clip. Every
    further confirmed frame pushes the stop deadline `clip_seconds` into the
    future, so a second event that happens during an ongoing clip extends that
    same recording instead of cutting it short or opening a second file.

    At ~10 FPS: confirm_frames=3 ~= 0.3s of sustained risk before recording;
    clip_seconds=10 keeps ~10s of footage after the last confirmed risk frame.
    """

    def __init__(self, snapshot_dir: str, fps: float = 10.0,
                 clip_seconds: float = 10.0, confirm_frames: int = 3):
        super().__init__(snapshot_dir, fps, suffix="risk")
        self._clip_seconds = clip_seconds
        self._confirm_frames = confirm_frames
        self._risk_streak = 0
        self._clip_deadline = 0.0

    def update(self, frame: np.ndarray, *, risk_active: bool,
               face_visible: bool = False) -> None:
        # face_visible is accepted for a uniform call site but unused here.
        self._risk_streak = self._risk_streak + 1 if risk_active else 0
        now = time.time()

        if self._risk_streak >= self._confirm_frames:
            # Re-arm on every confirmed frame: overlapping events extend the
            # same clip rather than starting a second recording.
            self._clip_deadline = now + self._clip_seconds
            if not self._writer.isOpened():
                self._start(frame)

        if not self._writer.isOpened():
            return
        self._writer.write(frame)
        if now >= self._clip_deadline:
            self._stop()
            self._risk_streak = 0
