"""Background-thread video writer with face-presence-debounced start/stop."""

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


class DebouncedRecorder:
    """Starts/stops an AsyncVideoWriter based on sustained face
    presence/absence, so a single missed detection doesn't flap recording.

    At ~20 FPS: start_frames=15 ~= 0.75s of presence, stop_frames=50 ~= 2.5s
    of absence, before toggling.
    """

    def __init__(self, snapshot_dir: str, start_frames: int = 15, stop_frames: int = 50):
        self._writer = AsyncVideoWriter()
        self._snapshot_dir = snapshot_dir
        self._start_frames = start_frames
        self._stop_frames = stop_frames
        self._present_count = 0
        self._absent_count = 0
        self.video_path: str | None = None

    def update(self, frame: np.ndarray, face_visible: bool) -> None:
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

    def _start(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(daily_log_dir(self._snapshot_dir), f"{ts}_video.avi")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        if not self._writer.open(self.video_path, fourcc, 20.0, (w, h)):
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
