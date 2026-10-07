"""Background acquisition, so the UI thread never blocks on the camera."""

from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any

from PySide6 import QtCore

from ..camera import (Camera, CameraError, Frame, FrameTimeoutError,
                      IncompleteFrameError, capture_name, save_capture)

log = logging.getLogger(__name__)


class AcquisitionThread(QtCore.QThread):
    """Owns the camera: opens it, streams frames and runs requests from the UI.

    The UI polls :meth:`take_latest` on its own timer, so a slow redraw skips
    frames instead of building a backlog.
    """

    opened = QtCore.Signal(object, object)  # CameraInfo, list[ControlSpec]
    controls_changed = QtCore.Signal(object)  # list[ControlSpec]
    streaming_changed = QtCore.Signal(bool)
    capture_saved = QtCore.Signal(object)  # CapturePaths
    warning = QtCore.Signal(str)
    failed = QtCore.Signal(str)  # the camera is unusable and the thread is exiting

    def __init__(self, camera: Camera, capture_dir: Path, parent=None):
        super().__init__(parent)
        self._camera = camera
        self._capture_dir = Path(capture_dir)
        self._commands: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._latest: Frame | None = None
        self._stop = threading.Event()
        self._capture_native: bool | None = None  # None = no capture pending
        self._last_frame_time = 0.0
        self.fps = 0.0
        self.frames_received = 0
        self.frames_dropped = 0

    # called from the UI thread -------------------------------------------------

    def take_latest(self) -> Frame | None:
        """The newest frame not yet taken, or None."""
        with self._lock:
            frame, self._latest = self._latest, None
        return frame

    def set_streaming(self, streaming: bool) -> None:
        self._commands.put(("stream", streaming))

    def set_control(self, name: str, value: Any) -> None:
        self._commands.put(("control", (name, value)))

    def request_capture(self, native_ply: bool = False) -> None:
        self._commands.put(("capture", native_ply))

    def shutdown(self) -> None:
        self._stop.set()

    # acquisition thread ------------------------------------------------------

    def run(self) -> None:
        camera = self._camera
        try:
            camera.open()
            self.opened.emit(camera.info, camera.controls())
            camera.start()
            self.streaming_changed.emit(True)
            while not self._stop.is_set():
                self._run_commands()
                if camera.is_streaming:
                    self._acquire()
                else:
                    time.sleep(0.05)
        except CameraError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            log.exception("Acquisition thread crashed")
            self.failed.emit(f"Unexpected error: {exc}")
        finally:
            try:
                camera.close()
            except Exception:
                log.exception("Closing the camera failed")
            self.streaming_changed.emit(False)

    def _run_commands(self) -> None:
        camera = self._camera
        while True:
            try:
                command, argument = self._commands.get_nowait()
            except queue.Empty:
                return
            if command == "stream":
                camera.start() if argument else camera.stop()
                self.fps = 0.0
                self.streaming_changed.emit(camera.is_streaming)
            elif command == "control":
                try:
                    camera.set_control(*argument)
                except CameraError as exc:
                    self.warning.emit(str(exc))
                self.controls_changed.emit(camera.controls())
            elif command == "capture":
                if camera.is_streaming:
                    self._capture_native = bool(argument)
                else:
                    self.warning.emit("Start the stream before capturing")

    def _acquire(self) -> None:
        camera = self._camera
        name = native_ply = None
        if self._capture_native is not None:
            name = capture_name()
            if self._capture_native and camera.supports_native_ply:
                native_ply = self._capture_dir / f"{name}_arena.ply"
        try:
            frame = camera.grab(timeout_s=1.0, native_ply=native_ply)
        except FrameTimeoutError:
            self.fps = 0.0
            self.warning.emit("No frames arriving from the camera")
            return
        except IncompleteFrameError as exc:
            self.frames_dropped += 1
            self.warning.emit(str(exc))
            return

        now = time.monotonic()
        if self._last_frame_time and now > self._last_frame_time:
            rate = 1.0 / (now - self._last_frame_time)
            self.fps = rate if self.fps == 0.0 else 0.9 * self.fps + 0.1 * rate
        self._last_frame_time = now
        self.frames_received += 1
        with self._lock:
            self._latest = frame

        if name is not None:
            self._capture_native = None
            if "native_ply_error" in frame.extra:
                self.warning.emit(f"Arena PLY export failed: {frame.extra['native_ply_error']}")
                native_ply = None
            try:
                paths = save_capture(frame, self._capture_dir, name, camera_info=camera.info,
                                     settings=camera.read_settings(), native_ply=native_ply)
            except (OSError, CameraError) as exc:
                self.warning.emit(f"Capture could not be saved: {exc}")
            else:
                self.capture_saved.emit(paths)
