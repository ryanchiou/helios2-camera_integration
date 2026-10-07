"""The interface every camera backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .config import CameraSettings
from .types import CameraInfo, ControlSpec, Frame


class Camera(ABC):
    """A depth camera that delivers decoded :class:`Frame` objects.

    Typical use::

        with HeliosCamera() as camera:
            camera.start()                 # free-running, for live preview
            frame = camera.grab()

            camera.stop()
            camera.start(triggered=True)   # one frame per software trigger
            frame = camera.trigger()

    A camera object must be used from one thread at a time.
    """

    supports_native_ply = False

    def __init__(self, settings: CameraSettings | None = None):
        self.settings = settings or CameraSettings()
        self.info: CameraInfo | None = None

    # lifecycle ---------------------------------------------------------------

    @abstractmethod
    def open(self) -> None:
        """Connect to the camera and apply ``self.settings``."""

    @abstractmethod
    def close(self) -> None:
        """Stop streaming and release the camera. Safe to call more than once."""

    @abstractmethod
    def start(self, triggered: bool = False) -> None:
        """Start streaming, free-running or waiting for software triggers."""

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming. Safe to call when not streaming."""

    @property
    @abstractmethod
    def is_open(self) -> bool: ...

    @property
    @abstractmethod
    def is_streaming(self) -> bool: ...

    # acquisition -------------------------------------------------------------

    @abstractmethod
    def grab(self, timeout_s: float = 2.0, discard: int = 0,
             native_ply: Path | None = None) -> Frame:
        """Return the next frame of a free-running stream.

        ``discard`` drops that many frames first, which guarantees the returned
        frame was exposed after the call (for example after the robot settled).
        ``native_ply`` additionally has the vendor library write the same frame
        to a PLY file, on backends where ``supports_native_ply`` is true.
        """

    @abstractmethod
    def trigger(self, timeout_s: float = 2.0, native_ply: Path | None = None) -> Frame:
        """Fire one software trigger and return the resulting frame."""

    # settings ----------------------------------------------------------------

    @abstractmethod
    def controls(self) -> list[ControlSpec]:
        """Adjustable features this camera exposes, with their current values."""

    @abstractmethod
    def set_control(self, name: str, value: Any) -> None:
        """Change one feature, restarting the stream if the camera requires it."""

    @abstractmethod
    def read_settings(self) -> dict[str, Any]:
        """Current values of the features that affect a capture, for metadata."""

    # context manager ---------------------------------------------------------

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.close()
        return False
