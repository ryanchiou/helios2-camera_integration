"""Runs the live preview window against the simulated camera, without a display."""

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")
pytest.importorskip("pyqtgraph")

from PySide6 import QtWidgets  # noqa: E402

from scanner.camera import SimulatedCamera, load_capture  # noqa: E402
from scanner.ui.main_window import MainWindow  # noqa: E402


def wait_until(app, condition, timeout_s=15.0):
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out waiting for the UI"
        app.processEvents()
        time.sleep(0.01)


def test_preview_streams_captures_and_shuts_down(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(SimulatedCamera(fps=60, seed=0), tmp_path)
    window.show()
    try:
        wait_until(app, lambda: window.frames_shown >= 5)
        assert "SIM0001" in window.windowTitle()
        assert window._capture_button.isEnabled()
        assert set(window._control_widgets) == {"Scan3dImageAccumulation",
                                                "Scan3dSpatialFilterEnable"}

        window._capture_button.click()
        wait_until(app, lambda: window.last_capture is not None)
        paths = window.last_capture
        assert paths.npz.exists() and paths.metadata.exists() and paths.ply.exists()
        assert load_capture(paths.npz).valid_count > 0

        window._stream_button.click()  # stop
        wait_until(app, lambda: window._stream_button.text() == "Start stream")
        assert not window._capture_button.isEnabled()
        shown = window.frames_shown
        window._stream_button.click()  # start again
        wait_until(app, lambda: window.frames_shown > shown + 2)
    finally:
        window.close()
    assert window._thread.isFinished()
