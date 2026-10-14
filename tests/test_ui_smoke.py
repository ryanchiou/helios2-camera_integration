"""Runs the live preview window against the simulated camera, without a display."""

import os
import time

import numpy as np
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
        assert "Scan3dFlyingPixelsDistanceThreshold" in window._control_widgets
        assert window._setting_forms["Noise filters"][0].isVisibleTo(window)
        assert not window._setting_forms["Other"][0].isVisibleTo(window)

        window._capture_button.click()
        wait_until(app, lambda: window.last_capture is not None)
        paths = window.last_capture
        assert paths.folder.parent == tmp_path
        assert paths.npz.exists() and paths.metadata.exists() and paths.ply.exists()
        assert paths.screenshot.parent == paths.folder and paths.screenshot.stat().st_size > 0
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


def test_settings_follow_the_camera_state(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(SimulatedCamera(fps=60, seed=0), tmp_path)
    window.show()
    try:
        wait_until(app, lambda: window.frames_shown >= 2)
        widgets = window._control_widgets
        # A threshold is greyed out while its filter is off.
        assert not widgets["Scan3dFlyingPixelsDistanceThreshold"].isEnabled()
        assert widgets["Scan3dConfidenceThresholdMin"].isEnabled()
        widgets["Scan3dFlyingPixelsRemovalEnable"].click()
        wait_until(app, lambda: widgets["Scan3dFlyingPixelsDistanceThreshold"].isEnabled())

        # HDR takes over from accumulation, which the camera then locks.
        window._thread.set_control("Scan3dHDRMode", "StandardHDR")
        wait_until(app, lambda: not widgets["Scan3dImageAccumulation"].isEnabled())
        assert "Locked" in widgets["Scan3dImageAccumulation"].toolTip()
        shown = window.frames_shown
        wait_until(app, lambda: window.frames_shown > shown + 2)
    finally:
        window.close()


def test_distance_limit_sliders_drop_points_in_the_camera(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(SimulatedCamera(fps=60, seed=0), tmp_path)
    window.show()
    try:
        wait_until(app, lambda: window.frames_shown >= 2)
        widgets = window._control_widgets

        def slider(name):
            return widgets[name].findChild(QtWidgets.QSlider)

        def spin(name):
            return widgets[name].findChild(QtWidgets.QSpinBox)

        # Raising the minimum past the maximum pushes the maximum along first.
        slider("Scan3dDistanceMax").setValue(880)
        wait_until(app, lambda: spin("Scan3dDistanceMax").value() == 880)
        slider("Scan3dDistanceMin").setValue(900)
        wait_until(app, lambda: spin("Scan3dDistanceMin").value() == 900
                   and spin("Scan3dDistanceMax").value() == 900)

        # Dropped in the camera, so the saved capture lacks them too.
        slider("Scan3dDistanceMin").setValue(0)
        wait_until(app, lambda: spin("Scan3dDistanceMin").value() == 0)
        window._capture_button.click()
        wait_until(app, lambda: window.last_capture is not None)
        saved = load_capture(window.last_capture.npz)
        assert np.linalg.norm(saved.points_m(), axis=1).max() <= 0.901
    finally:
        window.close()


def test_depth_clip_hides_points_outside_the_range_but_captures_keep_them(tmp_path):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = MainWindow(SimulatedCamera(fps=60, seed=0), tmp_path)
    window.show()
    try:
        wait_until(app, lambda: window.frames_shown >= 3)
        assert not window._clip_min_slider.isEnabled()

        # The simulated table is at 0.90 m; the box and sphere stand above it.
        window._clip_box.setChecked(True)
        window._clip_min_slider.setValue(700)
        window._clip_max_spin.setValue(0.85)
        assert window._clip_min_spin.value() == 0.70
        assert window._clip_max_slider.value() == 850
        shown_before = window.frames_shown
        wait_until(app, lambda: window.frames_shown > shown_before + 2)

        frame = window._frame
        shown = window._shown(frame)
        depth = frame.xyz_m[..., 2][shown]
        assert 0 < shown.sum() < frame.valid.sum()
        assert depth.min() >= 0.70 and depth.max() <= 0.85

        # Dragging one end past the other pushes the other along.
        window._clip_min_spin.setValue(0.90)
        assert window._clip_max_spin.value() == 0.90
        window._clip_max_spin.setValue(0.85)
        assert window._clip_min_spin.value() == 0.85

        window._capture_button.click()
        wait_until(app, lambda: window.last_capture is not None)
        saved = load_capture(window.last_capture.npz)
        assert saved.xyz_m[..., 2][saved.valid].max() > 0.86
    finally:
        window.close()
