"""Desktop user interface: the live camera preview, and later the merged-cloud viewer."""

from __future__ import annotations

import sys
from pathlib import Path

from ..camera import Camera


def run_preview(camera: Camera, capture_dir: Path, screenshot: Path | None = None,
                duration_s: float | None = None) -> int:
    """Open the live preview window and run until it is closed.

    ``duration_s`` closes the window automatically, saving ``screenshot`` first
    if one was requested; both exist for unattended checks.
    """
    from PySide6 import QtCore, QtGui, QtWidgets

    from .main_window import MainWindow

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setPalette(_dark_palette(QtGui))

    window = MainWindow(camera, capture_dir)
    window.show()

    if duration_s is not None:
        def finish():
            if screenshot is not None:
                window.save_screenshot(screenshot)
            window.close()
        QtCore.QTimer.singleShot(int(duration_s * 1000), finish)

    return app.exec()


def _dark_palette(QtGui):
    """A dark theme, so the panels match the image and 3D views."""
    roles = QtGui.QPalette.ColorRole
    colours = {
        roles.Window: "#25262b", roles.WindowText: "#e6e6e6",
        roles.Base: "#1b1c20", roles.AlternateBase: "#25262b",
        roles.Text: "#e6e6e6", roles.Button: "#34353b", roles.ButtonText: "#e6e6e6",
        roles.ToolTipBase: "#34353b", roles.ToolTipText: "#e6e6e6",
        roles.Highlight: "#3d6fb4", roles.HighlightedText: "#ffffff",
        roles.PlaceholderText: "#8a8a8a",
    }
    palette = QtGui.QPalette()
    for role, colour in colours.items():
        palette.setColor(role, QtGui.QColor(colour))
    disabled = QtGui.QPalette.ColorGroup.Disabled
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(disabled, role, QtGui.QColor("#7a7a7a"))
    return palette
