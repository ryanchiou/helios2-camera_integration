"""3D view whose mouse controls act about the point under the cursor."""

from __future__ import annotations

from math import radians, tan

import numpy as np
import pyqtgraph.opengl as gl
from PySide6 import QtCore, QtGui

PICK_RADIUS_PX = 8  # how far from the cursor a cloud point still counts as under it
WHEEL_ZOOM = 0.999  # distance factor per 1/8 degree of wheel turn, as in pyqtgraph

LEFT = QtCore.Qt.MouseButton.LeftButton
MIDDLE = QtCore.Qt.MouseButton.MiddleButton
CTRL = QtCore.Qt.KeyboardModifier.ControlModifier


class CursorGLViewWidget(gl.GLViewWidget):
    """A GLViewWidget that orbits, zooms and pans about the point under the cursor.

    pyqtgraph turns and zooms about the view's centre, which stays wherever it
    was last put. Here a drag orbits about the cloud point under the cursor when
    the drag began, the wheel zooms towards the point under the cursor, and a
    pan keeps the grabbed point under the cursor. Over empty space the point is
    taken on the cursor's ray at the centre's depth.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pick_item: gl.GLScatterPlotItem | None = None  # the cloud to pick from
        self._pivot: np.ndarray | None = None

    # geometry ------------------------------------------------------------------

    def _view_transform(self) -> np.ndarray:
        """World to eye coordinates, 4x4. The eye looks down its -Z axis."""
        return np.array(self.viewMatrix().copyDataTo(), dtype=np.float64).reshape(4, 4)

    def _focal_px(self) -> float:
        return self.width() / (2.0 * tan(0.5 * radians(self.opts["fov"])))

    def _centre(self) -> np.ndarray:
        return np.array(self.opts["center"].toTuple(), dtype=np.float64)

    def _set_centre(self, centre: np.ndarray) -> None:
        self.opts["center"] = QtGui.QVector3D(*(float(c) for c in centre))

    def pick(self, pos: QtCore.QPointF) -> np.ndarray:
        """World point under widget position ``pos``.

        The front-most cloud point within ``PICK_RADIUS_PX`` of it, or else the
        point on its ray at the depth of the view's centre.
        """
        view = self._view_transform()
        focal = self._focal_px()
        x = pos.x() - 0.5 * self.width()
        y = 0.5 * self.height() - pos.y()

        points = None if self.pick_item is None else self.pick_item.pos
        if points is not None and len(points):
            eye = points @ view[:3, :3].T + view[:3, 3]
            ahead = np.flatnonzero(eye[:, 2] < -1e-6)
            eye = eye[ahead]
            depth = -eye[:, 2]
            dx = focal * eye[:, 0] / depth - x
            dy = focal * eye[:, 1] / depth - y
            near = np.flatnonzero(dx * dx + dy * dy <= PICK_RADIUS_PX ** 2)
            if near.size:
                return points[ahead[near[np.argmin(depth[near])]]].astype(np.float64)

        depth = self.opts["distance"]
        eye_point = np.array([x * depth / focal, y * depth / focal, -depth, 1.0])
        return (np.linalg.inv(view) @ eye_point)[:3]

    # moves -----------------------------------------------------------------------

    def orbit_about(self, pivot: np.ndarray, azim: float, elev: float) -> None:
        """Orbit as ``orbit`` does, but about ``pivot``, which stays put on screen."""
        before = self._view_transform()[:3, :3]
        self.orbit(azim, elev)
        after = self._view_transform()[:3, :3]
        turn = after.T @ before  # the same turn, as a rotation of the world
        self._set_centre(pivot + turn @ (self._centre() - pivot))
        self.update()

    def zoom_about(self, pivot: np.ndarray, factor: float) -> None:
        """Scale the eye's distance from ``pivot``, which stays put on screen."""
        self._set_centre(pivot + factor * (self._centre() - pivot))
        self.opts["distance"] *= factor
        self.update()

    def pan_with(self, pivot: np.ndarray, dx: float, dy: float) -> None:
        """Shift the view so ``pivot`` moves by (dx, dy) pixels on screen."""
        view = self._view_transform()
        depth = -(view[2, :3] @ pivot + view[2, 3])
        if depth <= 0:
            depth = self.opts["distance"]
        scale = depth / self._focal_px()
        right, up = view[0, :3], view[1, :3]
        self._set_centre(self._centre() - scale * (dx * right - dy * up))
        self.update()

    # events ----------------------------------------------------------------------

    def mousePressEvent(self, ev) -> None:
        super().mousePressEvent(ev)
        self._pivot = self.pick(ev.position())

    def mouseMoveEvent(self, ev) -> None:
        if self._pivot is None:
            super().mouseMoveEvent(ev)
            return
        pos = ev.position()
        diff = pos - self.mousePos
        self.mousePos = pos
        buttons, ctrl = ev.buttons(), bool(ev.modifiers() & CTRL)
        if buttons == LEFT and not ctrl:
            self.orbit_about(self._pivot, -diff.x(), diff.y())
        elif buttons in (LEFT, MIDDLE):
            self.pan_with(self._pivot, diff.x(), diff.y())

    def mouseReleaseEvent(self, ev) -> None:
        if not ev.buttons():
            self._pivot = None

    def wheelEvent(self, ev) -> None:
        if ev.modifiers() & CTRL:  # pyqtgraph's field-of-view zoom
            super().wheelEvent(ev)
            return
        delta = ev.angleDelta().x() or ev.angleDelta().y()
        self.zoom_about(self.pick(ev.position()), WHEEL_ZOOM ** delta)
