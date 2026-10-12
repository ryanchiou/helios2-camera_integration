"""Live preview window: intensity image, depth map and 3D point cloud."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

from ..camera import Camera, CameraInfo, CapturePaths, ControlSpec, Frame
from .acquisition import AcquisitionThread

log = logging.getLogger(__name__)

pg.setConfigOptions(imageAxisOrder="row-major")

# Depth is a magnitude, so it gets one perceptually uniform ramp (near = bright).
# The depth map and the 3D view share it. Entry 0 is reserved for "no depth".
DEPTH_COLORMAP = pg.colormap.get("viridis")
DEPTH_COLORMAP.reverse()
DEPTH_LUT = np.zeros((256, 3), dtype=np.uint8)
DEPTH_LUT[1:] = DEPTH_COLORMAP.getLookupTable(0.0, 1.0, 255)[:, :3]
POINT_COLOURS = np.ones((256, 4), dtype=np.float32)
POINT_COLOURS[:, :3] = DEPTH_LUT / 255.0

# Square-root tone curve: ToF intensity falls off with distance squared, so a
# linear ramp leaves most of the scene too dark to aim by.
INTENSITY_CURVE = (np.sqrt(np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8)

RAW_COUNTS = np.arange(65536, dtype=np.float32)

# How often the UI checks for a new frame. Kept short because an idle check is
# nearly free and drawing is already paced by the display's refresh.
REFRESH_INTERVAL_MS = 10
CENTRE_WINDOW_PX = 10  # half-size of the patch behind the centre-distance readout
POINT_STEPS = (1, 2, 3, 4, 6, 8)
AXIS_LENGTH_M = 0.1
CLIP_SLIDER_MAX_MM = 3000  # the spin boxes beside the sliders go further, up to 10 m


def to_display(points_m: np.ndarray) -> np.ndarray:
    """Camera frame (X right, Y down, Z forward) to the 3D view's Z-up frame.

    Display only: nothing measured or saved goes through this.
    """
    display = np.empty(points_m.shape, dtype=np.float32)
    display[:, 0] = points_m[:, 0]
    display[:, 1] = points_m[:, 2]
    display[:, 2] = -points_m[:, 1]
    return display


def depth_colour_indices(frame: Frame, near_m: float, far_m: float,
                         shown: np.ndarray | None = None) -> np.ndarray:
    """Each pixel's row in ``DEPTH_LUT``: 0 for no depth, 1-255 from near to far.

    Colouring goes through a table over all 65536 raw Z counts, which is far
    cheaper per frame than arithmetic on every pixel. Pixels outside ``shown``
    (default: the valid ones) get 0, like pixels without depth.
    """
    z_m = (RAW_COUNTS * frame.transform.scale_mm[2] + frame.transform.offset_mm[2]) / 1000.0
    span = max(far_m - near_m, 1e-6)
    table = (1.0 + np.clip((z_m - near_m) * (254.0 / span), 0.0, 254.0)).astype(np.uint8)
    indices = table[frame.raw[..., 2]]
    indices[~(frame.valid if shown is None else shown)] = 0
    return indices


def clip_mask(frame: Frame, z_min_m: float, z_max_m: float) -> np.ndarray:
    """Valid pixels whose depth lies within ``[z_min_m, z_max_m]``.

    Display only: captures are saved from the frame itself, so they keep every point.
    """
    z_m = frame.xyz_m[..., 2]
    return frame.valid & (z_m >= z_min_m) & (z_m <= z_max_m)


def intensity_image(frame: Frame, peak: float) -> np.ndarray:
    """Intensity as 8-bit grey, with ``peak`` counts mapped to white."""
    table = INTENSITY_CURVE[np.minimum(RAW_COUNTS * (255.0 / peak), 255.0).astype(np.uint8)]
    return table[frame.intensity]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, camera: Camera, capture_dir: Path):
        super().__init__()
        self.setWindowTitle("Helios2+ live preview")
        self.resize(1500, 860)

        self._frame: Frame | None = None
        self._depth_range: tuple[float, float] | None = None
        self._intensity_max = 0.0
        self._image_size: tuple[int, int] | None = None
        self._hover: tuple[int, int] | None = None
        self._view_initialised = False
        self._failed = False
        self._control_widgets: dict[str, QtWidgets.QWidget] = {}
        self.frames_shown = 0
        self.last_capture: CapturePaths | None = None

        self._build_ui(camera)

        self._thread = AcquisitionThread(camera, capture_dir, self)
        self._thread.opened.connect(self._on_opened)
        self._thread.controls_changed.connect(self._set_controls)
        self._thread.streaming_changed.connect(self._on_streaming_changed)
        self._thread.capture_saved.connect(self._on_capture_saved)
        self._thread.warning.connect(lambda text: self.statusBar().showMessage(text, 5000))
        self._thread.failed.connect(self._on_failed)
        self._thread.start()

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(REFRESH_INTERVAL_MS)

    # construction ------------------------------------------------------------

    def _build_ui(self, camera: Camera) -> None:
        self._banner = QtWidgets.QLabel()
        self._banner.setWordWrap(True)
        self._banner.setStyleSheet(
            "background: #5c1f1f; color: #ffffff; padding: 8px; font-weight: bold;")
        self._banner.hide()

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_image_views())
        splitter.addWidget(self._build_cloud_view())
        splitter.addWidget(self._build_side_panel(camera))
        splitter.setSizes([520, 700, 280])
        splitter.setStretchFactor(1, 1)

        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._banner)
        layout.addWidget(splitter, 1)
        self.setCentralWidget(central)

        self._fps_label = QtWidgets.QLabel("-- fps")
        self._frame_label = QtWidgets.QLabel("frame --")
        self._valid_label = QtWidgets.QLabel("valid --")
        self._centre_label = QtWidgets.QLabel("centre --")
        self._hover_label = QtWidgets.QLabel("")
        self.statusBar().addWidget(self._hover_label, 1)
        for label in (self._centre_label, self._valid_label, self._frame_label, self._fps_label):
            self.statusBar().addPermanentWidget(label)
        self.statusBar().showMessage("Connecting to the camera...")

    def _build_image_views(self) -> QtWidgets.QWidget:
        self._graphics = pg.GraphicsLayoutWidget()
        self._graphics.ci.layout.setContentsMargins(6, 6, 6, 6)

        self._graphics.addLabel("Intensity", row=0, col=0, justify="left")
        self._intensity_view = self._graphics.addViewBox(
            row=1, col=0, lockAspect=True, invertY=True, enableMenu=False)
        self._intensity_item = pg.ImageItem()
        self._intensity_view.addItem(self._intensity_item)

        self._graphics.addLabel("Depth", row=2, col=0, justify="left")
        self._depth_view = self._graphics.addViewBox(
            row=3, col=0, lockAspect=True, invertY=True, enableMenu=False)
        self._depth_item = pg.ImageItem()
        self._depth_item.setLookupTable(DEPTH_LUT)
        self._depth_view.addItem(self._depth_item)
        self._depth_bar = pg.ColorBarItem(values=(0.0, 1.0), colorMap=DEPTH_COLORMAP,
                                          label="Depth (m)", interactive=False, width=14)
        self._graphics.addItem(self._depth_bar, row=3, col=1)

        # Marks the optical axis, which the centre-distance readout refers to.
        self._centre_marks = []
        for view in (self._intensity_view, self._depth_view):
            mark = pg.ScatterPlotItem(symbol="+", size=16, pen=pg.mkPen("w", width=1))
            view.addItem(mark)
            self._centre_marks.append(mark)

        self._mouse_proxy = pg.SignalProxy(self._graphics.scene().sigMouseMoved,
                                           rateLimit=30, slot=self._on_mouse_moved)
        return self._graphics

    def _build_cloud_view(self) -> QtWidgets.QWidget:
        self._cloud_view = None
        self._cloud_item = None
        try:
            import pyqtgraph.opengl as gl

            view = gl.GLViewWidget()
            view.setBackgroundColor("#0c0c10")
            item = gl.GLScatterPlotItem(pos=np.zeros((1, 3), dtype=np.float32),
                                        color=(1.0, 1.0, 1.0, 1.0), size=2.0, pxMode=True)
            item.setGLOptions("opaque")
            view.addItem(item)
            # Camera axes at the origin: X red, Y green (down), Z blue (forward).
            for axis, colour in ((0, (0.9, 0.25, 0.25, 1)), (1, (0.3, 0.8, 0.3, 1)),
                                 (2, (0.35, 0.55, 1.0, 1))):
                end = np.zeros((2, 3))
                end[1, axis] = AXIS_LENGTH_M
                view.addItem(gl.GLLinePlotItem(pos=to_display(end), color=colour,
                                               width=3, antialias=True))
        except Exception as exc:
            log.warning("3D view unavailable: %s", exc)
            placeholder = QtWidgets.QLabel(f"3D view unavailable (OpenGL): {exc}")
            placeholder.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            placeholder.setWordWrap(True)
            return placeholder

        self._cloud_view, self._cloud_item = view, item
        container = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(view, 1)
        legend = QtWidgets.QLabel(
            "Camera frame at the origin:  X red  ·  Y green (down)  ·  Z blue (forward)."
            "   Drag to orbit, scroll to zoom, Ctrl+drag to pan.")
        legend.setContentsMargins(8, 4, 8, 4)
        layout.addWidget(legend)
        return container

    def _build_side_panel(self, camera: Camera) -> QtWidgets.QWidget:
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)

        camera_box = QtWidgets.QGroupBox("Camera")
        camera_layout = QtWidgets.QVBoxLayout(camera_box)
        self._camera_label = QtWidgets.QLabel("Connecting...")
        self._camera_label.setWordWrap(True)
        self._stream_button = QtWidgets.QPushButton("Stop stream")
        self._stream_button.setEnabled(False)
        self._stream_button.clicked.connect(self._toggle_stream)
        self._capture_button = QtWidgets.QPushButton("Capture frame")
        self._capture_button.setEnabled(False)
        self._capture_button.clicked.connect(self._capture)
        self._native_ply_box = QtWidgets.QCheckBox("Also save Arena-written PLY")
        self._native_ply_box.setToolTip(
            "Has LUCID's Save library write the same frame (in millimetres),\n"
            "to cross-check this application's own decoding.")
        self._native_ply_box.setVisible(camera.supports_native_ply)
        self._capture_label = QtWidgets.QLabel("No captures yet")
        self._capture_label.setWordWrap(True)
        for widget in (self._camera_label, self._stream_button, self._capture_button,
                       self._native_ply_box, self._capture_label):
            camera_layout.addWidget(widget)
        layout.addWidget(camera_box)

        display_box = QtWidgets.QGroupBox("Display")
        form = QtWidgets.QFormLayout(display_box)
        self._auto_range_box = QtWidgets.QCheckBox("Auto depth range")
        self._auto_range_box.setChecked(True)
        self._auto_range_box.toggled.connect(self._on_auto_range_toggled)
        self._near_spin = self._metre_spin(0.30)
        self._far_spin = self._metre_spin(1.50)
        self._step_combo = QtWidgets.QComboBox()
        for step in POINT_STEPS:
            self._step_combo.addItem("every pixel" if step == 1 else f"every {step} pixels", step)
        self._step_combo.setCurrentIndex(POINT_STEPS.index(2))
        self._size_spin = QtWidgets.QDoubleSpinBox()
        self._size_spin.setRange(1.0, 6.0)
        self._size_spin.setSingleStep(0.5)
        self._size_spin.setValue(2.0)
        reset_button = QtWidgets.QPushButton("Reset 3D view")
        reset_button.clicked.connect(self.reset_view)
        form.addRow(self._auto_range_box)
        form.addRow("Near (m)", self._near_spin)
        form.addRow("Far (m)", self._far_spin)
        form.addRow("3D points", self._step_combo)
        form.addRow("Point size", self._size_spin)
        form.addRow(reset_button)
        layout.addWidget(display_box)
        self._on_auto_range_toggled(True)
        layout.addWidget(self._build_clip_box())

        self._controls_box = QtWidgets.QGroupBox("Camera settings")
        self._controls_form = QtWidgets.QFormLayout(self._controls_box)
        layout.addWidget(self._controls_box)
        self._controls_box.hide()

        layout.addStretch(1)
        panel.setMinimumWidth(260)
        return panel

    def _build_clip_box(self) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox("Depth clip")
        box.setToolTip("Hides points outside the range in the depth map and the 3D view.\n"
                       "Display only: captures still save every point.")
        form = QtWidgets.QFormLayout(box)
        self._clip_box = QtWidgets.QCheckBox("Show only points between")
        self._clip_min_slider, self._clip_min_spin = self._clip_slider(0.30)
        self._clip_max_slider, self._clip_max_spin = self._clip_slider(1.50)
        # Keep min <= max by pushing the other end along.
        self._clip_min_spin.valueChanged.connect(
            lambda value: self._clip_max_spin.setValue(max(value, self._clip_max_spin.value())))
        self._clip_max_spin.valueChanged.connect(
            lambda value: self._clip_min_spin.setValue(min(value, self._clip_min_spin.value())))
        self._clip_box.toggled.connect(self._on_clip_toggled)
        form.addRow(self._clip_box)
        form.addRow("Min (m)", self._clip_row(self._clip_min_slider, self._clip_min_spin))
        form.addRow("Max (m)", self._clip_row(self._clip_max_slider, self._clip_max_spin))
        self._on_clip_toggled(False)
        return box

    def _clip_slider(self, value: float) -> tuple[QtWidgets.QSlider, QtWidgets.QDoubleSpinBox]:
        """A millimetre slider tied to a metre spin box."""
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setRange(0, CLIP_SLIDER_MAX_MM)
        slider.setSingleStep(5)
        slider.setPageStep(50)
        spin = self._metre_spin(value)
        spin.setDecimals(3)
        spin.setSingleStep(0.005)
        slider.setValue(round(value * 1000))
        slider.valueChanged.connect(lambda mm: spin.setValue(mm / 1000.0))

        def follow(metres: float) -> None:
            slider.blockSignals(True)
            slider.setValue(min(round(metres * 1000), CLIP_SLIDER_MAX_MM))
            slider.blockSignals(False)
        spin.valueChanged.connect(follow)
        return slider, spin

    @staticmethod
    def _clip_row(slider: QtWidgets.QSlider, spin: QtWidgets.QDoubleSpinBox) -> QtWidgets.QWidget:
        row = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(slider, 1)
        layout.addWidget(spin)
        return row

    def _on_clip_toggled(self, clip: bool) -> None:
        for widget in (self._clip_min_slider, self._clip_min_spin,
                       self._clip_max_slider, self._clip_max_spin):
            widget.setEnabled(clip)

    def _shown(self, frame: Frame) -> np.ndarray:
        if not self._clip_box.isChecked():
            return frame.valid
        return clip_mask(frame, self._clip_min_spin.value(), self._clip_max_spin.value())

    @staticmethod
    def _metre_spin(value: float) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(0.0, 10.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    # acquisition-thread events -----------------------------------------------

    def _on_opened(self, info: CameraInfo, controls: list[ControlSpec]) -> None:
        self.setWindowTitle(f"Helios2+ live preview - {info.model} ({info.serial})")
        details = [f"{info.model}", f"Serial {info.serial}"]
        if info.ip:
            details.append(f"IP {info.ip}")
        if info.simulated:
            details.append("Synthetic data, no camera attached")
        self._camera_label.setText("\n".join(details))
        self._set_controls(controls)
        self.statusBar().showMessage("Camera connected", 3000)

    def _on_streaming_changed(self, streaming: bool) -> None:
        self._stream_button.setEnabled(not self._failed)
        self._stream_button.setText("Stop stream" if streaming else "Start stream")
        self._capture_button.setEnabled(streaming and not self._failed)

    def _on_capture_saved(self, paths: CapturePaths) -> None:
        self.last_capture = paths
        self._capture_label.setText(f"Saved {paths.npz.stem}\nin {paths.npz.parent}")
        self.statusBar().showMessage(f"Saved {paths.npz.stem}", 5000)

    def _on_failed(self, message: str) -> None:
        self._failed = True
        self._banner.setText(f"Camera error: {message}")
        self._banner.show()
        self._camera_label.setText("Not connected")
        self._stream_button.setEnabled(False)
        self._capture_button.setEnabled(False)
        self.statusBar().showMessage("Camera error")

    # user actions --------------------------------------------------------------

    def _toggle_stream(self) -> None:
        self._thread.set_streaming(self._stream_button.text().startswith("Start"))

    def _capture(self) -> None:
        self._thread.request_capture(native_ply=self._native_ply_box.isChecked())

    def _on_auto_range_toggled(self, auto: bool) -> None:
        self._near_spin.setEnabled(not auto)
        self._far_spin.setEnabled(not auto)

    def reset_view(self) -> None:
        """Look at the scene from just above and behind the camera."""
        if self._cloud_view is None or self._frame is None or not self._frame.valid.any():
            return
        depth = float(np.median(self._frame.xyz_m[..., 2][self._frame.valid]))
        self._cloud_view.setCameraPosition(pos=QtGui.QVector3D(0.0, depth, 0.0),
                                           distance=1.15 * depth, elevation=20, azimuth=-75)

    def _set_controls(self, controls: list[ControlSpec]) -> None:
        """Rebuild the settings form from what the camera reports."""
        while self._controls_form.rowCount():
            self._controls_form.removeRow(0)
        self._control_widgets.clear()
        for spec in controls:
            widget = self._control_widget(spec)
            widget.setEnabled(spec.writable)
            self._controls_form.addRow(spec.name, widget)
            self._control_widgets[spec.name] = widget
        self._controls_box.setVisible(bool(controls))

    def _control_widget(self, spec: ControlSpec) -> QtWidgets.QWidget:
        send = self._thread.set_control
        if spec.kind == "enum":
            widget = QtWidgets.QComboBox()
            widget.addItems(spec.options)
            widget.setCurrentText(str(spec.value))
            widget.textActivated.connect(lambda text, name=spec.name: send(name, text))
        elif spec.kind == "bool":
            widget = QtWidgets.QCheckBox()
            widget.setChecked(bool(spec.value))
            widget.clicked.connect(lambda checked, name=spec.name: send(name, checked))
        else:
            is_int = spec.kind == "int"
            widget = QtWidgets.QSpinBox() if is_int else QtWidgets.QDoubleSpinBox()
            cast = int if is_int else float
            if spec.minimum is not None and spec.maximum is not None:
                limit = 2 ** 31 - 1  # QSpinBox holds 32-bit values
                widget.setRange(cast(max(spec.minimum, -limit)), cast(min(spec.maximum, limit)))
            widget.setValue(cast(spec.value))
            widget.setKeyboardTracking(False)
            widget.editingFinished.connect(
                lambda name=spec.name, widget=widget: send(name, widget.value()))
        return widget

    # rendering -----------------------------------------------------------------

    def _refresh(self) -> None:
        frame = self._thread.take_latest()
        if frame is None:
            if not self._thread.fps:
                self._fps_label.setText("-- fps")
            return
        self._frame = frame
        self.frames_shown += 1
        shown = self._shown(frame)
        near, far = self._update_depth_range(frame, shown)
        colour_indices = depth_colour_indices(frame, near, far, shown)
        self._draw_images(frame, colour_indices)
        self._depth_bar.setLevels((near, far))
        self._draw_cloud(frame, colour_indices, shown)
        self._update_readouts(frame)
        if not self._view_initialised:
            self._view_initialised = True
            self._intensity_view.autoRange(padding=0)
            self._depth_view.autoRange(padding=0)
            self.reset_view()

    def _update_depth_range(self, frame: Frame, shown: np.ndarray) -> tuple[float, float]:
        if not self._auto_range_box.isChecked():
            near, far = self._near_spin.value(), self._far_spin.value()
            return (near, far) if far > near else (near, near + 0.01)
        sample = frame.xyz_m[::4, ::4, 2][shown[::4, ::4]]
        if sample.size:
            near, far = np.percentile(sample, (1, 99))
            if self._depth_range is not None:  # ease towards the new range to avoid flicker
                near = 0.8 * self._depth_range[0] + 0.2 * near
                far = 0.8 * self._depth_range[1] + 0.2 * far
            self._depth_range = (float(near), float(far))
            self._near_spin.setValue(near)
            self._far_spin.setValue(far)
        return self._depth_range or (0.0, 1.0)

    def _draw_images(self, frame: Frame, colour_indices: np.ndarray) -> None:
        peak = float(np.percentile(frame.intensity[::4, ::4], 99.5))
        self._intensity_max = peak if not self._intensity_max else (
            0.8 * self._intensity_max + 0.2 * peak)
        # 8-bit images with no levels take pyqtgraph's direct-to-QImage path.
        self._intensity_item.setImage(intensity_image(frame, max(self._intensity_max, 1.0)),
                                      autoLevels=False)
        self._depth_item.setImage(colour_indices, autoLevels=False)

        if self._image_size != (frame.width, frame.height):
            self._image_size = (frame.width, frame.height)
            for mark in self._centre_marks:
                mark.setData(pos=[(frame.width / 2, frame.height / 2)])

    def _draw_cloud(self, frame: Frame, colour_indices: np.ndarray, shown: np.ndarray) -> None:
        if self._cloud_item is None:
            return
        step = self._step_combo.currentData()
        keep = shown[::step, ::step]
        points = frame.xyz_m[::step, ::step][keep]
        if not len(points):
            self._cloud_item.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return
        self._cloud_item.setData(pos=to_display(points),
                                 color=POINT_COLOURS[colour_indices[::step, ::step][keep]],
                                 size=self._size_spin.value())

    def _update_readouts(self, frame: Frame) -> None:
        self._fps_label.setText(f"{self._thread.fps:.1f} fps")
        self._frame_label.setText(f"frame {frame.frame_id}")
        self._valid_label.setText(f"valid {100 * frame.valid_fraction:.1f}%")

        row, col, half = frame.height // 2, frame.width // 2, CENTRE_WINDOW_PX
        window = (slice(row - half, row + half + 1), slice(col - half, col + half + 1))
        depths = frame.xyz_m[..., 2][window][frame.valid[window]]
        self._centre_label.setText(
            f"centre {1000 * np.median(depths):.1f} mm" if depths.size else "centre no data")
        self._update_hover_label()

    def _on_mouse_moved(self, event) -> None:
        position = event[0]
        self._hover = None
        if self._frame is not None:
            for view in (self._intensity_view, self._depth_view):
                if view.sceneBoundingRect().contains(position):
                    point = view.mapSceneToView(position)
                    col, row = int(np.floor(point.x())), int(np.floor(point.y()))
                    if 0 <= row < self._frame.height and 0 <= col < self._frame.width:
                        self._hover = (row, col)
        self._update_hover_label()

    def _update_hover_label(self) -> None:
        if self._hover is None or self._frame is None:
            self._hover_label.setText("")
            return
        row, col = self._hover
        frame = self._frame
        text = f"pixel ({col}, {row})   intensity {int(frame.intensity[row, col])}"
        if frame.valid[row, col]:
            x, y, z = 1000 * frame.xyz_m[row, col]
            text += f"   X {x:.1f}  Y {y:.1f}  Z {z:.1f} mm"
        else:
            text += "   no depth"
        self._hover_label.setText(text)

    # lifecycle ---------------------------------------------------------------

    def save_screenshot(self, path: Path) -> None:
        self.grab().save(str(path))

    def closeEvent(self, event) -> None:
        self._timer.stop()
        self._thread.shutdown()
        self._thread.wait(5000)
        super().closeEvent(event)
