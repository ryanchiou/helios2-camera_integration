"""LUCID Helios2+ backend, built on LUCID's ``arena_api`` package.

``arena_api`` is imported lazily because importing it loads the Arena SDK's
native libraries; the rest of the project stays usable on machines without the
SDK. Only one process can control a camera at a time, so ArenaView must be
disconnected from the camera before this class opens it.
"""

from __future__ import annotations

import ctypes
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .base import Camera
from .config import TUNABLE_NODES, CameraSettings
from .decode import frame_from_raw
from .types import (PIXEL_FORMAT, CameraError, CameraInfo, CameraNotFoundError,
                    ControlSpec, CoordinateTransform, Frame, FrameTimeoutError,
                    IncompleteFrameError)

log = logging.getLogger(__name__)

_AXES = ("CoordinateA", "CoordinateB", "CoordinateC")
_BYTES_PER_PIXEL = 8
_NODE_KINDS = {"NodeEnumeration": "enum", "NodeBoolean": "bool",
               "NodeInteger": "int", "NodeFloat": "float"}
_COERCE = {"enum": str, "bool": bool, "int": int, "float": float}


def _system():
    try:
        from arena_api.system import system
    except Exception as exc:
        raise CameraError(
            "arena_api could not be loaded. Install the Arena SDK and the "
            "arena_api wheel (see README.md).") from exc
    return system


def _info_from_dict(device_info: dict) -> CameraInfo:
    return CameraInfo(model=str(device_info.get("model", "")),
                      serial=str(device_info.get("serial", "")),
                      firmware=str(device_info.get("version", "")),
                      ip=str(device_info.get("ip", "")),
                      mac=str(device_info.get("mac", "")))


class HeliosCamera(Camera):
    """A Helios2+ ToF camera streaming Coord3D_ABCY16 frames."""

    supports_native_ply = True

    def __init__(self, serial: str | None = None, settings: CameraSettings | None = None,
                 num_buffers: int = 4, armed_timeout_s: float = 5.0):
        super().__init__(settings)
        self._serial = None if serial is None else str(serial)
        self._num_buffers = num_buffers
        self.armed_timeout_s = armed_timeout_s
        self._device = None
        self._streaming = False
        self._triggered = False
        self._transform: CoordinateTransform | None = None
        self._controls: list[ControlSpec] = []

    @staticmethod
    def discover() -> list[CameraInfo]:
        """Cameras visible on the network right now."""
        return [_info_from_dict(d) for d in _system().device_infos]

    # lifecycle ---------------------------------------------------------------

    def open(self) -> None:
        if self._device is not None:
            return
        system = _system()
        chosen = self._select(system.device_infos)
        try:
            self._device = system.create_device(chosen)[0]
        except Exception as exc:
            raise CameraError(f"Could not open camera {chosen.get('serial')}: {exc}") from exc
        self.info = _info_from_dict(chosen)
        try:
            self._configure()
        except Exception:
            self.close()
            raise
        log.info("Opened %s (serial %s)", self.info.model, self.info.serial)

    def close(self) -> None:
        if self._device is None:
            return
        try:
            self.stop()
        finally:
            device, self._device = self._device, None
            try:
                _system().destroy_device(device)
            except Exception:
                log.exception("Releasing the camera failed")

    def start(self, triggered: bool = False) -> None:
        self._require_open()
        if self._streaming:
            if triggered == self._triggered:
                return
            self.stop()
        self._set_trigger_mode(triggered)
        try:
            self._device.start_stream(self._num_buffers)
        except Exception as exc:
            raise CameraError(f"Could not start the stream: {exc}") from exc
        self._streaming = True
        self._triggered = triggered

    def stop(self) -> None:
        if self._device is not None and self._streaming:
            self._streaming = False
            self._device.stop_stream()

    @property
    def is_open(self) -> bool:
        return self._device is not None

    @property
    def is_streaming(self) -> bool:
        return self._streaming

    # acquisition -------------------------------------------------------------

    def grab(self, timeout_s: float = 2.0, discard: int = 0,
             native_ply: Path | None = None) -> Frame:
        self._require_streaming()
        if self._triggered:
            raise CameraError("The stream is waiting for triggers; call trigger() instead")
        for _ in range(discard):
            self._release(self._next_buffer(timeout_s))
        return self._read_frame(timeout_s, native_ply)

    def trigger(self, timeout_s: float = 2.0, native_ply: Path | None = None) -> Frame:
        self._require_streaming()
        if not self._triggered:
            raise CameraError("Call start(triggered=True) before trigger()")
        # A frame left over from an earlier, failed trigger must not be paired
        # with this one.
        self._flush()
        self._wait_until_armed()
        try:
            self._node("TriggerSoftware").execute()
        except Exception as exc:
            raise CameraError(f"Software trigger failed: {exc}") from exc
        return self._read_frame(timeout_s, native_ply)

    def _next_buffer(self, timeout_s: float):
        try:
            return self._device.get_buffer(timeout=max(0, int(timeout_s * 1000)))
        except TimeoutError as exc:
            raise FrameTimeoutError(f"No frame arrived within {timeout_s:g} s") from exc
        except Exception as exc:
            raise CameraError(f"Frame retrieval failed: {exc}") from exc

    def _release(self, buffer) -> None:
        self._device.requeue_buffer(buffer)

    def _read_frame(self, timeout_s: float, native_ply: Path | None) -> Frame:
        buffer = self._next_buffer(timeout_s)
        native_ply_error = None
        try:
            host_time = time.time()
            raw = self._copy_payload(buffer)
            frame_id = int(buffer.frame_id)
            timestamp_ns = int(buffer.timestamp_ns)
            if native_ply is not None:
                # The vendor PLY is only a cross-check, so it must not cost the frame.
                try:
                    self._write_native_ply(buffer, Path(native_ply))
                except Exception as exc:
                    log.warning("Arena PLY export failed: %s", exc)
                    native_ply_error = str(exc)
        finally:
            # Arena owns the buffer memory; everything needed was copied above.
            self._release(buffer)
        frame = frame_from_raw(raw, self._transform, frame_id=frame_id,
                               timestamp_ns=timestamp_ns, host_time=host_time,
                               z_min_m=self.settings.z_min_m, z_max_m=self.settings.z_max_m)
        if native_ply_error is not None:
            frame.extra["native_ply_error"] = native_ply_error
        return frame

    @staticmethod
    def _copy_payload(buffer) -> np.ndarray:
        """Validate an Arena buffer and copy it into a read-only (H, W, 4) array."""
        if buffer.is_incomplete:
            raise IncompleteFrameError(
                "Frame arrived incomplete (lost packets). Check the network link "
                "and the StreamMissedPacketCount statistic.")
        pixel_format = buffer.pixel_format
        if getattr(pixel_format, "name", str(pixel_format)) != PIXEL_FORMAT:
            raise IncompleteFrameError(f"Unexpected pixel format {pixel_format!r}")
        width, height = int(buffer.width), int(buffer.height)
        if int(buffer.bits_per_pixel) != _BYTES_PER_PIXEL * 8:
            raise IncompleteFrameError(f"Unexpected {buffer.bits_per_pixel} bits per pixel")
        if int(buffer.padding_x) or int(buffer.padding_y):
            raise IncompleteFrameError("Padded image payloads are not supported")
        size = width * height * _BYTES_PER_PIXEL
        if int(buffer.size_filled) < size:
            raise IncompleteFrameError(
                f"Payload holds {buffer.size_filled} bytes, expected {size}")
        byte_order = ">" if getattr(buffer.pixel_endianness, "name", "") == "BIG" else "<"
        data = ctypes.string_at(buffer.pdata, size)
        return np.frombuffer(data, dtype=f"{byte_order}u2").reshape(height, width, 4)

    def _flush(self) -> None:
        for _ in range(self._num_buffers):
            try:
                buffer = self._device.get_buffer(timeout=0)
            except TimeoutError:
                return
            self._release(buffer)

    def _wait_until_armed(self) -> None:
        armed = self._node("TriggerArmed")
        deadline = time.monotonic() + self.armed_timeout_s
        while not bool(armed.value):
            if time.monotonic() >= deadline:
                raise FrameTimeoutError("Camera did not become trigger-ready")
            time.sleep(0.005)

    def _write_native_ply(self, buffer, path: Path) -> None:
        """Have LUCID's Save library write the frame, as a cross-check of our decode.

        The file is in millimetres and holds only valid points, with no intensity.
        """
        from arena_api.__future__.save import Writer

        scale_x, scale_y, scale_z = self._transform.scale_mm
        if not (math.isclose(scale_x, scale_y) and math.isclose(scale_x, scale_z)):
            raise CameraError("The Arena PLY writer needs the same scale on all three axes")
        offset_x, offset_y, offset_z = self._transform.offset_mm
        path.parent.mkdir(parents=True, exist_ok=True)
        Writer().save(buffer, str(path), scale=scale_x, offset_a=offset_x,
                      offset_b=offset_y, offset_c=offset_z)

    # settings ----------------------------------------------------------------

    def controls(self) -> list[ControlSpec]:
        if self._device is not None and not self._streaming:
            self._controls = self._read_controls()
        return list(self._controls)

    def set_control(self, name: str, value: Any) -> None:
        self._require_open()
        # Most features lock while streaming, and the coordinate scale can change
        # with the operating mode, so always reconfigure with the stream stopped.
        was_streaming, was_triggered = self._streaming, self._triggered
        self.stop()
        try:
            self._write(name, value)
            self._transform = self._read_transform()
            self._controls = self._read_controls()
        finally:
            if was_streaming:
                self.start(was_triggered)

    def read_settings(self) -> dict[str, Any]:
        self._require_open()
        names = ("PixelFormat", "Scan3dDistanceUnit", *TUNABLE_NODES,
                 *self.settings.extra_nodes)
        settings = {}
        for name in names:
            node = self._find_node(name)
            if node is not None and node.is_readable:
                settings[name] = node.value
        return settings

    def export_node_values(self, path: Path) -> None:
        """Write every streamable camera feature to a text file Arena can reload."""
        self._require_open()
        self._device.nodemap.write_streamable_node_values_to(str(path))

    # configuration helpers ---------------------------------------------------

    def _select(self, device_infos: list[dict]) -> dict:
        if not device_infos:
            raise CameraNotFoundError(
                "No camera found. Check PoE power and cabling, and that the camera "
                "and the network adapter are on the same subnet.")
        found = ", ".join(f"{d.get('model')} {d.get('serial')}" for d in device_infos)
        if self._serial is None:
            if len(device_infos) > 1:
                raise CameraError(f"Several cameras found ({found}); choose one by serial")
            return device_infos[0]
        for device_info in device_infos:
            if str(device_info.get("serial")) == self._serial:
                return device_info
        raise CameraNotFoundError(f"Camera {self._serial} not found. Discovered: {found}")

    def _configure(self) -> None:
        if self._find_node("Scan3dCoordinateScale") is None:
            raise CameraError(f"{self.info.model} does not provide 3D coordinate output")
        stream_nodemap = self._device.tl_stream_nodemap
        self._write("StreamAutoNegotiatePacketSize", True, stream_nodemap)
        self._write("StreamPacketResendEnable", True, stream_nodemap)
        self._write("PixelFormat", PIXEL_FORMAT)
        self._write("AcquisitionMode", "Continuous", required=False)
        for name, value in self.settings.node_values().items():
            self._write(name, value)
        self._transform = self._read_transform()
        self._controls = self._read_controls()

    def _set_trigger_mode(self, triggered: bool) -> None:
        self._write("TriggerSelector", "FrameStart", required=triggered)
        self._write("TriggerMode", "On" if triggered else "Off", required=triggered)
        if triggered:
            self._write("TriggerSource", "Software")
        # Triggered captures must arrive in order; a live preview only wants the
        # most recent frame.
        self._write("StreamBufferHandlingMode", "OldestFirst" if triggered else "NewestOnly",
                    self._device.tl_stream_nodemap)

    def _read_transform(self) -> CoordinateTransform:
        """Read each axis's scale and offset. Must follow any format or mode change."""
        unit = self._find_node("Scan3dDistanceUnit")
        if unit is not None and unit.is_readable and unit.value != "Millimeter":
            raise CameraError(f"Expected coordinates in millimetres, camera reports {unit.value}")
        selector = self._node("Scan3dCoordinateSelector")
        scales, offsets = [], []
        for axis in _AXES:
            selector.value = axis
            scales.append(float(self._node("Scan3dCoordinateScale").value))
            offsets.append(float(self._node("Scan3dCoordinateOffset").value))
        return CoordinateTransform(tuple(scales), tuple(offsets))

    def _read_controls(self) -> list[ControlSpec]:
        specs = []
        for name in dict.fromkeys((*TUNABLE_NODES, *self.settings.extra_nodes)):
            node = self._find_node(name)
            kind = _NODE_KINDS.get(type(node).__name__)
            if node is None or kind is None or not node.is_readable:
                continue
            numeric = kind in ("int", "float")
            specs.append(ControlSpec(
                name=name, kind=kind, value=node.value,
                options=tuple(node.enumentry_names) if kind == "enum" else (),
                minimum=node.min if numeric else None,
                maximum=node.max if numeric else None,
                writable=bool(node.is_writable)))
        return specs

    def _find_node(self, name: str, nodemap=None):
        nodemap = self._device.nodemap if nodemap is None else nodemap
        try:
            return nodemap[name]
        except KeyError:
            return None

    def _node(self, name: str, nodemap=None):
        node = self._find_node(name, nodemap)
        if node is None:
            raise CameraError(f"This camera has no '{name}' feature")
        return node

    def _write(self, name: str, value: Any, nodemap=None, required: bool = True) -> None:
        node = self._find_node(name, nodemap)
        if node is None:
            if required:
                raise CameraError(f"This camera has no '{name}' feature")
            return
        kind = _NODE_KINDS.get(type(node).__name__)
        if kind == "enum" and str(value) not in node.enumentry_names:
            raise CameraError(f"{name} has no option {value!r}. "
                              f"Available: {', '.join(node.enumentry_names)}")
        try:
            node.value = _COERCE[kind](value) if kind in _COERCE else value
        except Exception as exc:
            raise CameraError(
                f"Could not set {name} = {value!r}: {exc}. If ArenaView or another "
                "program is connected to this camera, close it and retry.") from exc

    def _require_open(self) -> None:
        if self._device is None:
            raise CameraError("The camera is not open")

    def _require_streaming(self) -> None:
        self._require_open()
        if not self._streaming:
            raise CameraError("The camera is not streaming; call start() first")
