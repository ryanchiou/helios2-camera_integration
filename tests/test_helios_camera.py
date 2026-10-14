"""HeliosCamera logic, exercised against a stand-in for arena_api.

No camera or Arena SDK is involved: these tests check that the wrapper drives
the API in the right order and handles its failure modes, not that the node
names and values match a real Helios2+.
"""

import ctypes
import sys
import types

import numpy as np
import pytest

from scanner.camera import (CameraError, CameraNotFoundError, CameraSettings,
                            FrameTimeoutError, HeliosCamera, IncompleteFrameError,
                            decode_abcy16)

WIDTH, HEIGHT = 8, 6
SCALES = (0.25, 0.5, 0.125)
OFFSETS = (-100.0, -200.0, 5.0)


class _Node:
    is_readable = True
    is_writable = True

    def __init__(self, name, value, log):
        self.name, self._value, self._log = name, value, log

    def _check(self, value):
        pass

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value):
        self._check(value)
        self._value = value
        self._log.append((self.name, value))


# The wrapper identifies node types by these class names, as arena_api defines them.
class NodeEnumeration(_Node):
    def __init__(self, name, value, log, options):
        super().__init__(name, value, log)
        self.enumentry_names = list(options)

    def _check(self, value):
        if not isinstance(value, str):
            raise TypeError("str expected")


class NodeBoolean(_Node):
    def _check(self, value):
        if not isinstance(value, bool):
            raise TypeError("bool expected")


class NodeInteger(_Node):
    min, max = 1, 8

    def _check(self, value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("int expected")


class _PerAxisFloat:
    """Scale or offset node: its value follows Scan3dCoordinateSelector."""

    is_readable = True

    def __init__(self, selector, values):
        self._selector, self._values = selector, values

    @property
    def value(self):
        return self._values[("CoordinateA", "CoordinateB", "CoordinateC").index(self._selector.value)]


class _Nodemap(dict):
    def __getitem__(self, name):
        if name not in self:
            raise KeyError(f"'{name}' node does not exist")
        return super().__getitem__(name)


class _Buffer:
    bits_per_pixel = 64
    padding_x = padding_y = 0
    pixel_format = types.SimpleNamespace(name="Coord3D_ABCY16")
    pixel_endianness = types.SimpleNamespace(name="LITTLE")

    def __init__(self, frame_id, raw, incomplete=False):
        self.frame_id = frame_id
        self.timestamp_ns = 1000 * frame_id
        self.width, self.height = raw.shape[1], raw.shape[0]
        self.is_incomplete = incomplete
        self.size_filled = raw.nbytes
        self.memory = (ctypes.c_ubyte * raw.nbytes).from_buffer_copy(raw.tobytes())
        self.pdata = ctypes.cast(self.memory, ctypes.POINTER(ctypes.c_ubyte))


def make_raw(seed):
    raw = np.random.default_rng(seed).integers(1, 60000, (HEIGHT, WIDTH, 4)).astype(np.uint16)
    raw[0, 0, 2] = 65535  # one invalid pixel
    return raw


class _Device:
    def __init__(self):
        self.writes = []
        self.queue = []
        self.requeued = []
        self.streaming = False
        self.start_count = 0
        self.next_frame_id = 1
        self.armed = True
        log = self.writes
        selector = NodeEnumeration("Scan3dCoordinateSelector", "CoordinateA", log,
                                   ["CoordinateA", "CoordinateB", "CoordinateC"])
        self.nodemap = _Nodemap({
            "PixelFormat": NodeEnumeration("PixelFormat", "Mono8", log,
                                           ["Mono8", "Coord3D_ABCY16"]),
            "AcquisitionMode": NodeEnumeration("AcquisitionMode", "SingleFrame", log,
                                               ["SingleFrame", "Continuous"]),
            "Scan3dOperatingMode": NodeEnumeration("Scan3dOperatingMode", "ModeA", log,
                                                   ["ModeA", "ModeB"]),
            "Scan3dImageAccumulation": NodeInteger("Scan3dImageAccumulation", 1, log),
            "Scan3dSpatialFilterEnable": NodeBoolean("Scan3dSpatialFilterEnable", False, log),
            "Scan3dCoordinateSelector": selector,
            "Scan3dCoordinateScale": _PerAxisFloat(selector, SCALES),
            "Scan3dCoordinateOffset": _PerAxisFloat(selector, OFFSETS),
            "TriggerSelector": NodeEnumeration("TriggerSelector", "AcquisitionStart", log,
                                               ["AcquisitionStart", "FrameStart"]),
            "TriggerMode": NodeEnumeration("TriggerMode", "Off", log, ["Off", "On"]),
            "TriggerSource": NodeEnumeration("TriggerSource", "Line0", log,
                                             ["Line0", "Software"]),
            # TriggerArmed is read through ``.value``; the device doubles as that node.
            "TriggerArmed": self,
            "TriggerSoftware": types.SimpleNamespace(execute=self._software_trigger),
        })
        self.tl_stream_nodemap = _Nodemap({
            "StreamAutoNegotiatePacketSize": NodeBoolean("StreamAutoNegotiatePacketSize",
                                                         False, log),
            "StreamPacketResendEnable": NodeBoolean("StreamPacketResendEnable", False, log),
            "StreamBufferHandlingMode": NodeEnumeration(
                "StreamBufferHandlingMode", "OldestFirst", log, ["OldestFirst", "NewestOnly"]),
        })

    @property
    def value(self):
        return self.armed

    def _new_buffer(self, incomplete=False):
        buffer = _Buffer(self.next_frame_id, make_raw(self.next_frame_id), incomplete)
        self.next_frame_id += 1
        return buffer

    def _software_trigger(self):
        self.queue.append(self._new_buffer())

    def start_stream(self, number_of_buffers):
        self.streaming = True
        self.start_count += 1

    def stop_stream(self):
        self.streaming = False

    def get_buffer(self, timeout=None):
        assert self.streaming
        if self.queue:
            return self.queue.pop(0)
        if self.nodemap["TriggerMode"].value == "On" or timeout == 0:
            raise TimeoutError
        return self._new_buffer()

    def requeue_buffer(self, buffer):
        self.requeued.append(buffer)


class _System:
    def __init__(self, devices):
        self.devices = devices
        self.destroyed = []

    @property
    def device_infos(self):
        return [{"model": "HTP003S-001", "serial": serial, "ip": "169.254.0.2",
                 "mac": "aa:bb", "version": "1.2.3"} for serial in self.devices]

    def create_device(self, device_info):
        return [self.devices[device_info["serial"]]]

    def destroy_device(self, device):
        self.destroyed.append(device)


class _Writer:
    calls = []
    fail = False

    def save(self, buffer, pattern, **kwargs):
        if self.fail:
            raise RuntimeError("disk full")
        self.calls.append((buffer, pattern, kwargs))


@pytest.fixture
def arena(monkeypatch):
    """Install the stand-in arena_api and return its system object."""
    system = _System({"111": _Device()})
    _Writer.calls, _Writer.fail = [], False
    modules = {
        "arena_api": types.ModuleType("arena_api"),
        "arena_api.system": types.ModuleType("arena_api.system"),
        "arena_api.__future__": types.ModuleType("arena_api.__future__"),
        "arena_api.__future__.save": types.ModuleType("arena_api.__future__.save"),
    }
    modules["arena_api.system"].system = system
    modules["arena_api.__future__.save"].Writer = _Writer
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    return system


@pytest.fixture
def device(arena):
    return arena.devices["111"]


def expected_decode(frame_id):
    return decode_abcy16(make_raw(frame_id), SCALES, OFFSETS)


def test_open_configures_stream_format_and_settings_in_order(arena, device):
    settings = CameraSettings(operating_mode="ModeB", image_accumulation=4, spatial_filter=True)
    with HeliosCamera(settings=settings) as camera:
        assert camera.info.serial == "111" and camera.info.firmware == "1.2.3"
    names = [name for name, _ in device.writes]
    assert ("PixelFormat", "Coord3D_ABCY16") in device.writes
    assert ("AcquisitionMode", "Continuous") in device.writes
    assert ("StreamAutoNegotiatePacketSize", True) in device.writes
    assert ("StreamPacketResendEnable", True) in device.writes
    assert ("Scan3dImageAccumulation", 4) in device.writes
    assert (names.index("PixelFormat") < names.index("Scan3dOperatingMode")
            < names.index("Scan3dImageAccumulation") < names.index("Scan3dSpatialFilterEnable"))
    # Scales and offsets are read only after everything that can change them.
    assert names.index("Scan3dSpatialFilterEnable") < names.index("Scan3dCoordinateSelector")
    assert arena.destroyed == [device]


def test_modes_are_written_before_exposure_and_filters_before_thresholds(arena, device):
    log = device.writes
    device.nodemap.update({
        "Scan3dHDRMode": NodeEnumeration("Scan3dHDRMode", "Off", log, ["Off", "StandardHDR"]),
        "Scan3dFlyingPixelsRemovalEnable": NodeBoolean("Scan3dFlyingPixelsRemovalEnable",
                                                       False, log),
        "Scan3dFlyingPixelsDistanceThreshold": NodeInteger("Scan3dFlyingPixelsDistanceThreshold",
                                                           1, log),
        "Scan3dDistanceMax": NodeInteger("Scan3dDistanceMax", 8, log),
    })
    settings = CameraSettings(distance_max_mm=6, flying_pixels_threshold_mm=5,
                              flying_pixels_removal=True, hdr_mode="StandardHDR",
                              operating_mode="ModeB")
    with HeliosCamera(settings=settings) as camera:
        assert camera.read_settings()["Scan3dFlyingPixelsDistanceThreshold"] == 5
        assert {spec.name for spec in camera.controls()} >= {
            "Scan3dHDRMode", "Scan3dFlyingPixelsRemovalEnable",
            "Scan3dFlyingPixelsDistanceThreshold", "Scan3dDistanceMax"}
    names = [name for name, _ in log]
    assert (names.index("Scan3dOperatingMode") < names.index("Scan3dHDRMode")
            < names.index("Scan3dFlyingPixelsRemovalEnable")
            < names.index("Scan3dFlyingPixelsDistanceThreshold")
            < names.index("Scan3dDistanceMax"))


def test_grab_decodes_with_per_axis_calibration_and_requeues(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        frame = camera.grab()
    assert ("TriggerMode", "Off") in device.writes
    assert ("StreamBufferHandlingMode", "NewestOnly") in device.writes
    assert frame.transform.scale_mm == SCALES and frame.transform.offset_mm == OFFSETS
    xyz_m, intensity, valid = expected_decode(1)
    assert np.array_equal(frame.xyz_m, xyz_m)
    assert np.array_equal(frame.intensity, intensity)
    assert np.array_equal(frame.valid, valid) and not frame.valid[0, 0]
    assert (frame.frame_id, frame.timestamp_ns) == (1, 1000)
    assert len(device.requeued) == 1


def test_frame_owns_its_data_after_the_buffer_is_reused(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        frame = camera.grab()
        ctypes.memset(device.requeued[0].memory, 0, len(device.requeued[0].memory))
        assert np.array_equal(frame.raw, make_raw(1))


def test_discard_drops_frames_and_requeues_them(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        frame = camera.grab(discard=2)
    assert frame.frame_id == 3
    assert len(device.requeued) == 3


def test_incomplete_frame_is_rejected_but_still_requeued(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        device.queue.append(device._new_buffer(incomplete=True))
        with pytest.raises(IncompleteFrameError):
            camera.grab()
        assert len(device.requeued) == 1


def test_unexpected_pixel_format_is_rejected(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        buffer = device._new_buffer()
        buffer.pixel_format = types.SimpleNamespace(name="Coord3D_ABC16")
        device.queue.append(buffer)
        with pytest.raises(IncompleteFrameError):
            camera.grab()


def test_trigger_configures_software_trigger_and_returns_the_new_frame(arena, device):
    with HeliosCamera() as camera:
        camera.start(triggered=True)
        assert ("TriggerSelector", "FrameStart") in device.writes
        assert ("TriggerMode", "On") in device.writes
        assert ("TriggerSource", "Software") in device.writes
        assert ("StreamBufferHandlingMode", "OldestFirst") in device.writes
        stale = device._new_buffer()
        device.queue.append(stale)  # left over from an earlier failed trigger
        frame = camera.trigger()
    assert frame.frame_id == 2
    assert device.requeued[0] is stale


def test_trigger_times_out_when_no_frame_follows(arena, device):
    with HeliosCamera() as camera:
        camera.start(triggered=True)
        device.nodemap["TriggerSoftware"] = types.SimpleNamespace(execute=lambda: None)
        with pytest.raises(FrameTimeoutError):
            camera.trigger(timeout_s=0.01)


def test_trigger_gives_up_if_the_camera_never_arms(arena, device):
    device.armed = False
    with HeliosCamera(armed_timeout_s=0.05) as camera:
        camera.start(triggered=True)
        with pytest.raises(FrameTimeoutError):
            camera.trigger()
    assert not device.queue  # no trigger was sent


def test_grab_and_trigger_are_tied_to_the_stream_mode(arena):
    with HeliosCamera() as camera:
        with pytest.raises(CameraError):
            camera.grab()
        camera.start()
        with pytest.raises(CameraError):
            camera.trigger()
        camera.start(triggered=True)
        with pytest.raises(CameraError):
            camera.grab()


def test_unknown_enum_option_lists_the_valid_ones(arena):
    camera = HeliosCamera(settings=CameraSettings(operating_mode="Nope"))
    with pytest.raises(CameraError, match="ModeA, ModeB"):
        camera.open()
    assert not camera.is_open


def test_controls_describe_only_features_the_camera_has(arena):
    with HeliosCamera() as camera:
        controls = {spec.name: spec for spec in camera.controls()}
    assert set(controls) == {"Scan3dOperatingMode", "Scan3dImageAccumulation",
                             "Scan3dSpatialFilterEnable"}
    assert controls["Scan3dOperatingMode"].options == ("ModeA", "ModeB")
    assert (controls["Scan3dImageAccumulation"].minimum,
            controls["Scan3dImageAccumulation"].maximum) == (1, 8)


def test_set_control_restarts_the_stream_and_coerces_types(arena, device):
    with HeliosCamera() as camera:
        camera.start()
        camera.set_control("Scan3dImageAccumulation", 4.0)  # e.g. from a float spin box
        assert camera.is_streaming and device.streaming
        assert device.start_count == 2
        assert device.nodemap["Scan3dImageAccumulation"].value == 4
        assert camera.read_settings()["Scan3dImageAccumulation"] == 4


def test_camera_selection_by_serial(arena):
    arena.devices["222"] = _Device()
    with pytest.raises(CameraError, match="choose one by serial"):
        HeliosCamera().open()
    with pytest.raises(CameraNotFoundError):
        HeliosCamera(serial="999").open()
    with HeliosCamera(serial=222) as camera:
        assert camera.info.serial == "222"


def test_no_camera_found(arena):
    arena.devices.clear()
    with pytest.raises(CameraNotFoundError):
        HeliosCamera().open()


def test_camera_without_3d_output_is_rejected(arena, device):
    del device.nodemap["Scan3dCoordinateScale"]
    with pytest.raises(CameraError, match="3D"):
        HeliosCamera().open()
    assert arena.destroyed == [device]


def test_native_ply_needs_one_scale_and_never_costs_the_frame(arena, device, tmp_path):
    with HeliosCamera() as camera:
        camera.start()
        frame = camera.grab(native_ply=tmp_path / "a.ply")
    assert "same scale" in frame.extra["native_ply_error"]
    assert not _Writer.calls and len(device.requeued) == 1


def test_native_ply_passes_the_camera_calibration_to_arena(arena, device, tmp_path):
    device.nodemap["Scan3dCoordinateScale"]._values = (0.25, 0.25, 0.25)
    with HeliosCamera() as camera:
        camera.start()
        frame = camera.grab(native_ply=tmp_path / "sub" / "a.ply")
        assert "native_ply_error" not in frame.extra
        _Writer.fail = True
        failed = camera.grab(native_ply=tmp_path / "b.ply")
    buffer, pattern, kwargs = _Writer.calls[0]
    assert pattern == str(tmp_path / "sub" / "a.ply")
    assert kwargs == {"scale": 0.25, "offset_a": -100.0, "offset_b": -200.0, "offset_c": 5.0}
    assert failed.extra["native_ply_error"] == "disk full" and failed.frame_id == 2
