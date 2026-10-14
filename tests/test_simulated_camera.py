import numpy as np
import pytest

from scanner.camera import CameraError, CameraSettings, FrameTimeoutError, SimulatedCamera
from scanner.camera.simulated import TABLE_Z_M


@pytest.fixture
def camera():
    with SimulatedCamera(fps=500, seed=0) as camera:
        yield camera


def test_frames_match_the_real_payload_layout(camera):
    camera.start()
    frame = camera.grab()
    assert frame.raw.shape == (480, 640, 4) and frame.raw.dtype == np.uint16
    assert frame.xyz_m.shape == (480, 640, 3)
    assert frame.intensity.shape == frame.valid.shape == (480, 640)
    assert not frame.raw.flags.writeable


def test_scene_geometry_is_plausible(camera):
    camera.start()
    frame = camera.grab()
    assert 0.9 < frame.valid_fraction < 1.0
    depths = frame.points_m()[:, 2]
    assert np.median(depths) == pytest.approx(TABLE_Z_M, abs=0.005)
    assert depths.min() > 0.7  # the box top is the nearest surface


def test_frame_ids_advance_and_discard_skips_frames(camera):
    camera.start()
    first = camera.grab()
    second = camera.grab()
    third = camera.grab(discard=2)
    assert second.frame_id == first.frame_id + 1
    assert third.frame_id == second.frame_id + 3


def test_triggered_mode_gives_one_frame_per_trigger(camera):
    camera.start(triggered=True)
    with pytest.raises(CameraError):
        camera.grab()
    assert camera.trigger().frame_id + 1 == camera.trigger().frame_id


def test_grab_requires_a_running_stream(camera):
    with pytest.raises(CameraError):
        camera.grab()


def test_accumulation_reduces_noise(camera):
    camera.start()

    def table_noise():
        frame = camera.grab()
        patch = (slice(400, 470), slice(20, 120))
        return np.std(frame.xyz_m[patch][frame.valid[patch]][:, 2])

    single = table_noise()
    camera.set_control("Scan3dImageAccumulation", 8)
    assert table_noise() < 0.6 * single
    assert camera.read_settings()["Scan3dImageAccumulation"] == 8


def off_table(frame):
    """Valid pixels clearly nearer than the table: the box and sphere, and flying pixels."""
    z = frame.xyz_m[..., 2]
    return frame.valid & (z < TABLE_Z_M - 0.01)


def test_flying_pixel_filter_removes_points_between_surfaces(camera):
    camera.start()
    before = off_table(camera.grab()).sum()
    camera.set_control("Scan3dFlyingPixelsRemovalEnable", True)
    after = off_table(camera.grab()).sum()
    assert 0.8 * before < after < before  # the surfaces stay, the stray points go
    assert camera.read_settings()["Scan3dFlyingPixelsDistanceThreshold"] == 20


def test_confidence_minimum_drops_weak_returns(camera):
    camera.start()
    camera.set_control("Scan3dConfidenceThresholdEnable", False)
    everything = camera.grab().valid_count
    camera.set_control("Scan3dConfidenceThresholdEnable", True)
    camera.set_control("Scan3dConfidenceThresholdMin", 1000)
    default = camera.grab().valid_count
    camera.set_control("Scan3dConfidenceThresholdMin", 9000)
    strict = camera.grab().valid_count
    assert strict < default < everything


def test_distance_limits_drop_points_outside_them(camera):
    camera.start()
    camera.set_control("Scan3dDistanceMax", 880)
    frame = camera.grab()
    distance = np.linalg.norm(frame.points_m(), axis=1)
    assert frame.valid_count > 0 and distance.max() <= 0.881


def test_hdr_locks_accumulation_and_slows_the_stream():
    with SimulatedCamera(CameraSettings(hdr_mode="LowNoiseHDRX4"), fps=500, seed=0) as camera:
        controls = {spec.name: spec for spec in camera.controls()}
        assert not controls["Scan3dImageAccumulation"].writable
        with pytest.raises(CameraError):
            camera.set_control("Scan3dImageAccumulation", 4)
        camera.start()
        camera.grab()
        with pytest.raises(FrameTimeoutError):  # 5 fps: no frame within 0.05 s
            camera.grab(timeout_s=0.05)
        camera.set_control("Scan3dHDRMode", "Off")
        camera.set_control("Scan3dImageAccumulation", 4)


def test_unknown_settings_and_options_are_rejected(camera):
    with pytest.raises(CameraError):
        camera.set_control("Scan3dNoSuchFeature", 1)
    with pytest.raises(CameraError, match="Available"):
        camera.set_control("Scan3dHDRMode", "Bright")
