import numpy as np
import pytest

from scanner.camera import CameraError, SimulatedCamera
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
