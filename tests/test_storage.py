import json

import numpy as np
import pytest

from scanner.camera import SimulatedCamera, load_capture, save_capture, write_ply


@pytest.fixture(scope="module")
def frame_and_camera():
    with SimulatedCamera(fps=500, seed=0) as camera:
        camera.start()
        yield camera.grab(), camera


def read_ply(path):
    """Minimal reader for the binary PLY files this project writes."""
    data = path.read_bytes()
    header, _, body = data.partition(b"end_header\n")
    lines = header.decode("ascii").splitlines()
    count = next(int(line.split()[-1]) for line in lines if line.startswith("element vertex"))
    names = [line.split()[-1] for line in lines if line.startswith("property")]
    types = {"x": "<f4", "y": "<f4", "z": "<f4", "intensity": "<u2"}
    return lines, np.frombuffer(body, dtype=[(name, types[name]) for name in names], count=count)


def test_capture_round_trips_exactly(frame_and_camera, tmp_path):
    frame, camera = frame_and_camera
    paths = save_capture(frame, tmp_path, "view_01", camera_info=camera.info,
                         settings=camera.read_settings())
    loaded = load_capture(paths.npz)
    assert np.array_equal(loaded.raw, frame.raw)
    assert np.array_equal(loaded.valid, frame.valid)
    assert np.array_equal(loaded.xyz_m, frame.xyz_m)
    assert loaded.transform == frame.transform
    assert (loaded.frame_id, loaded.timestamp_ns) == (frame.frame_id, frame.timestamp_ns)


def test_metadata_describes_the_capture(frame_and_camera, tmp_path):
    frame, camera = frame_and_camera
    paths = save_capture(frame, tmp_path, "view_02", camera_info=camera.info,
                         settings=camera.read_settings(), robot_pose=[0, 1, 2])
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    assert metadata["capture_id"] == "view_02"
    assert metadata["frame"]["valid_points"] == frame.valid_count
    assert metadata["coordinates"]["scale_mm"] == list(frame.transform.scale_mm)
    assert metadata["camera"]["serial"] == "SIM0001"
    assert metadata["robot_pose"] == [0, 1, 2]
    assert metadata["files"] == {"npz": "view_02.npz", "metadata": "view_02.json",
                                 "ply": "view_02.ply"}


def test_ply_holds_the_valid_points_in_metres(frame_and_camera, tmp_path):
    frame, _ = frame_and_camera
    paths = save_capture(frame, tmp_path, "view_03")
    header, vertices = read_ply(paths.ply)
    assert "comment units metres" in header
    assert len(vertices) == frame.valid_count
    points = np.column_stack([vertices["x"], vertices["y"], vertices["z"]])
    assert np.allclose(points, frame.points_m(), atol=1e-6)
    assert np.array_equal(vertices["intensity"], frame.intensity[frame.valid])


def test_ply_can_be_skipped(frame_and_camera, tmp_path):
    frame, _ = frame_and_camera
    paths = save_capture(frame, tmp_path, "view_04", ply=False)
    assert paths.ply is None and not (tmp_path / "view_04.ply").exists()


def test_ply_is_readable_by_open3d(tmp_path):
    o3d = pytest.importorskip("open3d")
    points = np.array([[0.0, 0.0, 1.0], [0.1, -0.2, 0.9], [-0.3, 0.4, 0.8]])
    path = tmp_path / "points.ply"
    write_ply(path, points, intensity=np.array([10, 20, 30], dtype=np.uint16))
    cloud = o3d.io.read_point_cloud(str(path))
    assert np.allclose(np.asarray(cloud.points), points, atol=1e-6)
