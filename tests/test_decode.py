import numpy as np
import pytest

from scanner.camera import CoordinateTransform, decode_abcy16, frame_from_raw


def make_raw(pixels):
    """An (H, W, 4) uint16 array from a nested list of (x, y, z, intensity) counts."""
    return np.array(pixels, dtype=np.uint16)


def test_applies_each_axis_scale_and_offset():
    raw = make_raw([[(1000, 2000, 3000, 7)]])
    xyz_m, intensity, valid = decode_abcy16(raw, (0.25, 0.5, 0.125), (-100.0, 50.0, 10.0))
    # mm: 1000*0.25-100 = 150, 2000*0.5+50 = 1050, 3000*0.125+10 = 385
    assert xyz_m[0, 0] == pytest.approx([0.150, 1.050, 0.385])
    assert intensity[0, 0] == 7
    assert valid[0, 0]


def test_keeps_fractional_millimetres():
    raw = make_raw([[(1, 1, 3, 0)]])
    xyz_m, _, _ = decode_abcy16(raw, (0.25, 0.25, 0.25), (0.0, 0.0, 0.0))
    assert xyz_m[0, 0] == pytest.approx([0.00025, 0.00025, 0.00075])


def test_marks_sentinel_and_zero_depth_invalid():
    raw = make_raw([[(10, 10, 65535, 1), (10, 10, 0, 1), (10, 10, 4000, 1)]])
    _, _, valid = decode_abcy16(raw, (0.25,) * 3, (0.0,) * 3)
    assert valid.tolist() == [[False, False, True]]


def test_depth_window_limits_valid_points():
    raw = make_raw([[(0, 0, 2000, 0), (0, 0, 4000, 0), (0, 0, 8000, 0)]])  # 0.5, 1.0, 2.0 m
    _, _, valid = decode_abcy16(raw, (0.25,) * 3, (0.0,) * 3, z_min_m=0.6, z_max_m=1.5)
    assert valid.tolist() == [[False, True, False]]


def test_all_invalid_frame_yields_no_points():
    raw = np.full((4, 5, 4), 65535, dtype=np.uint16)
    frame = frame_from_raw(raw, CoordinateTransform((0.25,) * 3, (0.0,) * 3))
    assert frame.valid_count == 0
    assert frame.points_m().shape == (0, 3)


def test_intensity_is_a_copy_and_keeps_full_range():
    raw = make_raw([[(0, 0, 100, 65535)]])
    _, intensity, _ = decode_abcy16(raw, (0.25,) * 3, (0.0,) * 3)
    assert intensity.dtype == np.uint16 and intensity[0, 0] == 65535
    assert not np.shares_memory(intensity, raw)


@pytest.mark.parametrize("bad", [
    np.zeros((4, 5, 3), dtype=np.uint16),  # wrong channel count
    np.zeros((4, 5, 4), dtype=np.int16),  # signed
    np.zeros((4, 5, 4), dtype=np.uint8),  # wrong width
    np.zeros((20, 4), dtype=np.uint16),  # not organised
])
def test_rejects_unexpected_layouts(bad):
    with pytest.raises(ValueError):
        decode_abcy16(bad, (0.25,) * 3, (0.0,) * 3)


def test_rejects_non_finite_calibration():
    raw = np.zeros((2, 2, 4), dtype=np.uint16)
    with pytest.raises(ValueError):
        decode_abcy16(raw, (0.25, np.nan, 0.25), (0.0,) * 3)
