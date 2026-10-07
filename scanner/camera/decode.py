"""Conversion of raw Coord3D_ABCY16 payloads into metric coordinates.

This module is pure NumPy so saved captures can be decoded on a machine that
has no Arena SDK installed.
"""

from __future__ import annotations

import numpy as np

from .types import INVALID_RAW_Z, CoordinateTransform, Frame


def decode_abcy16(raw_abcy, scales_mm, offsets_mm, z_min_m=0.0, z_max_m=np.inf):
    """Decode an (H, W, 4) unsigned 16-bit X/Y/Z/intensity array.

    Returns ``(xyz_m, intensity, valid)``. Invalid pixels stay in the organised
    arrays so pixel correspondence is preserved; use ``valid`` to select points.
    """
    raw_abcy = np.asarray(raw_abcy)
    if (raw_abcy.ndim != 3 or raw_abcy.shape[-1] != 4
            or raw_abcy.dtype.kind != "u"
            or raw_abcy.dtype.itemsize != 2):
        raise ValueError("Expected H x W x 4 unsigned 16-bit data")

    scales = np.asarray(scales_mm, dtype=np.float64).reshape(3)
    offsets = np.asarray(offsets_mm, dtype=np.float64).reshape(3)
    if not (np.isfinite(scales).all() and np.isfinite(offsets).all()):
        raise ValueError("Coordinate scales and offsets must be finite")

    # metres = (raw * scale_mm + offset_mm) / 1000, applied in place.
    xyz_m = raw_abcy[..., :3].astype(np.float64)
    xyz_m *= scales / 1000.0
    xyz_m += offsets / 1000.0

    z = xyz_m[..., 2]
    valid = (raw_abcy[..., 2] != INVALID_RAW_Z) & (z > z_min_m) & (z <= z_max_m)
    return xyz_m, raw_abcy[..., 3].copy(), valid


def frame_from_raw(raw_abcy, transform: CoordinateTransform, *, frame_id=0,
                   timestamp_ns=0, host_time=0.0, z_min_m=0.0, z_max_m=np.inf) -> Frame:
    """Build a :class:`Frame` from an application-owned raw array."""
    xyz_m, intensity, valid = decode_abcy16(
        raw_abcy, transform.scale_mm, transform.offset_mm, z_min_m, z_max_m)
    return Frame(raw=raw_abcy, transform=transform, xyz_m=xyz_m, intensity=intensity,
                 valid=valid, frame_id=int(frame_id), timestamp_ns=int(timestamp_ns),
                 host_time=float(host_time))
