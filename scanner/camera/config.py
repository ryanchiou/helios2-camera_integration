"""Camera configuration applied when a camera is opened."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Features offered as live controls and recorded with every capture, grouped the
# way the settings panel shows them. Each one is used only if the connected
# camera actually exposes it. The order is also the write order: the operating
# and HDR modes decide which exposure and accumulation values are allowed, and
# each filter is switched on before its threshold is set.
SETTING_GROUPS = (
    ("Mode and exposure", (
        "Scan3dOperatingMode",
        "Scan3dHDRMode",
        "ExposureTimeSelector",
        "ConversionGain",
    )),
    ("Noise filters", (
        "Scan3dImageAccumulation",
        "Scan3dSpatialFilterEnable",
        "Scan3dFlyingPixelsRemovalEnable",
        "Scan3dFlyingPixelsDistanceThreshold",
        "Scan3dConfidenceThresholdEnable",
        "Scan3dConfidenceThresholdMin",
    )),
    ("Distance limits (removes points)", (
        "Scan3dDistanceMin",
        "Scan3dDistanceMax",
    )),
)
TUNABLE_NODES = tuple(name for _, names in SETTING_GROUPS for name in names)

# CameraSettings field behind each tunable node.
_FIELDS = {
    "Scan3dOperatingMode": "operating_mode",
    "Scan3dHDRMode": "hdr_mode",
    "ExposureTimeSelector": "exposure_time",
    "ConversionGain": "conversion_gain",
    "Scan3dImageAccumulation": "image_accumulation",
    "Scan3dSpatialFilterEnable": "spatial_filter",
    "Scan3dFlyingPixelsRemovalEnable": "flying_pixels_removal",
    "Scan3dFlyingPixelsDistanceThreshold": "flying_pixels_threshold_mm",
    "Scan3dConfidenceThresholdEnable": "confidence_threshold",
    "Scan3dConfidenceThresholdMin": "confidence_threshold_min",
    "Scan3dDistanceMin": "distance_min_mm",
    "Scan3dDistanceMax": "distance_max_mm",
}


@dataclass
class CameraSettings:
    """Settings written to the camera before streaming.

    A value of ``None`` leaves the camera's current setting untouched. The valid
    names for the enum settings depend on the camera model and firmware, so read
    them from ``camera.controls()`` rather than assuming them.
    """

    operating_mode: str | None = None  # Scan3dOperatingMode
    hdr_mode: str | None = None  # Scan3dHDRMode; takes over from accumulation when on
    exposure_time: str | None = None  # ExposureTimeSelector
    conversion_gain: str | None = None  # ConversionGain
    image_accumulation: int | None = None  # Scan3dImageAccumulation
    spatial_filter: bool | None = None  # Scan3dSpatialFilterEnable
    flying_pixels_removal: bool | None = None  # Scan3dFlyingPixelsRemovalEnable
    flying_pixels_threshold_mm: int | None = None  # Scan3dFlyingPixelsDistanceThreshold
    confidence_threshold: bool | None = None  # Scan3dConfidenceThresholdEnable
    confidence_threshold_min: int | None = None  # Scan3dConfidenceThresholdMin
    # Radial distance limits applied in the camera (mm). A non-zero minimum also
    # moves the distance at which the operating mode wraps around.
    distance_min_mm: int | None = None  # Scan3dDistanceMin
    distance_max_mm: int | None = None  # Scan3dDistanceMax
    extra_nodes: dict[str, Any] = field(default_factory=dict)

    # Working volume along the optical axis; points outside it are marked invalid.
    z_min_m: float = 0.0
    z_max_m: float = float("inf")

    def node_values(self) -> dict[str, Any]:
        """Node name to value, in the order the nodes should be written."""
        values = {name: getattr(self, _FIELDS[name]) for name in TUNABLE_NODES}
        values = {name: value for name, value in values.items() if value is not None}
        values.update(self.extra_nodes)
        return values
