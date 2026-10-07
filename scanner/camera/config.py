"""Camera configuration applied when a camera is opened."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Features offered as live controls and recorded with every capture. Each one is
# used only if the connected camera actually exposes it.
TUNABLE_NODES = (
    "Scan3dOperatingMode",
    "ExposureTimeSelector",
    "ConversionGain",
    "Scan3dImageAccumulation",
    "Scan3dSpatialFilterEnable",
    "Scan3dConfidenceThresholdEnable",
)


@dataclass
class CameraSettings:
    """Settings written to the camera before streaming.

    A value of ``None`` leaves the camera's current setting untouched. The valid
    names for the enum settings depend on the camera model and firmware, so read
    them from ``camera.controls()`` rather than assuming them.
    """

    operating_mode: str | None = None  # Scan3dOperatingMode
    exposure_time: str | None = None  # ExposureTimeSelector
    conversion_gain: str | None = None  # ConversionGain
    image_accumulation: int | None = None  # Scan3dImageAccumulation
    spatial_filter: bool | None = None  # Scan3dSpatialFilterEnable
    confidence_threshold: bool | None = None  # Scan3dConfidenceThresholdEnable
    extra_nodes: dict[str, Any] = field(default_factory=dict)

    # Working volume along the optical axis; points outside it are marked invalid.
    z_min_m: float = 0.0
    z_max_m: float = float("inf")

    def node_values(self) -> dict[str, Any]:
        """Node name to value, in the order the nodes should be written."""
        named = {
            "Scan3dOperatingMode": self.operating_mode,
            "ExposureTimeSelector": self.exposure_time,
            "ConversionGain": self.conversion_gain,
            "Scan3dImageAccumulation": self.image_accumulation,
            "Scan3dSpatialFilterEnable": self.spatial_filter,
            "Scan3dConfidenceThresholdEnable": self.confidence_threshold,
        }
        values = {name: value for name, value in named.items() if value is not None}
        values.update(self.extra_nodes)
        return values
