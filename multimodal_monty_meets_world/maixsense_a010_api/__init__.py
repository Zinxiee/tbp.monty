"""Maixsense A010 Python API.

This package provides:
- USB serial control + streaming frame parsing
- HTTP frame/config access compatible with the vendor sample scripts
"""

from .models import (
    A010UsbFrame,
    DisplayMode,
    FrameConfig,
    HttpFrame,
    LensCoefficients,
)
from .monty_adapter import CameraIntrinsics, MaixsenseMontyObservationAdapter
from .usb_client import MaixsenseA010USB
from .http_client import MaixsenseA010HTTP

__all__ = [
    "A010UsbFrame",
    "DisplayMode",
    "FrameConfig",
    "HttpFrame",
    "LensCoefficients",
    "CameraIntrinsics",
    "MaixsenseMontyObservationAdapter",
    "MaixsenseA010USB",
    "MaixsenseA010HTTP",
]
