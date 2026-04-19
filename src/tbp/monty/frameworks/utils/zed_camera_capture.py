# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import importlib
import logging
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ZEDRGBDCapture:
    """Thin wrapper around ZED SDK for single-frame RGBD capture."""

    _RESOLUTION_ATTRS = {
        "HD2K": "HD2K",
        "HD1200": "HD1200",
        "HD1080": "HD1080",
        "HD720": "HD720",
        "SVGA": "SVGA",
        "VGA": "VGA",
    }

    _DEPTH_MODE_ATTRS = {
        "PERFORMANCE": "PERFORMANCE",
        "QUALITY": "QUALITY",
        "ULTRA": "ULTRA",
        "NEURAL": "NEURAL",
        "NEURAL_PLUS": "NEURAL_PLUS",
    }

    _UNIT_ATTRS = {
        "METER": "METER",
        "MILLIMETER": "MILLIMETER",
    }

    def __init__(
        self,
        resolution: str = "HD720",
        fps: int = 30,
        depth_mode: str = "NEURAL",
        units: str = "METER",
    ):
        self._sl = None
        self._zed = None
        self._left = None
        self._depth = None
        self._runtime = None
        self._available = False
        self._intrinsics: dict[str, Any] | None = None

        try:
            sl = importlib.import_module("pyzed.sl")
        except Exception as exc:  # pragma: no cover - depends on local SDK install
            logger.warning("pyzed SDK is unavailable; RGBD capture disabled: %s", exc)
            return

        self._sl = sl

        try:
            zed = sl.Camera()
            init = sl.InitParameters()
            init.camera_resolution = getattr(sl.RESOLUTION, self._RESOLUTION_ATTRS[resolution])
            init.camera_fps = fps
            init.depth_mode = getattr(sl.DEPTH_MODE, self._DEPTH_MODE_ATTRS[depth_mode])
            init.coordinate_units = getattr(sl.UNIT, self._UNIT_ATTRS[units])
            init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

            open_status = zed.open(init)
            if open_status != sl.ERROR_CODE.SUCCESS:
                logger.warning("Unable to open ZED camera: %s", open_status)
                zed.close()
                return

            self._zed = zed
            self._left = sl.Mat()
            self._depth = sl.Mat()
            self._runtime = sl.RuntimeParameters()
            self._available = True
            self._intrinsics = self._load_intrinsics()
        except Exception as exc:
            logger.warning("Failed to initialize ZED camera capture: %s", exc)
            self.close()

    def _load_intrinsics(self) -> dict[str, Any] | None:
        """Pull left-camera pinhole intrinsics from the ZED SDK.

        Returns None if anything goes wrong so downstream consumers can fall back
        to a default FOV.
        """
        if self._zed is None:
            return None
        try:
            cam_info = self._zed.get_camera_information()
            # ZED SDK v4 nests calibration under camera_configuration; older SDKs
            # expose it directly on CameraInformation. Try both shapes.
            cam_config = getattr(cam_info, "camera_configuration", None)
            if cam_config is not None:
                cal_params = cam_config.calibration_parameters
                resolution = cam_config.resolution
            else:
                cal_params = cam_info.calibration_parameters
                resolution = cam_info.camera_resolution
            left_cam = cal_params.left_cam
            return {
                "fx": float(left_cam.fx),
                "fy": float(left_cam.fy),
                "cx": float(left_cam.cx),
                "cy": float(left_cam.cy),
                "width": int(resolution.width),
                "height": int(resolution.height),
                "model": "pinhole_left",
            }
        except Exception as exc:
            logger.warning("Failed to read ZED calibration: %s", exc)
            return None

    def is_available(self) -> bool:
        return self._available

    def grab_single_frame(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any] | None]:
        if not self._available or self._zed is None:
            return None, None, None

        sl = self._sl
        assert sl is not None

        status = self._zed.grab(self._runtime)
        if status != sl.ERROR_CODE.SUCCESS:
            logger.warning("ZED grab failed: %s", status)
            return None, None, None

        self._zed.retrieve_image(self._left, sl.VIEW.LEFT)
        self._zed.retrieve_measure(self._depth, sl.MEASURE.DEPTH)

        bgra_image = self._left.get_data().copy()
        depth_array = self._depth.get_data().copy().astype(np.float32)

        metadata: dict[str, Any] = {
            "capture_time": datetime.now().isoformat(),
            "bgra_shape": list(bgra_image.shape),
            "depth_shape": list(depth_array.shape),
        }
        if self._intrinsics is not None:
            metadata["intrinsics"] = dict(self._intrinsics)
        return bgra_image, depth_array, metadata

    def close(self) -> None:
        if self._sl is not None:
            if self._left is not None:
                self._left.free(self._sl.MEM.CPU)
                self._left = None
            if self._depth is not None:
                self._depth.free(self._sl.MEM.CPU)
                self._depth = None

        if self._zed is not None:
            self._zed.close()
            self._zed = None

        self._available = False

    def __enter__(self) -> "ZEDRGBDCapture":
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        self.close()
