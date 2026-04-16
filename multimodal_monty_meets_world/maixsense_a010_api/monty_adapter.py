from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .models import A010UsbFrame, HttpFrame, LensCoefficients

logger = logging.getLogger(__name__)

_DEBUG_DUMP_ENV_VAR = "MONTY_ADAPTER_DEBUG_DUMP"
_DEBUG_DUMP_DIR_ENV_VAR = "MONTY_ADAPTER_DEBUG_DUMP_DIR"
_DEFAULT_DEBUG_DUMP_DIR = "~/monty_diag"


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics used for depth unprojection.

    Attributes:
        fx: Focal length in pixels on the x-axis.
        fy: Focal length in pixels on the y-axis.
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.
    """

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_lens_coefficients(cls, lens: LensCoefficients) -> "CameraIntrinsics":
        """Build intrinsics from Maixsense lens coefficients."""
        return cls(fx=lens.fx, fy=lens.fy, cx=lens.cx, cy=lens.cy)


class MaixsenseMontyObservationAdapter:
    """Convert Maixsense frames into Monty SensorObservation dictionaries.

    The output format matches the keys consumed by CameraSM/ObservationProcessor:
    - depth
    - rgba
    - semantic_3d
    - sensor_frame_data
    - world_camera

    Notes:
        - CameraSM assumes square patches internally. If incoming depth is not square,
          this adapter crops a centered square patch by default.
        - Semantics are synthesized from valid-depth pixels unless an explicit
          semantic image is provided.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        *,
        crop_center_to_square: bool = True,
        patch_height: int | None = None,
        patch_width: int | None = None,
        patch_offset_bottom_px: int | None = None,
        patch_offset_left_px: int | None = None,
        min_valid_depth_m: float = 1e-6,
        max_valid_depth_m: float | None = None,
        semantic_zero_bottom_fraction: float = 0.0,
        world_y_min_m: float | None = None,
        world_x_max_m: float | None = None,
        world_x_min_m: float | None = None,
        world_z_max_m: float | None = None,
        world_z_min_m: float | None = None,
        semantic_debug_logging: bool = False,
    ) -> None:
        self._intrinsics = intrinsics
        self._crop_center_to_square = crop_center_to_square
        roi_params = (
            patch_height,
            patch_width,
            patch_offset_bottom_px,
            patch_offset_left_px,
        )
        if any(p is not None for p in roi_params) and any(p is None for p in roi_params):
            raise ValueError(
                "patch_height, patch_width, patch_offset_bottom_px, and "
                "patch_offset_left_px must all be set together, or all be None."
            )
        self._patch_height = patch_height
        self._patch_width = patch_width
        self._patch_offset_bottom_px = patch_offset_bottom_px
        self._patch_offset_left_px = patch_offset_left_px
        self._min_valid_depth_m = min_valid_depth_m
        self._max_valid_depth_m = (
            None if max_valid_depth_m is None else float(max_valid_depth_m)
        )
        self._semantic_zero_bottom_fraction = float(
            np.clip(semantic_zero_bottom_fraction, 0.0, 0.95)
        )
        self._world_y_min_m = None if world_y_min_m is None else float(world_y_min_m)
        self._world_x_max_m = None if world_x_max_m is None else float(world_x_max_m)
        self._world_x_min_m = None if world_x_min_m is None else float(world_x_min_m)
        self._world_z_max_m = None if world_z_max_m is None else float(world_z_max_m)
        self._world_z_min_m = None if world_z_min_m is None else float(world_z_min_m)
        self._semantic_debug_logging = bool(semantic_debug_logging)

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return the pinhole intrinsics used by this adapter."""
        return self._intrinsics

    def from_usb_frame(
        self,
        frame: A010UsbFrame,
        *,
        world_camera: Optional[np.ndarray] = None,
        unit: int = 0,
        rgba: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Convert a USB frame to a Monty SensorObservation-like dictionary.

        Args:
            frame: Parsed USB frame.
            world_camera: 4x4 sensor-to-world transform. Uses identity if omitted.
            unit: Distance unit argument passed to `distance_mm_image`.
            rgba: Optional RGBA image aligned with depth.
            semantic: Optional semantic mask aligned with depth.

        Returns:
            Dictionary containing the observation keys CameraSM expects.
        """
        depth_mm = frame.distance_mm_image(unit=unit)
        return self.from_depth_mm(
            depth_mm,
            world_camera=world_camera,
            rgba=rgba,
            semantic=semantic,
        )

    def from_http_frame(
        self,
        frame: HttpFrame,
        *,
        world_camera: Optional[np.ndarray] = None,
        rgba: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
        depth_is_millimeters: bool = True,
    ) -> dict[str, np.ndarray]:
        """Convert an HTTP frame to a Monty SensorObservation-like dictionary.

        Args:
            frame: Parsed HTTP frame.
            world_camera: 4x4 sensor-to-world transform. Uses identity if omitted.
            rgba: Optional RGBA image aligned with depth.
            semantic: Optional semantic mask aligned with depth.
            depth_is_millimeters: Whether `frame.depth` is in millimeters.

        Returns:
            Dictionary containing the observation keys CameraSM expects.
        """
        if frame.depth is None:
            raise ValueError("HTTP frame does not contain a depth image")

        if depth_is_millimeters:
            depth_m = frame.depth.astype(np.float64) * 1e-3
        else:
            depth_m = frame.depth.astype(np.float64)

        return self.from_depth_m(
            depth_m,
            world_camera=world_camera,
            rgba=rgba,
            semantic=semantic,
        )

    def from_depth_mm(
        self,
        depth_mm: np.ndarray,
        *,
        world_camera: Optional[np.ndarray] = None,
        rgba: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Convert a depth image in millimeters to Monty observation format."""
        depth_m = np.asarray(depth_mm, dtype=np.float64) * 1e-3
        return self.from_depth_m(
            depth_m,
            world_camera=world_camera,
            rgba=rgba,
            semantic=semantic,
        )

    def from_depth_m(
        self,
        depth_m: np.ndarray,
        *,
        world_camera: Optional[np.ndarray] = None,
        rgba: Optional[np.ndarray] = None,
        semantic: Optional[np.ndarray] = None,
    ) -> dict[str, np.ndarray]:
        """Convert a depth image in meters to Monty observation format.

        Returns:
            A dictionary compatible with `SensorObservation` entries consumed by
            CameraSM.
        """
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError(f"Expected 2D depth image, got shape {depth.shape}")

        if self._patch_height is not None:
            raw_h, raw_w = depth.shape
            top = raw_h - self._patch_offset_bottom_px - self._patch_height
            left = self._patch_offset_left_px
            if (
                top < 0
                or left < 0
                or top + self._patch_height > raw_h
                or left + self._patch_width > raw_w
            ):
                raise ValueError(
                    f"ROI patch (top={top}, left={left}, "
                    f"h={self._patch_height}, w={self._patch_width}) "
                    f"falls outside raw frame of shape {depth.shape}."
                )
            depth = _crop_to_roi(depth, top, left, self._patch_height, self._patch_width)
            if rgba is not None:
                rgba = _crop_to_roi(
                    rgba, top, left, self._patch_height, self._patch_width
                )
            if semantic is not None:
                semantic = _crop_to_roi(
                    semantic, top, left, self._patch_height, self._patch_width
                )
            effective_intrinsics = CameraIntrinsics(
                fx=self._intrinsics.fx,
                fy=self._intrinsics.fy,
                cx=self._intrinsics.cx - left,
                cy=self._intrinsics.cy - top,
            )
        elif self._crop_center_to_square:
            depth = _center_crop_to_square(depth)
            if rgba is not None:
                rgba = _center_crop_to_square(rgba)
            if semantic is not None:
                semantic = _center_crop_to_square(semantic)
            effective_intrinsics = self._intrinsics
        elif depth.shape[0] != depth.shape[1]:
            raise ValueError(
                "CameraSM assumes square patches. Enable center-cropping or provide "
                "a square depth image."
            )
        else:
            effective_intrinsics = self._intrinsics

        h, w = depth.shape
        world_camera_t = _ensure_world_camera(world_camera)
        rgba_img = _normalize_rgba(rgba, shape=(h, w))
        semantic_mask = _normalize_semantic(
            semantic,
            depth,
            min_valid_depth_m=self._min_valid_depth_m,
            max_valid_depth_m=self._max_valid_depth_m,
        )
        semantic_count_normalized = int(np.sum(semantic_mask > 0))
        if self._semantic_zero_bottom_fraction > 0.0:
            semantic_mask = _zero_semantic_bottom_rows(
                semantic_mask,
                self._semantic_zero_bottom_fraction,
            )
        semantic_count_post_bottom = int(np.sum(semantic_mask > 0))

        depth_valid_mask = np.isfinite(depth) & (depth > self._min_valid_depth_m)
        if self._max_valid_depth_m is not None:
            depth_valid_mask = depth_valid_mask & (depth < self._max_valid_depth_m)
        depth_valid_count = int(np.sum(depth_valid_mask))

        sensor_xyz = _unproject_depth_to_sensor_xyz(depth, effective_intrinsics)

        world_xyz = _transform_xyz(sensor_xyz, world_camera_t)
        semantic_mask = semantic_mask.reshape(-1)
        valid_mask_pre_world = semantic_mask.copy()

        # Optionally filter out background (e.g. table) and out-of-bounds points
        # in the fixed world frame so filtering remains invariant to sensor rotation.
        semantic_mask = _apply_world_bounds_filter(
            semantic_mask,
            world_xyz,
            world_y_min_m=self._world_y_min_m,
            world_x_min_m=self._world_x_min_m,
            world_x_max_m=self._world_x_max_m,
            world_z_min_m=self._world_z_min_m,
            world_z_max_m=self._world_z_max_m,
        )
        semantic_count_post_world = int(np.sum(semantic_mask > 0))

        if self._semantic_debug_logging:
            logger.info(
                "SEMANTIC_FILTER_COUNTS depth_valid=%d normalized=%d post_bottom=%d "
                "pre_world=%d post_world=%d total=%d bounds={y_min:%s,x_min:%s,"
                "x_max:%s,z_min:%s,z_max:%s}",
                depth_valid_count,
                semantic_count_normalized,
                semantic_count_post_bottom,
                int(np.sum(valid_mask_pre_world > 0)),
                semantic_count_post_world,
                int(semantic_mask.shape[0]),
                self._world_y_min_m,
                self._world_x_min_m,
                self._world_x_max_m,
                self._world_z_min_m,
                self._world_z_max_m,
            )

        if os.environ.get(_DEBUG_DUMP_ENV_VAR):
            _dump_pipeline_stage(
                depth_m=depth,
                valid_mask=valid_mask_pre_world,
                sensor_xyz=sensor_xyz,
                world_xyz=world_xyz,
                world_camera=world_camera_t,
                final_semantic_mask=semantic_mask,
                world_bounds={
                    "world_y_min_m": self._world_y_min_m,
                    "world_x_min_m": self._world_x_min_m,
                    "world_x_max_m": self._world_x_max_m,
                    "world_z_min_m": self._world_z_min_m,
                    "world_z_max_m": self._world_z_max_m,
                },
                min_valid_depth_m=self._min_valid_depth_m,
                max_valid_depth_m=self._max_valid_depth_m,
            )

        sensor_xyz4 = np.column_stack([sensor_xyz, semantic_mask])
        world_xyz4 = np.column_stack([world_xyz, semantic_mask])

        return {
            "depth": depth,
            "rgba": rgba_img,
            "semantic_3d": world_xyz4,
            "sensor_frame_data": sensor_xyz4,
            "world_camera": world_camera_t,
        }


def create_adapter_from_http_calibration(
    http_client,
    *,
    crop_center_to_square: bool = True,
    min_valid_depth_m: float = 1e-6,
    max_valid_depth_m: float | None = None,
    semantic_zero_bottom_fraction: float = 0.0,
    world_y_min_m: float | None = None,
    world_x_max_m: float | None = None,
    world_x_min_m: float | None = None,
    world_z_max_m: float | None = None,
    world_z_min_m: float | None = None,
) -> MaixsenseMontyObservationAdapter:
    """Create a Monty adapter using Maixsense HTTP-reported lens coefficients.

    Args:
        http_client: Object with a `get_lens_coefficients()` method (for example,
            `MaixsenseA010HTTP`).
        crop_center_to_square: Whether non-square frames should be center-cropped
            for CameraSM compatibility.
        min_valid_depth_m: Minimum depth used to synthesize semantic validity.
        max_valid_depth_m: Optional maximum depth used to synthesize semantic validity.
        semantic_zero_bottom_fraction: Fraction of bottom rows to zero in semantic mask.
        world_y_min_m: Minimum Monty world-frame Y (meters) for semantic inclusion.  See
            ``MaixsenseMontyObservationAdapter`` for details.
        world_x_min_m: Minimum Monty world-frame X (meters) for semantic inclusion.
        world_x_max_m: Maximum Monty world-frame X (meters) for semantic inclusion.
        world_z_min_m: Minimum Monty world-frame Z (meters) for semantic inclusion.
        world_z_max_m: Maximum Monty world-frame Z (meters) for semantic inclusion.

    Returns:
        Configured `MaixsenseMontyObservationAdapter`.

    Raises:
        RuntimeError: If lens coefficients are unavailable from the device.
    """
    lens = http_client.get_lens_coefficients()
    if lens is None:
        raise RuntimeError(
            "Could not retrieve lens coefficients from Maixsense HTTP /getinfo."
        )

    intrinsics = CameraIntrinsics.from_lens_coefficients(lens)
    return MaixsenseMontyObservationAdapter(
        intrinsics,
        crop_center_to_square=crop_center_to_square,
        min_valid_depth_m=min_valid_depth_m,
        max_valid_depth_m=max_valid_depth_m,
        semantic_zero_bottom_fraction=semantic_zero_bottom_fraction,
        world_y_min_m=world_y_min_m,
        world_x_min_m=world_x_min_m,
        world_x_max_m=world_x_max_m,
        world_z_min_m=world_z_min_m,
        world_z_max_m=world_z_max_m,
    )


def _ensure_world_camera(world_camera: Optional[np.ndarray]) -> np.ndarray:
    if world_camera is None:
        return np.eye(4, dtype=np.float64)

    mat = np.asarray(world_camera, dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"world_camera must be shape (4, 4), got {mat.shape}")
    return mat


def _crop_to_roi(
    arr: np.ndarray, top: int, left: int, height: int, width: int
) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected array with at least 2 dims, got shape {arr.shape}")
    return arr[top : top + height, left : left + width, ...]


def _center_crop_to_square(arr: np.ndarray) -> np.ndarray:
    if arr.ndim < 2:
        raise ValueError(f"Expected array with at least 2 dims, got shape {arr.shape}")

    h, w = arr.shape[:2]
    if h == w:
        return arr

    side = min(h, w)
    top = (h - side) // 2
    left = (w - side) // 2
    return arr[top : top + side, left : left + side, ...]


def _normalize_rgba(rgba: Optional[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    if rgba is None:
        out = np.zeros((h, w, 4), dtype=np.uint8)
        out[..., 3] = 255
        return out

    img = np.asarray(rgba)
    if img.shape[0] != h or img.shape[1] != w:
        raise ValueError(
            f"rgba spatial shape must match depth {(h, w)}, got {img.shape[:2]}"
        )

    if img.ndim == 2:
        rgb = np.repeat(img[..., None], 3, axis=2)
        alpha = np.full((h, w, 1), 255, dtype=rgb.dtype)
        img = np.concatenate([rgb, alpha], axis=2)
    elif img.ndim == 3 and img.shape[2] == 3:
        alpha = np.full((h, w, 1), 255, dtype=img.dtype)
        img = np.concatenate([img, alpha], axis=2)
    elif img.ndim == 3 and img.shape[2] == 4:
        pass
    else:
        raise ValueError("rgba must be HxW, HxWx3, or HxWx4")

    return img


def _normalize_semantic(
    semantic: Optional[np.ndarray],
    depth_m: np.ndarray,
    min_valid_depth_m: float = 1e-6,
    max_valid_depth_m: float | None = None,
) -> np.ndarray:
    if semantic is None:
        valid = np.isfinite(depth_m) & (depth_m > min_valid_depth_m)
        if max_valid_depth_m is not None:
            valid = valid & (depth_m < max_valid_depth_m)
        return valid.astype(np.int32)

    sem = np.asarray(semantic)
    if sem.shape[:2] != depth_m.shape:
        raise ValueError(
            f"semantic spatial shape must match depth {depth_m.shape}, got {sem.shape[:2]}"
        )

    if sem.ndim > 2:
        sem = sem[..., 0]

    return (sem > 0).astype(np.int32)


def _zero_semantic_bottom_rows(semantic_mask: np.ndarray, bottom_fraction: float) -> np.ndarray:
    if bottom_fraction <= 0.0:
        return semantic_mask

    masked = np.array(semantic_mask, copy=True)
    rows = masked.shape[0]
    cut_rows = int(np.floor(rows * bottom_fraction))
    if cut_rows <= 0:
        return masked

    masked[rows - cut_rows :, :] = 0
    return masked


def _apply_world_bounds_filter(
    semantic_mask: np.ndarray,
    world_xyz: np.ndarray,
    *,
    world_y_min_m: float | None = None,
    world_x_min_m: float | None = None,
    world_x_max_m: float | None = None,
    world_z_min_m: float | None = None,
    world_z_max_m: float | None = None,
) -> np.ndarray:
    filtered = np.asarray(semantic_mask, dtype=np.int32)
    include = np.ones(filtered.shape, dtype=bool)

    # World Y maps to robot-base Z (up), so this excludes table/ground points.
    if world_y_min_m is not None:
        include &= world_xyz[:, 1] > world_y_min_m

    if world_x_min_m is not None:
        include &= world_xyz[:, 0] > world_x_min_m
    if world_x_max_m is not None:
        include &= world_xyz[:, 0] < world_x_max_m

    if world_z_min_m is not None:
        include &= world_xyz[:, 2] > world_z_min_m
    if world_z_max_m is not None:
        include &= world_xyz[:, 2] < world_z_max_m

    return (filtered.astype(bool) & include).astype(np.int32)


def _unproject_depth_to_sensor_xyz(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))

    # Keep right-handed coordinates with +x right, +y up, and camera looking along -z
    # to align with Monty's existing transformed observations.
    x = ((u - intrinsics.cx) / intrinsics.fx) * depth_m
    y = -((v - intrinsics.cy) / intrinsics.fy) * depth_m
    z = -depth_m

    return np.column_stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)])


def _transform_xyz(sensor_xyz: np.ndarray, world_camera: np.ndarray) -> np.ndarray:
    ones = np.ones((sensor_xyz.shape[0], 1), dtype=np.float64)
    xyz_h = np.hstack([sensor_xyz, ones])
    xyz_world_h = (world_camera @ xyz_h.T).T
    return xyz_world_h[:, :3]


def _axis_stats(arr: np.ndarray, mask: np.ndarray) -> dict:
    """Per-axis min/max/mean for the rows where mask is truthy.

    Returns:
        Dict with keys ``count``, ``min``, ``max``, ``mean``. The min/max/mean
        values are ``None`` when no rows are selected.
    """
    selected = arr[mask.astype(bool)]
    if selected.size == 0:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": int(selected.shape[0]),
        "min": np.min(selected, axis=0).tolist(),
        "max": np.max(selected, axis=0).tolist(),
        "mean": np.mean(selected, axis=0).tolist(),
    }


def _dump_pipeline_stage(
    *,
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    sensor_xyz: np.ndarray,
    world_xyz: np.ndarray,
    world_camera: np.ndarray,
    final_semantic_mask: np.ndarray,
    world_bounds: dict,
    min_valid_depth_m: float,
    max_valid_depth_m: float | None,
) -> None:
    """Dump one frame's worth of pipeline state for diagnostic inspection.

    Triggered when the MONTY_ADAPTER_DEBUG_DUMP env var is set.
    Saves a .npz alongside a single summary log line.
    """
    dump_dir = Path(
        os.environ.get(_DEBUG_DUMP_DIR_ENV_VAR, _DEFAULT_DEBUG_DUMP_DIR)
    ).expanduser()
    dump_dir.mkdir(parents=True, exist_ok=True)
    millis_suffix = int(time.monotonic() * 1000) % 1000
    timestamp = time.strftime("%Y%m%d_%H%M%S") + f"_{millis_suffix:03d}"
    npz_path = dump_dir / f"frame_{timestamp}.npz"

    h, w = depth_m.shape
    finite_depth = depth_m[np.isfinite(depth_m)]
    nonzero_finite = finite_depth[finite_depth > min_valid_depth_m]
    histogram_max = float(finite_depth.max()) if finite_depth.size > 0 else 0.0
    histogram_edges = np.linspace(0.0, max(histogram_max, 1e-6), 11)
    histogram, _ = np.histogram(finite_depth, bins=histogram_edges)

    center_depth = float(depth_m[h // 2, w // 2])

    sensor_stats = _axis_stats(sensor_xyz, valid_mask)
    world_pre_filter_stats = _axis_stats(world_xyz, valid_mask)
    world_post_filter_stats = _axis_stats(world_xyz, final_semantic_mask)

    summary = {
        "depth_shape": (h, w),
        "depth_min": float(depth_m.min()),
        "depth_max": float(depth_m.max()),
        "depth_mean": float(depth_m.mean()),
        "depth_median": float(np.median(depth_m)),
        "depth_zero_count": int(np.sum(depth_m == 0)),
        "depth_finite_count": int(finite_depth.size),
        "depth_above_min_count": int(nonzero_finite.size),
        "depth_at_center": center_depth,
        "depth_histogram_edges": histogram_edges.tolist(),
        "depth_histogram_counts": histogram.tolist(),
        "valid_mask_pre_world_filter_count": int(valid_mask.sum()),
        "sensor_xyz_valid": sensor_stats,
        "world_xyz_valid_pre_filter": world_pre_filter_stats,
        "world_xyz_post_filter": world_post_filter_stats,
        "world_camera": world_camera.tolist(),
        "world_bounds": world_bounds,
        "min_valid_depth_m": min_valid_depth_m,
        "max_valid_depth_m": max_valid_depth_m,
    }

    np.savez(
        npz_path,
        depth_m=depth_m,
        valid_mask_pre_world_filter=valid_mask.reshape(h, w),
        sensor_xyz=sensor_xyz,
        world_xyz=world_xyz,
        final_semantic_mask=final_semantic_mask.reshape(h, w),
        world_camera=world_camera,
    )

    logger.warning("MONTY_ADAPTER_DEBUG_DUMP %s", summary)
    logger.warning("MONTY_ADAPTER_DEBUG_DUMP saved arrays to %s", npz_path)
