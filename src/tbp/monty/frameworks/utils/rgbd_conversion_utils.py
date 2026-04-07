# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def bgra_to_rgba_png_bytes(bgra_image: np.ndarray) -> bytes:
    """Convert a BGRA image array to RGBA PNG bytes.

    Args:
        bgra_image: Image array with shape (H, W, 4) and uint8-compatible values.

    Returns:
        PNG bytes encoded in RGBA mode.
    """
    if bgra_image.ndim != 3 or bgra_image.shape[2] != 4:
        raise ValueError(
            f"Expected image with shape (H, W, 4), got {bgra_image.shape}"
        )

    image_uint8 = np.clip(bgra_image, 0, 255).astype(np.uint8)
    rgba = cv2.cvtColor(image_uint8, cv2.COLOR_BGRA2RGBA)
    pil_image = Image.fromarray(rgba, mode="RGBA")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return buffer.getvalue()


def depth_array_to_http_payload_bytes(depth_array: np.ndarray) -> bytes:
    """Convert a depth array to raw float32 payload bytes.

    The output matches the depth_*.data format expected by worldimages scenes.

    Args:
        depth_array: Depth values with shape (H, W).

    Returns:
        Raw bytes of a flattened float32 array.
    """
    if depth_array.ndim != 2:
        raise ValueError(
            f"Expected 2D depth array with shape (H, W), got {depth_array.shape}"
        )

    depth_float32 = depth_array.astype(np.float32)
    return depth_float32.flatten().tobytes()


def depth_to_vis(depth_array: np.ndarray) -> np.ndarray:
    """Create a colorized depth visualization for debugging and QA."""
    valid = np.isfinite(depth_array) & (depth_array > 0)
    if not np.any(valid):
        return np.zeros((*depth_array.shape, 3), dtype=np.uint8)

    low = np.percentile(depth_array[valid], 2)
    high = np.percentile(depth_array[valid], 98)
    if high <= low:
        high = low + 1e-6

    scaled = np.zeros(depth_array.shape, dtype=np.float32)
    scaled[valid] = np.clip((depth_array[valid] - low) / (high - low), 0.0, 1.0)

    vis = np.zeros(depth_array.shape, dtype=np.uint8)
    vis[valid] = np.rint(scaled[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)
