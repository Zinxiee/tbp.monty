# Copyright 2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from tbp.monty.frameworks.utils.rgbd_conversion_utils import (
    depth_array_to_http_payload_bytes,
    bgra_to_rgba_png_bytes,
)


class IncomingScenesManager:
    """Create and update worldimages scene/version files under incoming_scenes."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path).expanduser().resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)

    def create_next_scene_name(self, prefix: str = "zed_capture") -> str:
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        max_index = 0
        for scene_dir in self.data_path.iterdir():
            if not scene_dir.is_dir():
                continue
            match = pattern.match(scene_dir.name)
            if match is None:
                continue
            max_index = max(max_index, int(match.group(1)))

        return f"{prefix}_{max_index + 1:03d}"

    def create_scene_folder(self, scene_name: str) -> Path:
        scene_path = self.data_path / scene_name
        scene_path.mkdir(parents=True, exist_ok=True)
        return scene_path

    def resolve_scene_path(self, scene_name: str) -> Path:
        scene_path = self.data_path / scene_name
        if not scene_path.exists() or not scene_path.is_dir():
            raise ValueError(f"Scene folder does not exist: {scene_name}")
        return scene_path

    def get_next_version_index(self, scene_path: str | Path) -> int:
        path = Path(scene_path)
        versions: set[int] = set()

        for depth_file in path.glob("depth_*.data"):
            version = self._parse_version(depth_file.name, prefix="depth_", suffix=".data")
            if version is not None:
                versions.add(version)

        for rgb_file in path.glob("rgb_*.png"):
            version = self._parse_version(rgb_file.name, prefix="rgb_", suffix=".png")
            if version is not None:
                versions.add(version)

        return 0 if len(versions) == 0 else max(versions) + 1

    def save_rgbd_capture(
        self,
        scene_path: str | Path,
        version: int,
        rgb_image: np.ndarray,
        depth_array: np.ndarray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        path = Path(scene_path)
        path.mkdir(parents=True, exist_ok=True)

        rgb_path = path / f"rgb_{version}.png"
        depth_path = path / f"depth_{version}.data"
        metadata_path = path / f"metadata_{version}.json"

        rgb_path.write_bytes(bgra_to_rgba_png_bytes(rgb_image))
        depth_path.write_bytes(depth_array_to_http_payload_bytes(depth_array))

        if metadata is not None:
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        result = {
            "rgb_path": rgb_path,
            "depth_path": depth_path,
        }
        if metadata is not None:
            result["metadata_path"] = metadata_path
        return result

    @staticmethod
    def _parse_version(name: str, prefix: str, suffix: str) -> int | None:
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None

        version_text = name[len(prefix) : -len(suffix)]
        if not version_text.isdigit():
            return None

        return int(version_text)
