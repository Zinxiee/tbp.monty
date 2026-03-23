# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path

import numpy as np


def _load_module(module_name: str):
    repo_root = Path(__file__).resolve().parents[4]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return importlib.import_module(module_name)


class _FakeLens:
    def __init__(self, fx: float, fy: float, cx: float, cy: float) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy


class _FakeHttpClient:
    def __init__(self, lens) -> None:  # noqa: ANN001
        self._lens = lens

    def get_lens_coefficients(self):  # noqa: ANN201
        return self._lens


class MaixsenseMontyObservationAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.monty_adapter = _load_module(
            "multimodal_monty_meets_world.maixsense_a010_api.monty_adapter"
        )
        self.models = _load_module(
            "multimodal_monty_meets_world.maixsense_a010_api.models"
        )

        self.adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0)
        )

    def test_from_depth_m_emits_camera_sm_required_keys_and_shapes(self) -> None:
        depth = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        obs = self.adapter.from_depth_m(depth)

        self.assertEqual(set(obs.keys()), {
            "depth",
            "rgba",
            "semantic_3d",
            "sensor_frame_data",
            "world_camera",
        })
        self.assertEqual(obs["depth"].shape, (2, 2))
        self.assertEqual(obs["rgba"].shape, (2, 2, 4))
        self.assertEqual(obs["semantic_3d"].shape, (4, 4))
        self.assertEqual(obs["sensor_frame_data"].shape, (4, 4))
        self.assertEqual(obs["world_camera"].shape, (4, 4))

    def test_non_square_input_is_center_cropped_by_default(self) -> None:
        depth = np.arange(15, dtype=np.float64).reshape(3, 5) + 1.0
        obs = self.adapter.from_depth_m(depth)

        self.assertEqual(obs["depth"].shape, (3, 3))

    def test_invalid_world_camera_shape_raises(self) -> None:
        depth = np.ones((2, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            self.adapter.from_depth_m(depth, world_camera=np.eye(3))

    def test_semantic_synthesized_from_valid_depth(self) -> None:
        depth = np.array([[0.0, 1.0], [np.nan, 2.0]], dtype=np.float64)
        obs = self.adapter.from_depth_m(depth)

        semantic = obs["semantic_3d"][:, 3].reshape(2, 2)
        expected = np.array([[0, 1], [0, 1]], dtype=np.int32)
        np.testing.assert_array_equal(semantic, expected)

    def test_from_depth_mm_converts_to_meters_in_z(self) -> None:
        adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        )
        depth_mm = np.array([[1000.0]], dtype=np.float64)

        obs = adapter.from_depth_mm(depth_mm)

        self.assertAlmostEqual(obs["depth"][0, 0], 1.0, places=7)
        self.assertAlmostEqual(obs["sensor_frame_data"][0, 2], -1.0, places=7)

    def test_from_http_frame_without_depth_raises(self) -> None:
        frame = self.models.HttpFrame(
            frame_id=1,
            stamp_msec=0,
            config=None,  # type: ignore[arg-type]
            depth=None,
            ir=None,
            status=None,
            rgb=None,
        )
        with self.assertRaises(ValueError):
            self.adapter.from_http_frame(frame)

    def test_create_adapter_from_http_calibration_uses_device_lens(self) -> None:
        client = _FakeHttpClient(_FakeLens(fx=11.0, fy=12.0, cx=13.0, cy=14.0))

        adapter = self.monty_adapter.create_adapter_from_http_calibration(client)

        self.assertAlmostEqual(adapter.intrinsics.fx, 11.0)
        self.assertAlmostEqual(adapter.intrinsics.fy, 12.0)
        self.assertAlmostEqual(adapter.intrinsics.cx, 13.0)
        self.assertAlmostEqual(adapter.intrinsics.cy, 14.0)

    def test_create_adapter_from_http_calibration_raises_when_missing(self) -> None:
        client = _FakeHttpClient(None)

        with self.assertRaises(RuntimeError):
            self.monty_adapter.create_adapter_from_http_calibration(client)


if __name__ == "__main__":
    unittest.main()
