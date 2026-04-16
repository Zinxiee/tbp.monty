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

    def test_roi_patch_crop_shape_and_unprojection(self) -> None:
        raw_h, raw_w = 100, 100
        patch_h, patch_w = 10, 10
        offset_bottom, offset_left = 20, 45
        top = raw_h - offset_bottom - patch_h
        left = offset_left

        fx, fy, cx, cy = 71.41, 86.60, 50.0, 50.0
        adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy),
            patch_height=patch_h,
            patch_width=patch_w,
            patch_offset_bottom_px=offset_bottom,
            patch_offset_left_px=offset_left,
        )

        yy, xx = np.mgrid[0:raw_h, 0:raw_w]
        depth = (0.1 + 0.001 * (yy + xx)).astype(np.float64)

        obs = adapter.from_depth_m(depth)
        self.assertEqual(obs["depth"].shape, (patch_h, patch_w))

        cropped = depth[top : top + patch_h, left : left + patch_w]
        u, v = np.meshgrid(
            np.arange(patch_w, dtype=np.float64),
            np.arange(patch_h, dtype=np.float64),
        )
        expected_x = ((u - (cx - left)) / fx) * cropped
        expected_y = -((v - (cy - top)) / fy) * cropped
        expected_z = -cropped
        np.testing.assert_allclose(
            obs["sensor_frame_data"][:, 0], expected_x.reshape(-1)
        )
        np.testing.assert_allclose(
            obs["sensor_frame_data"][:, 1], expected_y.reshape(-1)
        )
        np.testing.assert_allclose(
            obs["sensor_frame_data"][:, 2], expected_z.reshape(-1)
        )

    def test_roi_patch_out_of_bounds_raises(self) -> None:
        adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=1.0, fy=1.0, cx=50.0, cy=50.0),
            patch_height=10,
            patch_width=10,
            patch_offset_bottom_px=95,
            patch_offset_left_px=0,
        )
        depth = np.ones((100, 100), dtype=np.float64)
        with self.assertRaises(ValueError):
            adapter.from_depth_m(depth)

    def test_roi_patch_params_require_all_or_none(self) -> None:
        with self.assertRaises(ValueError):
            self.monty_adapter.MaixsenseMontyObservationAdapter(
                self.monty_adapter.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
                patch_height=10,
            )

    def test_create_adapter_from_http_calibration_raises_when_missing(self) -> None:
        client = _FakeHttpClient(None)

        with self.assertRaises(RuntimeError):
            self.monty_adapter.create_adapter_from_http_calibration(client)

    def test_semantic_debug_logging_emits_filter_counts(self) -> None:
        adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0),
            semantic_debug_logging=True,
        )
        depth = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float64)

        with self.assertLogs(
            "multimodal_monty_meets_world.maixsense_a010_api.monty_adapter",
            level="INFO",
        ) as captured:
            adapter.from_depth_m(depth)

        self.assertTrue(
            any("SEMANTIC_FILTER_COUNTS" in msg for msg in captured.output)
        )

    def test_y_min_only_can_zero_mask_and_reports_reject_counts(self) -> None:
        adapter = self.monty_adapter.MaixsenseMontyObservationAdapter(
            self.monty_adapter.CameraIntrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0),
            semantic_debug_logging=True,
            world_y_min_m=0.01,
        )
        depth = np.full((2, 2), 0.1, dtype=np.float64)
        world_camera = np.eye(4, dtype=np.float64)
        world_camera[1, 3] = -0.2

        with self.assertLogs(
            "multimodal_monty_meets_world.maixsense_a010_api.monty_adapter",
            level="INFO",
        ) as captured:
            obs = adapter.from_depth_m(depth, world_camera=world_camera)

        semantic = obs["semantic_3d"][:, 3]
        np.testing.assert_array_equal(semantic, np.zeros_like(semantic))

        self.assertTrue(
            any(
                (
                    "SEMANTIC_FILTER_COUNTS" in msg
                    and "pre_world=4" in msg
                    and "post_world=0" in msg
                    and "rejects={y_min:4,x_min:0,x_max:0,z_min:0,z_max:0,any:4}" in msg
                )
                for msg in captured.output
            )
        )


if __name__ == "__main__":
    unittest.main()
