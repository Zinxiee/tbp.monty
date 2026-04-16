# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as rot


class RealWorldLite6ProbeConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[4]
        self.no_goal_dispatch_cfg = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/"
            "lite6_maixsense_unsupervised_probe_no_goal_dispatch.yaml"
        )
        self.probe_baseline_cfg = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/"
            "lite6_maixsense_unsupervised_probe_baseline.yaml"
        )

    def test_no_goal_dispatch_disables_env_dispatch(self) -> None:
        cfg = OmegaConf.load(self.no_goal_dispatch_cfg)

        self.assertFalse(cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch)
        self.assertFalse(cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch)
        self.assertEqual(
            cfg.experiment.config.logging.run_name,
            "real_world_lite6_maixsense_unsupervised_probe_no_goal_dispatch",
        )

    def test_probe_baseline_enables_probe_motion_settings(self) -> None:
        cfg = OmegaConf.load(self.probe_baseline_cfg)
        env_args = cfg.experiment.config.environment.env_init_args

        self.assertTrue(cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch)
        self.assertTrue(cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch)
        self.assertFalse(env_args.require_object_swap_confirmation)
        self.assertTrue(env_args.motion_debug_logging)
        self.assertEqual(env_args.settle_convergence_position_tolerance_mm, 3.0)
        self.assertEqual(env_args.settle_convergence_required_consecutive_samples, 2)
        self.assertTrue(env_args.probe_move_forward_only)
        self.assertEqual(env_args.probe_move_forward_distance_m, 0.01)
        self.assertEqual(env_args.probe_max_steps, 6)
        self.assertTrue(env_args.goal_adapter_config.debug_logging)
        self.assertEqual(
            cfg.experiment.config.logging.run_name,
            "real_world_lite6_maixsense_unsupervised_probe_baseline",
        )


class SemanticMaskWorldYFilterTest(unittest.TestCase):
    """Tests for the world_y_min_m table-exclusion filter in MaixsenseMontyObservationAdapter."""

    def _make_adapter(self, world_y_min_m=None):
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )
        intrinsics = CameraIntrinsics(fx=226.5, fy=227.9, cx=50.0, cy=50.0)
        return MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_y_min_m=world_y_min_m,
        )

    def _identity_world_camera(self):
        return np.eye(4, dtype=np.float64)

    def test_no_filter_all_valid_pixels_are_on_object(self) -> None:
        """Without world_y_min_m, every finite-depth pixel is on the object."""
        adapter = self._make_adapter(world_y_min_m=None)
        depth = np.full((4, 4), 0.30, dtype=np.float64)

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())

        semantic = obs["semantic_3d"][:, 3]
        self.assertTrue(np.all(semantic == 1))

    def test_world_y_filter_excludes_low_points(self) -> None:
        """Pixels whose world-Y is below world_y_min_m must be excluded from semantics.

        With identity world_camera (sensor optical frame = world frame) and
        Monty's convention (z = -depth), world Y = -(v - cy) / fy * depth.
        A pixel at v=0 (above principal point) has positive world Y;
        a pixel at v=cy (exactly at principal point) has world Y = 0.
        We use world_y_min_m = 0.01 and verify that pixels at or below 0.01 m
        world-Y are excluded while pixels above are included.
        """
        # 3x3 depth image, all at 0.20 m.
        # With cy=1.0, pixel row 0 → world Y = -((0-1)/fy)*0.2 = +0.2/fy ≈ +0.00088 m
        # That is below 0.01, so all pixels here should be EXCLUDED.
        intrinsics_cy1 = None
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )
        # Use fy=1 so world Y = -(v - cy) * depth for easy arithmetic.
        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        adapter = MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_y_min_m=0.05,
        )
        # 3x3, depth=0.10 m everywhere.
        depth = np.full((3, 3), 0.10, dtype=np.float64)
        # With fy=1, cy=1: world_y(row=v) = -((v - 1) / 1) * 0.10
        #   v=0 → world_y = +0.10  > 0.05 → included
        #   v=1 → world_y = 0.00  < 0.05 → excluded
        #   v=2 → world_y = -0.10 < 0.05 → excluded

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())
        semantic = obs["semantic_3d"][:, 3].reshape(3, 3)

        # Row 0 (v=0): included
        nptest.assert_array_equal(semantic[0, :], [1, 1, 1])
        # Row 1 (v=1): excluded
        nptest.assert_array_equal(semantic[1, :], [0, 0, 0])
        # Row 2 (v=2): excluded
        nptest.assert_array_equal(semantic[2, :], [0, 0, 0])

    def test_world_x_max_filter_excludes_points_at_or_above_threshold(self) -> None:
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )

        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        adapter = MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_x_max_m=0.0,
        )
        depth = np.full((3, 3), 0.10, dtype=np.float64)

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())
        semantic = obs["semantic_3d"][:, 3].reshape(3, 3)

        nptest.assert_array_equal(semantic[:, 0], [1, 1, 1])
        nptest.assert_array_equal(semantic[:, 1], [0, 0, 0])
        nptest.assert_array_equal(semantic[:, 2], [0, 0, 0])

    def test_world_x_min_filter_excludes_points_at_or_below_threshold(self) -> None:
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )

        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=1.0, cy=1.0)
        adapter = MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_x_min_m=0.0,
        )
        depth = np.full((3, 3), 0.10, dtype=np.float64)

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())
        semantic = obs["semantic_3d"][:, 3].reshape(3, 3)

        nptest.assert_array_equal(semantic[:, 0], [0, 0, 0])
        nptest.assert_array_equal(semantic[:, 1], [0, 0, 0])
        nptest.assert_array_equal(semantic[:, 2], [1, 1, 1])

    def test_world_z_max_filter_excludes_points_not_beyond_threshold(self) -> None:
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )

        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        adapter = MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_z_max_m=-0.10,
        )
        depth = np.array([[0.05, 0.10], [0.15, 0.20]], dtype=np.float64)

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())
        semantic = obs["semantic_3d"][:, 3].reshape(2, 2)

        nptest.assert_array_equal(semantic, [[0, 0], [1, 1]])

    def test_world_z_min_filter_excludes_points_at_or_below_threshold(self) -> None:
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            MaixsenseMontyObservationAdapter,
        )

        intrinsics = CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        adapter = MaixsenseMontyObservationAdapter(
            intrinsics,
            crop_center_to_square=False,
            world_z_min_m=-0.10,
        )
        depth = np.array([[0.05, 0.10], [0.15, 0.20]], dtype=np.float64)

        obs = adapter.from_depth_m(depth, world_camera=self._identity_world_camera())
        semantic = obs["semantic_3d"][:, 3].reshape(2, 2)

        nptest.assert_array_equal(semantic, [[1, 0], [0, 0]])


class RealWorldSurfacePolicyTest(unittest.TestCase):
    """Tests for RealWorldSurfacePolicy."""

    def test_touch_sensor_id_returns_patch(self) -> None:
        from multimodal_monty_meets_world.real_world_surface_policy import (
            RealWorldSurfacePolicy,
        )
        from tbp.monty.frameworks.actions.action_samplers import (
            UniformlyDistributedSampler,
        )
        from tbp.monty.frameworks.actions.actions import LookUp
        from tbp.monty.frameworks.agents import AgentID
        from tbp.monty.frameworks.sensors import SensorID

        agent_id = AgentID("agent_id_0")
        policy = RealWorldSurfacePolicy(
            alpha=0.1,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=agent_id,
            desired_object_distance=0.12,
        )

        self.assertEqual(policy._touch_sensor_id(), SensorID("patch"))

    def test_custom_patch_sensor_id_is_respected(self) -> None:
        from multimodal_monty_meets_world.real_world_surface_policy import (
            RealWorldSurfacePolicy,
        )
        from tbp.monty.frameworks.actions.action_samplers import (
            UniformlyDistributedSampler,
        )
        from tbp.monty.frameworks.actions.actions import LookUp
        from tbp.monty.frameworks.agents import AgentID
        from tbp.monty.frameworks.sensors import SensorID

        agent_id = AgentID("agent_id_0")
        policy = RealWorldSurfacePolicy(
            alpha=0.1,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=agent_id,
            desired_object_distance=0.12,
            patch_sensor_id="my_sensor",
        )

        self.assertEqual(policy._touch_sensor_id(), SensorID("my_sensor"))

    def test_disable_orient_compensation_translation_zeros_offsets(self) -> None:
        from multimodal_monty_meets_world.real_world_surface_policy import (
            RealWorldSurfacePolicy,
        )
        from tbp.monty.frameworks.actions.action_samplers import (
            UniformlyDistributedSampler,
        )
        from tbp.monty.frameworks.actions.actions import LookUp
        from tbp.monty.frameworks.agents import AgentID

        agent_id = AgentID("agent_id_0")
        policy = RealWorldSurfacePolicy(
            alpha=0.1,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=agent_id,
            desired_object_distance=0.12,
            disable_orient_compensation_translation=True,
        )
        percept = mock.Mock()
        percept.get_feature_by_name.return_value = 0.2

        lateral, forward = policy._compensating_distances(15.0, percept)

        self.assertEqual(lateral, 0.0)
        self.assertEqual(forward, 0.0)

    def test_orient_decomposition_logging_emits_diagnostics(self) -> None:
        from multimodal_monty_meets_world.real_world_surface_policy import (
            RealWorldSurfacePolicy,
        )
        from tbp.monty.frameworks.actions.action_samplers import (
            UniformlyDistributedSampler,
        )
        from tbp.monty.frameworks.actions.actions import LookUp
        from tbp.monty.frameworks.agents import AgentID

        class _FakeAgentState:
            def __init__(self):
                self.rotation = qt.one

        agent_id = AgentID("agent_id_0")
        policy = RealWorldSurfacePolicy(
            alpha=0.1,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=agent_id,
            desired_object_distance=0.12,
            enable_orient_decomposition_logging=True,
        )
        state = {agent_id: _FakeAgentState()}
        percept = mock.Mock()
        percept.get_feature_by_name.return_value = 0.2
        percept.get_surface_normal.return_value = np.array([0.2, -0.3, 0.9])

        with self.assertLogs(
            "multimodal_monty_meets_world.real_world_surface_policy",
            level="INFO",
        ) as captured:
            policy._log_orient_diagnostics(
                "horizontal",
                raw_angle=5.0,
                rotation_degrees=5.0,
                lateral=0.01,
                forward=0.001,
                state=state,
                percept=percept,
            )

        self.assertTrue(
            any("orient_horizontal_decomposition" in msg for msg in captured.output)
        )


class MotionValidationConfigTest(unittest.TestCase):
    """Tests for the motion validation experiment config and action file."""

    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[4]
        self.actions_file = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/motion_validation_actions.jsonl"
        )
        self.config_file = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/"
            "lite6_maixsense_motion_validation.yaml"
        )
        self.policy_file = self.repo_root / (
            "src/tbp/monty/conf/monty/motor_system_config/policy/"
            "predefined_validation.yaml"
        )

    def test_motion_validation_action_file_has_14_actions(self) -> None:
        lines = [
            line for line in self.actions_file.read_text().splitlines() if line.strip()
        ]
        self.assertEqual(len(lines), 14)

    def test_motion_validation_action_file_covers_all_four_types(self) -> None:
        import json

        actions = [
            json.loads(line)
            for line in self.actions_file.read_text().splitlines()
            if line.strip()
        ]
        types_present = {a["action"] for a in actions}
        for expected in (
            "move_forward",
            "move_tangentially",
            "orient_horizontal",
            "orient_vertical",
        ):
            self.assertIn(expected, types_present)

    def test_motion_validation_config_key_fields(self) -> None:
        cfg = OmegaConf.load(self.config_file)
        exp = cfg.experiment.config
        self.assertEqual(exp.max_train_steps, 14)
        self.assertEqual(exp.max_eval_steps, 0)
        self.assertEqual(exp.n_eval_epochs, 0)
        self.assertEqual(
            exp.logging.run_name, "real_world_lite6_maixsense_motion_validation"
        )
        self.assertEqual(exp.environment.env_init_args.settle_time_s, 1.5)
        self.assertFalse(exp.environment.env_init_args.probe_move_forward_only)
        policy = exp.monty_config.motor_system_config.policy_selector.policy
        self.assertEqual(
            policy._target_,
            "tbp.monty.frameworks.models.motor_policies.PredefinedPolicy",
        )
        self.assertTrue(policy.terminate_on_exhaustion)


class Link6ToSensorFrameTest(unittest.TestCase):
    """Verify link6_to_sensor_rotation_dict is consistent with physical mounting.

    Physical facts:
    - Sensor optical axis (Monty -Z = look direction) faces link6 +Z (downward
      when tool roll=180° at home pose).
    - Image top (sensor +Y) faces link6 +X (robot/Monty forward = world -Z).
    - Image right (sensor +X) faces link6 +Y.

    These constraints uniquely determine the rotation R_link6_sensor and the
    quaternion stored in the YAML is [w=0, x=1/√2, y=1/√2, z=0].
    """

    def _build_rotation_chain(self, link6_to_sensor_wxyz):
        """Return world-frame rotation for the sensor at home pose (roll=180°)."""
        # world_to_robot rotation from YAML: w=0.5, x=0.5, y=-0.5, z=-0.5
        w2r = rot.from_quat([0.5, -0.5, -0.5, 0.5])  # [x,y,z,w]

        # Home pose: roll=180°, pitch=0°, yaw≈0° in robot-base xyz Euler extrinsic
        r_tcp = rot.from_euler("xyz", [180.0, 0.0, 0.0], degrees=True)

        # world_link6_rot = world_to_robot.inv() * R_tcp
        world_link6 = w2r.inv() * r_tcp

        # Apply link6_to_sensor rotation
        w, x, y, z = link6_to_sensor_wxyz
        link6_to_sensor = rot.from_quat([x, y, z, w])
        return world_link6 * link6_to_sensor

    def test_sensor_looks_downward_at_home_pose(self) -> None:
        """Sensor -Z (look direction) must point world -Y (downward) at home."""
        wxyz = (0.0, 0.70710678, 0.70710678, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        sensor_forward_in_world = world_rot.apply([0.0, 0.0, -1.0])
        nptest.assert_allclose(sensor_forward_in_world, [0.0, -1.0, 0.0], atol=1e-6)

    def test_image_top_faces_robot_forward_at_home_pose(self) -> None:
        """Sensor +Y (image top) must point world -Z (robot forward) at home."""
        wxyz = (0.0, 0.70710678, 0.70710678, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        image_top_in_world = world_rot.apply([0.0, 1.0, 0.0])
        nptest.assert_allclose(image_top_in_world, [0.0, 0.0, -1.0], atol=1e-6)

    def test_identity_rotation_fails_look_direction(self) -> None:
        """Confirm identity link6_to_sensor gives the wrong look direction.

        With identity the sensor +Z (not -Z) ends up pointing world +Y (up),
        meaning the sensor would look upward — the pre-fix miscalibration.
        """
        wxyz = (1.0, 0.0, 0.0, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        sensor_forward_in_world = world_rot.apply([0.0, 0.0, -1.0])
        # Should NOT point downward (world -Y)
        self.assertFalse(
            np.allclose(sensor_forward_in_world, [0.0, -1.0, 0.0], atol=0.1),
            "Identity rotation unexpectedly gives the correct look direction",
        )


if __name__ == "__main__":
    unittest.main()
