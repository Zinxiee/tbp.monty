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

        self.assertFalse(
            cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch
        )
        self.assertFalse(
            cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch
        )
        self.assertEqual(
            cfg.experiment.config.logging.run_name,
            "real_world_lite6_maixsense_unsupervised_probe_no_goal_dispatch",
        )

    def test_probe_baseline_enables_probe_motion_settings(self) -> None:
        cfg = OmegaConf.load(self.probe_baseline_cfg)
        env_args = cfg.experiment.config.environment.env_init_args

        self.assertTrue(
            cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch
        )
        self.assertTrue(
            cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch
        )
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

    def test_orient_vertical_large_raw_angle_has_cm_scale_translation(self) -> None:
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
        )
        state = {agent_id: _FakeAgentState()}
        percept = mock.Mock()

        def _feature_by_name(name: str):
            if name == "object_coverage":
                return 0.9  # High coverage: no throttling
            return 0.2

        percept.get_feature_by_name.side_effect = _feature_by_name

        with mock.patch.object(
            policy,
            "orienting_angle_from_normal",
            return_value=66.129,
        ), mock.patch.object(
            policy,
            "_filtered_forward_depth_from_stashed_obs",
            return_value=0.1121,
        ):
            action = policy._orient_vertical(state, percept)

        self.assertAlmostEqual(action.rotation_degrees, 15.0, places=6)
        # Fix 5 suppresses positive down_distance for vertical orient
        # (prevents cumulative downward drift toward the table).
        self.assertAlmostEqual(action.down_distance, 0.0, places=5)
        self.assertAlmostEqual(action.forward_distance, 0.003954, places=5)


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
    - Image right (sensor +X) points toward robot base = link6 -X at home.
    - Image top (sensor +Y) points robot-right = link6 +Y at home = world +X.

    These constraints uniquely determine the rotation R_link6_sensor = Ry(180°)
    and the quaternion stored in the YAML is [w=0, x=0, y=1, z=0].
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
        wxyz = (0.0, 0.0, 1.0, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        sensor_forward_in_world = world_rot.apply([0.0, 0.0, -1.0])
        nptest.assert_allclose(sensor_forward_in_world, [0.0, -1.0, 0.0], atol=1e-6)

    def test_image_right_faces_robot_base_at_home_pose(self) -> None:
        """Sensor +X (image right) must point world +Z (toward robot base) at home."""
        wxyz = (0.0, 0.0, 1.0, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        image_right_in_world = world_rot.apply([1.0, 0.0, 0.0])
        nptest.assert_allclose(image_right_in_world, [0.0, 0.0, 1.0], atol=1e-6)

    def test_image_top_faces_world_right_at_home_pose(self) -> None:
        """Sensor +Y (image top) must point world +X (Monty right) at home."""
        wxyz = (0.0, 0.0, 1.0, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        image_top_in_world = world_rot.apply([0.0, 1.0, 0.0])
        nptest.assert_allclose(image_top_in_world, [1.0, 0.0, 0.0], atol=1e-6)

    def test_old_quaternion_gives_wrong_image_orientation(self) -> None:
        """Old Rz(-90)@Ry(180) quaternion maps image right to world +X, not +Z.

        The prior quaternion [w=0, x=1/sqrt2, y=1/sqrt2, z=0] had an erroneous
        Rz(-90) component that rotated the point cloud 90° in the horizontal
        plane, causing the agent to move into objects instead of along surfaces.
        """
        wxyz = (0.0, 0.70710678, 0.70710678, 0.0)
        world_rot = self._build_rotation_chain(wxyz)
        image_right_in_world = world_rot.apply([1.0, 0.0, 0.0])
        # Old quaternion incorrectly maps image right to world +X (right)
        nptest.assert_allclose(image_right_in_world, [1.0, 0.0, 0.0], atol=1e-6)
        # It should NOT map to world +Z (toward robot base) — that's the fix
        self.assertFalse(
            np.allclose(image_right_in_world, [0.0, 0.0, 1.0], atol=0.1),
            "Old quaternion unexpectedly gives correct image orientation",
        )


class WorldTransformPipelineTest(unittest.TestCase):
    """Verify the full sensor→world transform maps table pixels correctly.

    Uses the actual _unproject_depth_to_sensor_xyz (radial-corrected) and
    _transform_xyz functions from the adapter to confirm that a flat table at
    a known world height is reconstructed accurately.

    Physical setup (from YAML):
    - world_to_robot quaternion wxyz: [0.5, 0.5, -0.5, -0.5]
    - link6_to_sensor quaternion wxyz: [0, √2/2, √2/2, 0]
    - link6_to_sensor translation (link6 frame): [0, -0.060, 0.0135]
    - Robot base is 7mm above the physical table (table at robot z = -7mm)
    - Home pose: robot [220, 9, 210] mm, roll=180°, pitch=0°, yaw=0°
    - Patch intrinsics: fx=71.41, fy=86.60, cx_eff=5.0, cy_eff=-20.0
    """

    TABLE_WORLD_Y_M = -0.007  # table is 7mm below robot base origin

    def _world_camera_at_home(self):
        """Build a 4x4 world_camera matrix for the home pose."""
        w2r = rot.from_quat([0.5, -0.5, -0.5, 0.5])  # xyzw
        r_tcp = rot.from_euler("xyz", [np.pi, 0.0, 0.0], degrees=False)
        world_link6_rot = w2r.inv() * r_tcp

        l2s_rot = rot.from_quat([0.70710678, 0.70710678, 0.0, 0.0])  # xyzw
        l2s_trans = np.array([0.0, -0.060, 0.0135])

        robot_pos_m = np.array([0.220, 0.009, 0.210])
        world_link6_pos = w2r.inv().apply(robot_pos_m)
        world_sensor_pos = world_link6_pos + world_link6_rot.apply(l2s_trans)
        world_sensor_rot = world_link6_rot * l2s_rot

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = world_sensor_rot.as_matrix()
        mat[:3, 3] = world_sensor_pos
        return mat

    def _world_camera_tilted(self, tilt_deg):
        """Build world_camera tilted by OrientVertical from home."""
        w2r = rot.from_quat([0.5, -0.5, -0.5, 0.5])
        r_tcp = rot.from_euler("xyz", [np.pi, 0.0, 0.0], degrees=False)
        world_link6_rot = w2r.inv() * r_tcp

        l2s_rot = rot.from_quat([0.70710678, 0.70710678, 0.0, 0.0])
        l2s_trans = np.array([0.0, -0.060, 0.0135])

        world_sensor_rot_home = world_link6_rot * l2s_rot
        # OrientVertical: rotate around sensor X axis
        x_tilt = rot.from_rotvec([np.radians(tilt_deg), 0.0, 0.0])
        world_sensor_rot = world_sensor_rot_home * x_tilt

        robot_pos_m = np.array([0.220, 0.009, 0.210])
        world_link6_pos = w2r.inv().apply(robot_pos_m)
        world_sensor_pos = world_link6_pos + world_link6_rot.apply(l2s_trans)

        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = world_sensor_rot.as_matrix()
        mat[:3, 3] = world_sensor_pos
        return mat

    def _make_table_radial_depth(self, world_camera, patch_h=10, patch_w=10):
        """Generate synthetic radial depth for a flat table at TABLE_WORLD_Y_M.

        For each pixel, compute the ray direction in world frame, intersect
        with the table plane (world Y = TABLE_WORLD_Y_M), and return the
        radial distance (Euclidean distance from sensor origin to hit point).
        """
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
        )

        intrinsics = CameraIntrinsics(fx=71.41, fy=86.60, cx=5.0, cy=-20.0)
        R = world_camera[:3, :3]
        t = world_camera[:3, 3]

        depth = np.zeros((patch_h, patch_w), dtype=np.float64)
        for v in range(patch_h):
            for u in range(patch_w):
                dir_x = (u - intrinsics.cx) / intrinsics.fx
                dir_y = -((v - intrinsics.cy) / intrinsics.fy)
                dir_z = -1.0
                ray_sensor = np.array([dir_x, dir_y, dir_z])
                ray_sensor /= np.linalg.norm(ray_sensor)
                ray_world = R @ ray_sensor

                # Intersect with plane world Y = TABLE_WORLD_Y_M
                if abs(ray_world[1]) < 1e-12:
                    depth[v, u] = 0.0
                    continue
                t_hit = (self.TABLE_WORLD_Y_M - t[1]) / ray_world[1]
                if t_hit < 0:
                    depth[v, u] = 0.0
                    continue
                depth[v, u] = t_hit  # radial distance
        return depth, intrinsics

    def test_home_pose_table_pixels_world_y(self):
        """At home pose, all table pixels should map to world Y ≈ -0.007."""
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            _transform_xyz,
            _unproject_depth_to_sensor_xyz,
        )

        wc = self._world_camera_at_home()
        depth, intrinsics = self._make_table_radial_depth(wc)
        valid = depth > 0.01
        self.assertTrue(np.all(valid), "Some pixels have no table intersection")

        sensor_xyz = _unproject_depth_to_sensor_xyz(depth, intrinsics)
        world_xyz = _transform_xyz(sensor_xyz, wc)

        world_y = world_xyz[:, 1]
        nptest.assert_allclose(
            world_y,
            self.TABLE_WORLD_Y_M,
            atol=0.002,
            err_msg=(
                f"Table pixels should all be at world Y ≈ {self.TABLE_WORLD_Y_M}, "
                f"got Y range [{world_y.min():.4f}, {world_y.max():.4f}]"
            ),
        )

    def test_tilted_30deg_table_pixels_world_y(self):
        """After 30° OrientVertical tilt, table pixels still at Y ≈ -0.007."""
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            _transform_xyz,
            _unproject_depth_to_sensor_xyz,
        )

        wc = self._world_camera_tilted(30.0)
        depth, intrinsics = self._make_table_radial_depth(wc)
        valid = depth > 0.01
        self.assertTrue(np.all(valid), "Some pixels have no table intersection")

        sensor_xyz = _unproject_depth_to_sensor_xyz(depth, intrinsics)
        world_xyz = _transform_xyz(sensor_xyz, wc)

        world_y = world_xyz[:, 1]
        nptest.assert_allclose(
            world_y,
            self.TABLE_WORLD_Y_M,
            atol=0.002,
            err_msg=(
                f"Tilted table pixels should be at Y ≈ {self.TABLE_WORLD_Y_M}, "
                f"got Y range [{world_y.min():.4f}, {world_y.max():.4f}]"
            ),
        )

    def test_radial_vs_zdist_error_magnitude(self):
        """Radial correction should reduce world Y error vs old z-distance formula.

        The old formula (z = -depth) overshoots for off-axis pixels. With the
        10×10 patch at cy_eff=-20 and depth ≈ 0.12m, the error should be ~5mm.
        """
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            CameraIntrinsics,
            _transform_xyz,
        )

        wc = self._world_camera_at_home()
        depth, intrinsics = self._make_table_radial_depth(wc)

        h, w = depth.shape
        u, v = np.meshgrid(
            np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64)
        )

        # Old (z-distance) unprojection
        x_old = ((u - intrinsics.cx) / intrinsics.fx) * depth
        y_old = -((v - intrinsics.cy) / intrinsics.fy) * depth
        z_old = -depth
        old_sensor = np.column_stack([x_old.ravel(), y_old.ravel(), z_old.ravel()])
        old_world = _transform_xyz(old_sensor, wc)

        # New (radial) unprojection
        from multimodal_monty_meets_world.maixsense_a010_api.monty_adapter import (
            _unproject_depth_to_sensor_xyz,
        )

        new_sensor = _unproject_depth_to_sensor_xyz(depth, intrinsics)
        new_world = _transform_xyz(new_sensor, wc)

        old_y_error = np.abs(old_world[:, 1] - self.TABLE_WORLD_Y_M)
        new_y_error = np.abs(new_world[:, 1] - self.TABLE_WORLD_Y_M)

        # Old method should have ~3-7mm error
        self.assertGreater(
            old_y_error.max(),
            0.002,
            "Old z-distance method should have measurable Y error",
        )
        # New method should have <2mm error
        self.assertLess(
            new_y_error.max(),
            0.002,
            "Radial correction should bring Y error below 2mm",
        )


if __name__ == "__main__":
    unittest.main()
