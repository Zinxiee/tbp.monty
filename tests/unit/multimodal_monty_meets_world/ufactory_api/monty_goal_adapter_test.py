# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult


class _FakeRobotInterface:
    def __init__(self) -> None:
        self.last_command: tuple[float, float, float, float, float, float] | None = None
        self.feasible = True
        self.healthy = True
        self.wait_until_ready_result = True
        self.wait_until_ready_calls = 0
        self.sense_state = {
            "joints": [0.0] * 6,
            "end_effector": [300.0, 0.0, 200.0, 0.0, 0.0, 0.0],
            "api_status": {"joint_code": 0, "position_code": 0},
        }
        self.stop_reason: str | None = None

    def move_arm(self, x, y, z, roll, pitch, yaw) -> None:  # noqa: ANN001
        self.last_command = (x, y, z, roll, pitch, yaw)

    def get_sense_state(self):
        return self.sense_state

    def is_api_healthy(self):
        return self.healthy

    def wait_until_ready(self, timeout_s=2.0, poll_interval_s=0.02):
        self.wait_until_ready_calls += 1
        return self.wait_until_ready_result

    def get_api_health_snapshot(self):
        return {
            "joint_code": 0 if self.healthy else -1,
            "position_code": 0 if self.healthy else -1,
            "error_code": 0 if self.healthy else 23,
        }

    def get_joint_limit_margin_rad(self, _joint_limits_rad):
        return 1.0

    def is_target_pose_feasible(self, *_args):
        return self.feasible

    def stop_motion(self, reason):
        self.stop_reason = reason


def _load_adapter_module():
    repo_root = Path(__file__).resolve().parents[4]
    adapter_dir = repo_root / "multimodal_monty_meets_world" / "ufactory_api"

    # The adapter imports `robot_interface` as a top-level module.
    sys.path.insert(0, str(adapter_dir))

    module_path = adapter_dir / "monty_goal_adapter.py"
    module_spec = importlib.util.spec_from_file_location("monty_goal_adapter", module_path)
    assert module_spec is not None
    assert module_spec.loader is not None

    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


class MontyGoalToRobotAdapterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _load_adapter_module()

    def test_dispatch_goal_pose_sends_mm_and_degrees(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([0.0, -1.0, 0.0]),
                workspace_max_xyz_m=np.array([1.0, 1.0, 1.0]),
            ),
        )

        result = MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
        dispatched = adapter.dispatch_motor_policy_result(result)

        self.assertTrue(dispatched)
        assert robot.last_command is not None
        x, y, z, roll, pitch, yaw = robot.last_command

        self.assertAlmostEqual(x, 300.0, places=6)
        self.assertAlmostEqual(y, 0.0, places=6)
        self.assertAlmostEqual(z, 200.0, places=6)
        self.assertAlmostEqual(roll, 0.0, places=6)
        self.assertAlmostEqual(pitch, 0.0, places=6)
        self.assertAlmostEqual(yaw, 0.0, places=6)

    def test_dispatch_without_goal_pose_returns_false(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
        )

        dispatched = adapter.dispatch_motor_policy_result(MotorPolicyResult(actions=[]))

        self.assertFalse(dispatched)
        self.assertIsNone(robot.last_command)

    def test_link6_to_sensor_offset_is_compensated(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            link6_to_sensor=self.module.Link6ToSensorTransform(
                translation_m=np.array([0.0, -0.06, 0.0135]),
                rotation_quat_wxyz=qt.one,
            ),
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([0.0, -1.0, 0.0]),
                workspace_max_xyz_m=np.array([1.0, 1.0, 1.0]),
            ),
        )

        result = MotorPolicyResult(goal_pose=(np.array([0.30, 0.0, 0.20]), qt.one))
        dispatched = adapter.dispatch_motor_policy_result(result)

        self.assertTrue(dispatched)
        assert robot.last_command is not None
        x, y, z, _, _, _ = robot.last_command

        self.assertAlmostEqual(x, 300.0, places=6)
        self.assertAlmostEqual(y, 60.0, places=6)
        self.assertAlmostEqual(z, 186.5, places=6)

    def test_dispatch_rejected_when_ik_infeasible(self) -> None:
        robot = _FakeRobotInterface()
        robot.feasible = False
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([0.0, -1.0, 0.0]),
                workspace_max_xyz_m=np.array([1.0, 1.0, 1.0]),
            ),
        )

        result = MotorPolicyResult(goal_pose=(np.array([0.30, 0.0, 0.20]), qt.one))
        dispatched = adapter.dispatch_motor_policy_result(result)

        self.assertFalse(dispatched)
        assert robot.stop_reason is not None
        self.assertTrue(robot.stop_reason.startswith("ik_infeasible"))
        self.assertIsNone(robot.last_command)

    def test_dispatch_rejected_when_wait_until_ready_times_out(self) -> None:
        robot = _FakeRobotInterface()
        robot.wait_until_ready_result = False
        robot.healthy = False
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(wait_until_ready_timeout_s=0.0),
        )

        result = MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
        dispatched = adapter.dispatch_motor_policy_result(result)

        self.assertFalse(dispatched)
        self.assertEqual(robot.wait_until_ready_calls, 1)
        assert robot.stop_reason is not None
        self.assertTrue(robot.stop_reason.startswith("robot_not_ready_timeout"))

    def test_command_interval_waits_instead_of_rejecting(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([0.0, -1.0, 0.0]),
                workspace_max_xyz_m=np.array([1.0, 1.0, 1.0]),
                min_command_interval_s=0.05,
                wait_for_min_command_interval=True,
            ),
        )

        result = MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
        self.assertTrue(adapter.dispatch_motor_policy_result(result))
        self.assertTrue(adapter.dispatch_motor_policy_result(result))

        assert robot.stop_reason is None
        assert robot.last_command is not None

    def test_very_relaxed_profile_bypasses_translation_step_rejection(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(
                max_translation_step_m=0.001,
                safety_profile="very_relaxed",
            ),
        )

        self.assertTrue(
            adapter.dispatch_motor_policy_result(
                MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
            )
        )
        self.assertTrue(
            adapter.dispatch_motor_policy_result(
                MotorPolicyResult(goal_pose=(np.array([0.6, 0.0, 0.2]), qt.one))
            )
        )

    def test_world_to_robot_transform_applies_rotation_and_translation(self) -> None:
        robot = _FakeRobotInterface()
        world_to_robot = self.module.WorldToRobotTransform(
            translation_m=np.array([0.1, 0.2, 0.0], dtype=float),
            rotation_quat_wxyz=qt.from_rotation_vector([0.0, 0.0, np.pi / 2]),
        )
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=world_to_robot,
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([-1.0, -1.0, -1.0]),
                workspace_max_xyz_m=np.array([2.0, 2.0, 2.0]),
            ),
        )

        dispatched = adapter.dispatch_motor_policy_result(
            MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
        )

        self.assertTrue(dispatched)
        assert robot.last_command is not None
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = robot.last_command
        self.assertAlmostEqual(x_mm, 100.0, places=3)
        self.assertAlmostEqual(y_mm, 500.0, places=3)
        self.assertAlmostEqual(z_mm, 200.0, places=3)
        self.assertAlmostEqual(roll_deg, 0.0, places=3)
        self.assertAlmostEqual(pitch_deg, 0.0, places=3)
        self.assertAlmostEqual(yaw_deg, 90.0, places=3)

    def test_workspace_rejection_sets_rejection_details(self) -> None:
        robot = _FakeRobotInterface()
        adapter = self.module.MontyGoalToRobotAdapter(
            robot=robot,
            world_to_robot=self.module.identity_world_to_robot_transform(),
            safety_config=self.module.SafetyConfig(
                workspace_min_xyz_m=np.array([0.0, 0.0, 0.0]),
                workspace_max_xyz_m=np.array([0.1, 0.1, 0.1]),
            ),
        )

        dispatched = adapter.dispatch_motor_policy_result(
            MotorPolicyResult(goal_pose=(np.array([0.3, 0.0, 0.2]), qt.one))
        )

        self.assertFalse(dispatched)
        self.assertIsNotNone(adapter.last_rejection_details)
        assert adapter.last_rejection_details is not None
        self.assertEqual(adapter.last_rejection_details["reason_code"], "workspace_bounds")
        self.assertIn("workspace_bounds", adapter.last_rejection_details["details"])


if __name__ == "__main__":
    unittest.main()
