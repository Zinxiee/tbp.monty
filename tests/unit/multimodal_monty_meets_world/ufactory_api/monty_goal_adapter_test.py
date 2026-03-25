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
        return True

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
        self.assertEqual(robot.stop_reason, "ik_infeasible")
        self.assertIsNone(robot.last_command)


if __name__ == "__main__":
    unittest.main()
