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

    def move_arm(self, x, y, z, roll, pitch, yaw) -> None:  # noqa: ANN001
        self.last_command = (x, y, z, roll, pitch, yaw)


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


if __name__ == "__main__":
    unittest.main()
