# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import types
import unittest

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.real_world_environment import (
    RealWorldLite6A010Environment,
    RealWorldSafetyStopError,
)
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult


class _FakeRobot:
    def __init__(self, end_effector: list[float] | None = None) -> None:
        self._end_effector = end_effector or [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.stop_motion_calls: list[str] = []

    def get_sense_state(self) -> dict[str, list[float]]:
        return {"end_effector": self._end_effector}

    def stop_motion(self, reason: str) -> None:
        self.stop_motion_calls.append(reason)


class _FakeSensor:
    def get_frame(self, timeout_s: float = 1.0) -> np.ndarray:  # noqa: ARG002
        return np.ones((2, 2), dtype=np.float32)


class _FakeObservationAdapter:
    def from_depth_m(
        self,
        frame: np.ndarray,
        world_camera: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return {
            "depth": frame,
            "world_camera": world_camera,
        }


class _FakeGoalAdapter:
    def __init__(self, dispatch_ok: bool = True) -> None:
        self.dispatch_ok = dispatch_ok
        self.last_rejection_details = {
            "reason_code": "TEST_REJECT",
            "details": "simulated rejection",
        }
        self.safety_config = types.SimpleNamespace(
            max_translation_step_m=0.05,
            max_rotation_step_deg=10.0,
        )

    def dispatch_motor_policy_result(self, result: MotorPolicyResult) -> bool:  # noqa: ARG002
        return self.dispatch_ok

    def send_world_goal_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> bool:  # noqa: ARG002
        return True


class RealWorldEnvironmentMathTest(unittest.TestCase):
    def _make_env(
        self,
        *,
        robot: _FakeRobot | None = None,
        goal_adapter: _FakeGoalAdapter | None = None,
        goal_rejection_hard_stop: bool = False,
        sensor_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sensor_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    ) -> RealWorldLite6A010Environment:
        return RealWorldLite6A010Environment(
            robot_interface=robot or _FakeRobot(),
            sensor_client=_FakeSensor(),
            observation_adapter=_FakeObservationAdapter(),
            goal_adapter=goal_adapter,
            input_fn=lambda prompt: "",  # noqa: ARG005
            settle_time_s=0.0,
            require_object_swap_confirmation=False,
            goal_rejection_hard_stop=goal_rejection_hard_stop,
            sensor_translation_m=sensor_translation_m,
            sensor_rotation_wxyz=sensor_rotation_wxyz,
        )

    def test_relative_move_forward_goal_pose_identity_rotation(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        goal_pos, goal_quat = env._goal_pose_from_relative_action(
            MoveForward(agent_id=AgentID("agent_id_0"), distance=0.02)
        )

        nptest.assert_allclose(goal_pos, np.array([0.0, 0.0, 0.02]))
        nptest.assert_allclose(qt.as_float_array(goal_quat), np.array([1.0, 0.0, 0.0, 0.0]))

    def test_relative_move_tangentially_invalid_direction_hard_stops(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        with self.assertRaises(RealWorldSafetyStopError) as exc:
            env._goal_pose_from_relative_action(
                MoveTangentially(
                    agent_id=AgentID("agent_id_0"),
                    distance=0.01,
                    direction=(0.0, 0.0, 0.0),
                )
            )

        self.assertEqual(exc.exception.reason_code, "INVALID_ACTION_PARAMETERS")

    def test_orient_horizontal_rotation_is_clipped_by_safety_config(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        _, goal_quat = env._goal_pose_from_relative_action(
            OrientHorizontal(
                agent_id=AgentID("agent_id_0"),
                rotation_degrees=90.0,
                left_distance=0.0,
                forward_distance=0.0,
            )
        )

        expected = rot.from_euler("z", 10.0, degrees=True).as_quat()
        expected_wxyz = np.array([expected[3], expected[0], expected[1], expected[2]])
        nptest.assert_allclose(qt.as_float_array(goal_quat), expected_wxyz, atol=1e-7)

    def test_translation_step_is_clipped_by_safety_config(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        clipped = env._clip_translation_step(np.array([0.2, 0.0, 0.0]))

        nptest.assert_allclose(clipped, np.array([0.05, 0.0, 0.0]))

    def test_dispatch_goal_pose_rejection_respects_config_soft_mode(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter(dispatch_ok=False))
        result = MotorPolicyResult(
            actions=[],
            goal_pose=(np.array([0.1, 0.0, 0.2]), qt.one),
        )

        accepted = env._dispatch_goal_pose(result)

        self.assertFalse(accepted)

    def test_dispatch_goal_pose_rejection_respects_config_hard_stop(self) -> None:
        env = self._make_env(
            goal_adapter=_FakeGoalAdapter(dispatch_ok=False),
            goal_rejection_hard_stop=True,
        )
        result = MotorPolicyResult(
            actions=[],
            goal_pose=(np.array([0.1, 0.0, 0.2]), qt.one),
        )

        with self.assertRaises(RealWorldSafetyStopError) as exc:
            env._dispatch_goal_pose(result)

        self.assertEqual(exc.exception.reason_code, "GOAL_DISPATCH_REJECTED")

    def test_world_camera_matrix_uses_sensor_extrinsics(self) -> None:
        env = self._make_env(
            goal_adapter=_FakeGoalAdapter(),
            sensor_translation_m=(0.1, 0.2, 0.3),
            sensor_rotation_wxyz=(0.0, 1.0, 0.0, 0.0),
        )

        world_camera = env._compute_world_camera_matrix()

        nptest.assert_allclose(world_camera[:3, 3], np.array([0.1, 0.2, 0.3]))
        expected_rot = rot.from_quat([1.0, 0.0, 0.0, 0.0]).as_matrix()
        nptest.assert_allclose(world_camera[:3, :3], expected_rot)

    def test_proprioceptive_state_rotates_sensor_offset_with_agent_pose(self) -> None:
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 0.0, 0.0, np.pi / 2])
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            sensor_translation_m=(0.1, 0.0, 0.0),
        )

        proprio = env._build_proprioceptive_state()
        sensor_state = proprio[AgentID("agent_id_0")].sensors["patch"]

        nptest.assert_allclose(np.array(sensor_state.position), np.array([0.0, 0.1, 0.0]), atol=1e-7)


if __name__ == "__main__":
    unittest.main()
