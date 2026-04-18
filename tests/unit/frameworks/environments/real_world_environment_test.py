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
from unittest.mock import patch

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.actions.actions import (
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    SetAgentPose,
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
        self._timestamp_s = 0.0
        self.stop_motion_calls: list[str] = []
        self.move_arm_calls: list[tuple[float, float, float, float, float, float]] = []

    def get_sense_state(self) -> dict[str, list[float]]:
        self._timestamp_s += 0.01
        return {"end_effector": self._end_effector, "timestamp_s": self._timestamp_s}

    def stop_motion(self, reason: str) -> None:
        self.stop_motion_calls.append(reason)

    def move_arm(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        self.move_arm_calls.append((x, y, z, roll, pitch, yaw))


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
        self.dispatch_stop_on_rejection_calls: list[bool] = []
        self.send_stop_on_rejection_calls: list[bool] = []
        self._last_command_position_m = np.array([0.0, 0.0, 0.0], dtype=float)
        self.last_rejection_details = {
            "reason_code": "TEST_REJECT",
            "details": "simulated rejection",
        }
        self.world_to_robot = types.SimpleNamespace(
            translation_m=np.zeros(3, dtype=float),
            rotation_quat_wxyz=qt.one,
        )
        self.link6_to_sensor = types.SimpleNamespace(
            translation_m=np.zeros(3, dtype=float),
            rotation_quat_wxyz=qt.one,
        )
        self.safety_config = types.SimpleNamespace(
            max_translation_step_m=0.05,
            max_rotation_step_deg=10.0,
        )

    def dispatch_motor_policy_result(
        self,
        result: MotorPolicyResult,  # noqa: ARG002
        *,
        stop_on_rejection: bool = True,
    ) -> bool:
        self.dispatch_stop_on_rejection_calls.append(bool(stop_on_rejection))
        return self.dispatch_ok

    def send_world_goal_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
        *,
        stop_on_rejection: bool = True,
    ) -> bool:
        self.send_stop_on_rejection_calls.append(bool(stop_on_rejection))
        self.last_sent_location_m = np.asarray(location_m, dtype=float).copy()
        self.last_sent_rotation_wxyz = rotation_quat_wxyz
        return self.dispatch_ok


class RealWorldEnvironmentMathTest(unittest.TestCase):
    def _assert_quat_equivalent(
        self,
        actual: qt.quaternion,
        expected_wxyz: np.ndarray,
        *,
        atol: float = 1e-7,
    ) -> None:
        actual_wxyz = qt.as_float_array(actual)
        if np.allclose(actual_wxyz, expected_wxyz, atol=atol):
            return
        nptest.assert_allclose(actual_wxyz, -expected_wxyz, atol=atol)

    def _make_env(
        self,
        *,
        robot: _FakeRobot | None = None,
        goal_adapter: _FakeGoalAdapter | None = None,
        goal_rejection_hard_stop: bool = False,
        sensor_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sensor_rotation_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
        **extra_env_kwargs,
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
            **extra_env_kwargs,
        )

    def test_relative_move_forward_goal_pose_identity_rotation(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        goal_pos, goal_quat = env._goal_pose_from_relative_action(
            MoveForward(agent_id=AgentID("agent_id_0"), distance=0.02)
        )

        nptest.assert_allclose(goal_pos, np.array([0.0, 0.0, -0.02]))
        self._assert_quat_equivalent(goal_quat, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_get_agent_pose_world_applies_world_to_robot_inverse_transform(self) -> None:
        robot = _FakeRobot(end_effector=[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_adapter = _FakeGoalAdapter()
        goal_adapter.world_to_robot = types.SimpleNamespace(
            translation_m=np.array([0.1, 0.2, 0.0], dtype=float),
            rotation_quat_wxyz=qt.from_rotation_vector(np.array([0.0, 0.0, np.pi / 2])),
        )
        env = self._make_env(robot=robot, goal_adapter=goal_adapter)

        world_pos, world_quat = env._get_agent_pose_world()

        expected_pos = rot.from_rotvec([0.0, 0.0, -np.pi / 2]).apply(
            np.array([1.0, 0.0, 0.0]) - np.array([0.1, 0.2, 0.0])
        )
        nptest.assert_allclose(world_pos, expected_pos, atol=1e-7)
        expected_world_quat_xyzw = rot.from_rotvec([0.0, 0.0, -np.pi / 2]).as_quat()
        expected_world_quat_wxyz = np.array(
            [
                expected_world_quat_xyzw[3],
                expected_world_quat_xyzw[0],
                expected_world_quat_xyzw[1],
                expected_world_quat_xyzw[2],
            ]
        )
        self._assert_quat_equivalent(world_quat, expected_world_quat_wxyz, atol=1e-7)

    def test_get_agent_pose_world_applies_link6_to_sensor_transform(self) -> None:
        robot = _FakeRobot(end_effector=[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        goal_adapter = _FakeGoalAdapter()
        goal_adapter.link6_to_sensor = types.SimpleNamespace(
            translation_m=np.array([0.0, -0.06, 0.0135], dtype=float),
            rotation_quat_wxyz=qt.one,
        )
        env = self._make_env(robot=robot, goal_adapter=goal_adapter)

        world_pos, world_quat = env._get_agent_pose_world()

        nptest.assert_allclose(world_pos, np.array([1.0, -0.06, 0.0135]), atol=1e-7)
        self._assert_quat_equivalent(world_quat, np.array([1.0, 0.0, 0.0, 0.0]))

    def test_get_agent_pose_world_parses_sensed_orientation_degrees(self) -> None:
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 180.0, 0.0, 0.0])
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            sensed_orientation_degrees=True,
        )

        _, world_quat = env._get_agent_pose_world()

        self._assert_quat_equivalent(world_quat, np.array([0.0, 1.0, 0.0, 0.0]))

    def test_get_agent_pose_world_applies_orientation_index_order_and_signs(self) -> None:
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 0.1, 0.2, 0.3])
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            sensed_orientation_index_order=(2, 1, 0),
            sensed_orientation_signs=(-1.0, 1.0, 1.0),
        )

        _, world_quat = env._get_agent_pose_world()

        expected_xyzw = rot.from_euler("xyz", np.array([-0.3, 0.2, 0.1])).as_quat()
        expected_wxyz = np.array(
            [expected_xyzw[3], expected_xyzw[0], expected_xyzw[1], expected_xyzw[2]]
        )
        self._assert_quat_equivalent(world_quat, expected_wxyz)

    def test_get_agent_pose_world_applies_orientation_offset(self) -> None:
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        offset_quat = qt.from_rotation_vector(np.array([0.0, 0.0, np.pi / 2]))
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            sensed_orientation_offset_wxyz=qt.as_float_array(offset_quat),
        )

        _, world_quat = env._get_agent_pose_world()

        self._assert_quat_equivalent(world_quat, qt.as_float_array(offset_quat))

    def test_move_home_waits_for_fresh_convergence(self) -> None:
        robot = _FakeRobot(end_effector=[330.0, 20.0, 280.0, 170.0, 0.0, 0.0])
        home_pose = (336.3, 10.7, 287.7, 180.0, 0.0, 0.0)
        states = [
            {"end_effector": [330.0, 20.0, 280.0, 170.0, 0.0, 0.0], "timestamp_s": 0.1},
            {"end_effector": [336.3, 10.7, 287.7, np.pi, 0.0, 0.0], "timestamp_s": 0.2},
        ]

        def get_state() -> dict[str, list[float]]:
            return states.pop(0) if states else {"end_effector": [336.3, 10.7, 287.7, np.pi, 0.0, 0.0], "timestamp_s": 0.3}

        robot.get_sense_state = get_state  # type: ignore[method-assign]
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            home_pose_mm_deg=home_pose,
            home_reset_timeout_s=0.5,
            home_reset_poll_s=0.0,
        )

        with patch(
            "tbp.monty.frameworks.environments.real_world_environment.time.monotonic",
            side_effect=[0.0, 0.05, 0.06, 0.07],
        ):
            env._move_home_if_configured()

        self.assertEqual(len(robot.move_arm_calls), 1)

    def test_move_home_times_out_when_pose_is_not_fresh_or_converged(self) -> None:
        home_pose = (336.3, 10.7, 287.7, 180.0, 0.0, 0.0)
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        robot.get_sense_state = lambda: {  # type: ignore[method-assign]
            "end_effector": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "timestamp_s": -1.0,
        }
        env = self._make_env(
            robot=robot,
            goal_adapter=_FakeGoalAdapter(),
            home_pose_mm_deg=home_pose,
            home_reset_timeout_s=0.01,
            home_reset_poll_s=0.0,
        )

        with patch(
            "tbp.monty.frameworks.environments.real_world_environment.time.monotonic",
            side_effect=[0.0, 0.02, 0.04],
        ):
            with self.assertRaises(RealWorldSafetyStopError) as exc:
                env._move_home_if_configured()

        self.assertEqual(exc.exception.reason_code, "HOME_RESET_TIMEOUT")

    def test_relative_move_forward_uses_parsed_sensed_orientation(self) -> None:
        robot = _FakeRobot(end_effector=[0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0])
        env = self._make_env(robot=robot, goal_adapter=_FakeGoalAdapter())

        goal_pos, _ = env._goal_pose_from_relative_action(
            MoveForward(agent_id=AgentID("agent_id_0"), distance=0.02)
        )

        nptest.assert_allclose(goal_pos, np.array([-0.02, 0.0, 0.0]), atol=1e-7)

    def test_relative_move_forward_applies_translation_only_motion_offset(self) -> None:
        env = self._make_env(
            goal_adapter=_FakeGoalAdapter(),
            sensed_motion_offset_wxyz=qt.as_float_array(
                qt.from_rotation_vector(np.array([0.0, np.pi / 2, 0.0]))
            ),
        )

        goal_pos, goal_quat = env._goal_pose_from_relative_action(
            MoveForward(agent_id=AgentID("agent_id_0"), distance=0.02)
        )

        nptest.assert_allclose(goal_pos, np.array([-0.02, 0.0, 0.0]), atol=1e-7)
        self._assert_quat_equivalent(goal_quat, np.array([1.0, 0.0, 0.0, 0.0]))

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
        self._assert_quat_equivalent(goal_quat, expected_wxyz, atol=1e-7)

    def test_translation_step_is_clipped_by_safety_config(self) -> None:
        env = self._make_env(goal_adapter=_FakeGoalAdapter())

        clipped = env._clip_translation_step(np.array([0.2, 0.0, 0.0]))

        nptest.assert_allclose(clipped, np.array([0.05, 0.0, 0.0]))

    def test_dispatch_goal_pose_rejection_respects_config_soft_mode(self) -> None:
        goal_adapter = _FakeGoalAdapter(dispatch_ok=False)
        env = self._make_env(goal_adapter=goal_adapter)
        result = MotorPolicyResult(
            actions=[],
            goal_pose=(np.array([0.1, 0.0, 0.2]), qt.one),
        )

        accepted = env._dispatch_goal_pose(result, actions=[])

        self.assertFalse(accepted)
        self.assertEqual(goal_adapter.send_stop_on_rejection_calls, [False])

    def test_dispatch_goal_pose_rejection_respects_config_hard_stop(self) -> None:
        goal_adapter = _FakeGoalAdapter(dispatch_ok=False)
        env = self._make_env(
            goal_adapter=goal_adapter,
            goal_rejection_hard_stop=True,
        )
        result = MotorPolicyResult(
            actions=[],
            goal_pose=(np.array([0.1, 0.0, 0.2]), qt.one),
        )

        with self.assertRaises(RealWorldSafetyStopError) as exc:
            env._dispatch_goal_pose(result, actions=[])

        self.assertEqual(exc.exception.reason_code, "GOAL_DISPATCH_REJECTED")
        self.assertEqual(goal_adapter.send_stop_on_rejection_calls, [True])

    def test_block_until_settled_uses_convergence_gate_when_enabled(self) -> None:
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        goal_adapter._last_command_position_m = np.array([0.3363, 0.0107, 0.2877])
        robot = _FakeRobot(end_effector=[330.0, 10.7, 287.7, 0.0, 0.0, 0.0])
        states = [
            {"end_effector": [330.0, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.1},
            {"end_effector": [336.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.2},
        ]

        def get_state() -> dict[str, list[float]]:
            return states.pop(0) if states else {"end_effector": [336.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.3}

        robot.get_sense_state = get_state  # type: ignore[method-assign]
        env = self._make_env(
            robot=robot,
            goal_adapter=goal_adapter,
            settle_use_goal_convergence_gate=True,
            settle_convergence_timeout_s=0.5,
            settle_convergence_position_tolerance_mm=5.0,
            settle_convergence_poll_s=0.0,
        )
        env._step_dispatched_command = True

        with patch(
            "tbp.monty.frameworks.environments.real_world_environment.time.monotonic",
            side_effect=[0.0, 0.1, 0.2],
        ):
            env._block_until_settled()

        self.assertEqual(len(states), 0)

    def test_step_convergence_requires_consecutive_in_tolerance_samples(self) -> None:
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        goal_adapter._last_command_position_m = np.array([0.3363, 0.0107, 0.2877])
        robot = _FakeRobot(end_effector=[336.3, 10.7, 287.7, 0.0, 0.0, 0.0])
        states = [
            {"end_effector": [335.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.1},
            {"end_effector": [325.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.2},
            {"end_effector": [336.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.3},
            {"end_effector": [336.3, 10.7, 287.7, 0.0, 0.0, 0.0], "timestamp_s": 0.4},
        ]

        def get_state() -> dict[str, list[float]]:
            return states.pop(0) if states else {
                "end_effector": [336.3, 10.7, 287.7, 0.0, 0.0, 0.0],
                "timestamp_s": 0.5,
            }

        robot.get_sense_state = get_state  # type: ignore[method-assign]
        env = self._make_env(
            robot=robot,
            goal_adapter=goal_adapter,
            settle_use_goal_convergence_gate=True,
            settle_convergence_timeout_s=0.5,
            settle_convergence_position_tolerance_mm=5.0,
            settle_convergence_required_consecutive_samples=2,
            settle_convergence_poll_s=0.0,
        )

        with patch(
            "tbp.monty.frameworks.environments.real_world_environment.time.monotonic",
            side_effect=[0.0, 0.1, 0.2, 0.3],
        ):
            env._wait_for_step_command_convergence()

        self.assertEqual(len(states), 0)
        self.assertEqual(robot.stop_motion_calls, [])

    def test_step_convergence_timeout_triggers_hard_stop(self) -> None:
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        goal_adapter._last_command_position_m = np.array([0.3363, 0.0107, 0.2877])
        robot = _FakeRobot(end_effector=[300.0, 10.7, 287.7, 0.0, 0.0, 0.0])
        robot.get_sense_state = lambda: {  # type: ignore[method-assign]
            "end_effector": [300.0, 10.7, 287.7, 0.0, 0.0, 0.0],
            "timestamp_s": 0.1,
        }
        env = self._make_env(
            robot=robot,
            goal_adapter=goal_adapter,
            settle_use_goal_convergence_gate=True,
            settle_convergence_timeout_s=0.01,
            settle_convergence_position_tolerance_mm=5.0,
            settle_convergence_required_consecutive_samples=2,
            settle_convergence_poll_s=0.0,
        )

        with patch(
            "tbp.monty.frameworks.environments.real_world_environment.time.monotonic",
            side_effect=[0.0, 0.02],
        ):
            with self.assertRaises(RealWorldSafetyStopError) as exc:
                env._wait_for_step_command_convergence()

        self.assertEqual(exc.exception.reason_code, "SETTLE_CONVERGENCE_TIMEOUT")
        self.assertEqual(len(robot.stop_motion_calls), 1)

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


    def test_dispatch_goal_pose_clips_large_orient_horizontal(self) -> None:
        """Goal dispatch must clip rotation and translation like the fallback."""
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        env = self._make_env(goal_adapter=goal_adapter)

        action = OrientHorizontal(
            agent_id=AgentID("agent_id_0"),
            rotation_degrees=45.0,
            left_distance=0.12,
            forward_distance=0.05,
        )
        # Policy's unclipped goal_pose (would send 45 deg + ~13 cm).
        unclipped_goal = (np.array([0.12, 0.0, -0.05]), qt.one)
        result = MotorPolicyResult(
            actions=[action],
            goal_pose=unclipped_goal,
        )

        accepted = env._dispatch_goal_pose(result, actions=[action])

        self.assertTrue(accepted)
        # The sent pose must come from _goal_pose_from_relative_action which
        # clips rotation to max_rotation_step_deg=10 and translation to
        # max_translation_step_m=0.05.
        sent_pos = goal_adapter.last_sent_location_m
        self.assertIsNotNone(sent_pos)
        step_norm = float(np.linalg.norm(sent_pos))
        self.assertLessEqual(step_norm, 0.05 + 1e-9)

    def test_dispatch_and_fallback_produce_identical_results(self) -> None:
        """Both code paths must produce the same goal for the same action."""
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        env = self._make_env(goal_adapter=goal_adapter)

        action = OrientHorizontal(
            agent_id=AgentID("agent_id_0"),
            rotation_degrees=30.0,
            left_distance=0.08,
            forward_distance=0.03,
        )

        # Fallback path result.
        fallback_pos, fallback_quat = env._goal_pose_from_relative_action(action)

        # Dispatch path — the sent pose should match the fallback exactly.
        result = MotorPolicyResult(
            actions=[action],
            goal_pose=(np.array([9.9, 9.9, 9.9]), qt.one),  # bogus
        )
        env._dispatch_goal_pose(result, actions=[action])

        nptest.assert_allclose(
            goal_adapter.last_sent_location_m, fallback_pos, atol=1e-12
        )
        self._assert_quat_equivalent(
            goal_adapter.last_sent_rotation_wxyz,
            qt.as_float_array(fallback_quat),
            atol=1e-12,
        )

    def test_dispatch_goal_pose_preserves_set_agent_pose(self) -> None:
        """SetAgentPose actions should pass the policy's goal_pose unclipped."""
        goal_adapter = _FakeGoalAdapter(dispatch_ok=True)
        env = self._make_env(goal_adapter=goal_adapter)

        set_pose_action = SetAgentPose(
            agent_id=AgentID("agent_id_0"),
            location=(0.5, 0.3, -0.2),
            rotation_quat=qt.from_euler_angles(0, 0, np.radians(60)),
        )
        policy_goal = (
            np.array([0.5, 0.3, -0.2]),
            qt.from_euler_angles(0, 0, np.radians(60)),
        )
        result = MotorPolicyResult(
            actions=[set_pose_action],
            goal_pose=policy_goal,
        )

        accepted = env._dispatch_goal_pose(result, actions=[set_pose_action])

        self.assertTrue(accepted)
        nptest.assert_allclose(
            goal_adapter.last_sent_location_m,
            policy_goal[0],
            atol=1e-12,
        )
        self._assert_quat_equivalent(
            goal_adapter.last_sent_rotation_wxyz,
            qt.as_float_array(policy_goal[1]),
            atol=1e-12,
        )


class DepthBurstAveragingTest(unittest.TestCase):
    """Burst-average N consecutive depth frames per observation step."""

    class _SequencedSensor:
        def __init__(self, frames: list[np.ndarray]) -> None:
            self._frames = frames
            self._i = 0
            self.get_frame_calls = 0

        def get_frame(self, timeout_s: float = 1.0) -> np.ndarray:  # noqa: ARG002
            self.get_frame_calls += 1
            frame = self._frames[self._i % len(self._frames)]
            self._i += 1
            return frame

    class _CapturingAdapter:
        def __init__(self) -> None:
            self.from_depth_m_calls: list[dict] = []

        def from_depth_m(
            self,
            frame: np.ndarray,
            world_camera: np.ndarray,
        ) -> dict[str, np.ndarray]:
            self.from_depth_m_calls.append(
                {"frame": np.array(frame, copy=True), "world_camera": world_camera}
            )
            return {"depth": frame, "world_camera": world_camera}

    def _make_env(
        self,
        *,
        frames: list[np.ndarray],
        depth_burst_n: int,
    ) -> tuple[RealWorldLite6A010Environment, _CapturingAdapter, _SequencedSensor]:
        sensor = DepthBurstAveragingTest._SequencedSensor(frames)
        adapter = DepthBurstAveragingTest._CapturingAdapter()
        env = RealWorldLite6A010Environment(
            robot_interface=_FakeRobot(),
            sensor_client=sensor,
            observation_adapter=adapter,
            goal_adapter=None,
            input_fn=lambda prompt: "",  # noqa: ARG005
            settle_time_s=0.0,
            require_object_swap_confirmation=False,
            depth_burst_n=depth_burst_n,
        )
        return env, adapter, sensor

    def test_burst_averages_per_pixel_across_n_frames(self) -> None:
        frames = [
            np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
            np.array([[2.0, 3.0], [4.0, 5.0]], dtype=np.float64),
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
            np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float64),
            np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        ]
        env, adapter, sensor = self._make_env(frames=frames, depth_burst_n=5)

        env._capture_sensor_observation()

        self.assertEqual(sensor.get_frame_calls, 5)
        self.assertEqual(len(adapter.from_depth_m_calls), 1)
        nptest.assert_allclose(
            adapter.from_depth_m_calls[0]["frame"],
            np.array([[3.0, 4.0], [5.0, 6.0]], dtype=np.float64),
        )

    def test_burst_skips_zero_pixels_in_mean_and_preserves_zero_holes(self) -> None:
        # Pixel (0,0) is zero in 3 of 5 frames; valid mean should be (3+5)/2=4.
        # Pixel (1,1) is zero in every frame; output must stay 0.
        frames = [
            np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64),
            np.array([[0.0, 2.0], [3.0, 0.0]], dtype=np.float64),
            np.array([[3.0, 2.0], [3.0, 0.0]], dtype=np.float64),
            np.array([[5.0, 2.0], [3.0, 0.0]], dtype=np.float64),
        ]
        env, adapter, _ = self._make_env(frames=frames, depth_burst_n=5)

        env._capture_sensor_observation()

        nptest.assert_allclose(
            adapter.from_depth_m_calls[0]["frame"],
            np.array([[3.0, 2.0], [3.0, 0.0]], dtype=np.float64),
        )

    def test_burst_n_equals_one_reads_single_frame(self) -> None:
        frames = [np.array([[7.0, 7.0], [7.0, 7.0]], dtype=np.float64)]
        env, adapter, sensor = self._make_env(frames=frames, depth_burst_n=1)

        env._capture_sensor_observation()

        self.assertEqual(sensor.get_frame_calls, 1)
        nptest.assert_allclose(
            adapter.from_depth_m_calls[0]["frame"],
            frames[0],
        )


if __name__ == "__main__":
    unittest.main()
