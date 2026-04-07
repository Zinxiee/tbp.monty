# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as nptest
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.cmp import Message
from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import (
    Action,
    ActionJSONEncoder,
    LookDown,
    LookUp,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    TurnLeft,
    TurnRight,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    MotorSystemState,
    SensorState,
)
from tbp.monty.frameworks.models.motor_policies import (
    InformedPolicy,
    PredefinedPolicy,
    SurfacePolicy,
    SurfacePolicyCurvatureInformed,
)
from tbp.monty.frameworks.models.motor_system_state import MotorSystemState
from tests.unit.frameworks.models.fakes.cmp import FakeMessage
from tbp.monty.frameworks.sensors import SensorID
from tbp.monty.frameworks.utils.transform_utils import numpy_to_scipy_quat


class SurfacePolicyCurvatureInformedTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.policy = SurfacePolicyCurvatureInformed(
            alpha=0.1,
            pc_alpha=0.5,
            max_pc_bias_steps=32,
            min_general_steps=8,
            min_heading_steps=12,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            desired_object_distance=0.025,
        )
        self.location = np.array([1.0, 2.0, 3.0])
        self.tangent_norm = np.array([0, 1, 0])
        self.percept = Message(
            location=self.location,
            morphological_features={
                "pose_vectors": np.array(
                    [self.tangent_norm.tolist(), [1, 0, 0], [0, 0, -1]]
                ),
                "pose_fully_defined": True,
                "on_object": 1,
            },
            non_morphological_features={
                "principal_curvatures_log": [0, 0.5],
                "hsv": [0, 1, 1],
            },
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def test_appends_to_tangent_locs_and_tangent_norms_if_last_action_is_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(len(self.policy.tangent_norms), 1)
        nptest.assert_array_equal(self.policy.tangent_norms[0], self.tangent_norm)

    def test_appends_none_to_tangent_norms_if_last_action_is_orient_vertical_but_no_pose_vectors_in_state(  # noqa: E501
        self,
    ):
        del self.percept.morphological_features["pose_vectors"]
        self.policy.last_surface_policy_action = OrientVertical(
            agent_id=self.agent_id,
            rotation_degrees=90,
            down_distance=1,
            forward_distance=1,
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(len(self.policy.tangent_locs), 1)
        nptest.assert_array_equal(self.policy.tangent_locs[0], self.location)
        self.assertEqual(self.policy.tangent_norms, [None])

    def test_does_not_append_to_tangent_locs_and_tangent_norms_if_last_action_is_not_orient_vertical(  # noqa: E501
        self,
    ):
        self.policy.last_surface_policy_action = LookUp(
            agent_id=self.agent_id, rotation_degrees=0
        )

        with patch("tbp.monty.frameworks.models.motor_policies.SurfacePolicy.__call__"):
            self.policy(
                RuntimeContext(rng=np.random.RandomState(42)),
                Observations(),
                MotorSystemState(),
                self.percept,
                None,
            )

        self.assertEqual(self.policy.tangent_locs, [])
        self.assertEqual(self.policy.tangent_norms, [])


class PredefinedPolicyReadActionFileTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.actions_file = Path(__file__).parent / "motor_policies_test_actions.jsonl"

    def test_read_action_file(self) -> None:
        # For this test, we write our own actions to a temporary file instead of
        # loading a file on disk. It's a better guarantee that we're loading the
        # actions exactly as expected.
        expected = [
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
            LookDown(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnLeft(agent_id=self.agent_id, rotation_degrees=10.0),
            LookUp(
                agent_id=self.agent_id,
                rotation_degrees=10.0,
                constraint_degrees=90.0,
            ),
            TurnRight(agent_id=self.agent_id, rotation_degrees=5.0),
        ]
        with tempfile.TemporaryDirectory() as data_path:
            actions_file = Path(data_path) / "actions.jsonl"
            actions_file.write_text(
                "\n".join(json.dumps(a, cls=ActionJSONEncoder) for a in expected) + "\n"
            )
            loaded = PredefinedPolicy.read_action_file(actions_file)
            self.assertEqual(len(loaded), len(expected))
            for loaded_action, expected_action in zip(loaded, expected):
                self.assertEqual(dict(loaded_action), dict(expected_action))

    def test_cycles_continuously(self) -> None:
        policy = PredefinedPolicy(
            agent_id=self.agent_id,
            file_name=self.actions_file,
        )
        cycle_length = len(policy.action_list)
        ctx = RuntimeContext(rng=np.random.RandomState(42))
        observations = Observations()
        returned_actions: list[Action] = []
        for _ in range(2 * cycle_length):
            result = policy(ctx, observations, MotorSystemState(), FakeMessage(), None)
            assert len(result.actions) == 1, "Expected one action"
            returned_actions.append(result.actions[0])

        for i in range(cycle_length):
            first_occurrence = returned_actions[i]
            second_occurrence = returned_actions[i + cycle_length]
            self.assertEqual(first_occurrence, second_occurrence)


class GoalPoseEmissionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_goal_pose")

        self.policy = InformedPolicy(
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            use_goal_state_driven_actions=True,
        )

        self.state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors={
                        SensorID("view_finder"): SensorState(
                            position=(0.0, 0.0, 0.0),
                            rotation=(1.0, 0.0, 0.0, 0.0),
                        )
                    },
                    position=(0.0, 0.0, 0.0),
                    rotation=(1.0, 0.0, 0.0, 0.0),
                )
            }
        )

        self.goal_state = Message(
            location=np.array([0.1, 0.2, 0.3]),
            morphological_features={
                "pose_vectors": np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                "pose_fully_defined": True,
            },
            non_morphological_features={},
            confidence=1.0,
            use_state=True,
            sender_id="patch",
            sender_type="SM",
        )

    def test_goal_driven_jump_populates_goal_pose(self) -> None:
        self.policy.set_driving_goal_state(self.goal_state)

        result = self.policy._goal_driven_actions(
            observations=Observations(),
            state=self.state,
        )

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIsNotNone(result.goal_pose)
        assert result.goal_pose is not None
        target_loc, _ = result.goal_pose
        nptest.assert_allclose(target_loc, np.array([0.1, 0.2, 0.3]))

        self.assertTrue(any(isinstance(action, SetAgentPose) for action in result.actions))

    def test_numpy_to_scipy_quat_accepts_numpy_quaternion(self) -> None:
        quat_xyzw = numpy_to_scipy_quat(qt.one)
        nptest.assert_allclose(quat_xyzw, np.array([0.0, 0.0, 0.0, 1.0]))


class SurfacePolicyGoalPoseFromActionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_surface")
        self.policy = SurfacePolicy(
            alpha=0.1,
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
            desired_object_distance=0.025,
        )

        self.state = MotorSystemState(
            {
                self.agent_id: AgentState(
                    sensors={
                        SensorID("patch"): SensorState(
                            position=(0.0, 0.0, 0.0),
                            rotation=qt.one,
                        )
                    },
                    position=(1.0, 2.0, 3.0),
                    rotation=qt.one,
                )
            }
        )

    def test_move_forward_goal_pose_translates_along_agent_forward_axis(self) -> None:
        action = MoveForward(agent_id=self.agent_id, distance=0.2)

        goal_pose = self.policy._compute_goal_pose_from_action(self.state, action)

        assert goal_pose is not None
        goal_pos, goal_quat = goal_pose
        nptest.assert_allclose(goal_pos, np.array([1.0, 2.0, 3.2]))
        nptest.assert_allclose(qt.as_float_array(goal_quat), qt.as_float_array(qt.one))

    def test_move_tangentially_goal_pose_normalizes_direction(self) -> None:
        action = MoveTangentially(
            agent_id=self.agent_id,
            distance=0.3,
            direction=(3.0, 4.0, 0.0),
        )

        goal_pose = self.policy._compute_goal_pose_from_action(self.state, action)

        assert goal_pose is not None
        goal_pos, goal_quat = goal_pose
        nptest.assert_allclose(goal_pos, np.array([1.18, 2.24, 3.0]), atol=1e-7)
        nptest.assert_allclose(qt.as_float_array(goal_quat), qt.as_float_array(qt.one))

    def test_orient_horizontal_goal_pose_updates_rotation_and_translation(self) -> None:
        action = OrientHorizontal(
            agent_id=self.agent_id,
            rotation_degrees=90.0,
            left_distance=0.1,
            forward_distance=0.2,
        )

        goal_pose = self.policy._compute_goal_pose_from_action(self.state, action)

        assert goal_pose is not None
        goal_pos, goal_quat = goal_pose
        expected_rot = rot.from_rotvec([0.0, 0.0, np.radians(90.0)])
        expected_quat_xyzw = expected_rot.as_quat()
        expected_quat = qt.quaternion(
            expected_quat_xyzw[3],
            expected_quat_xyzw[0],
            expected_quat_xyzw[1],
            expected_quat_xyzw[2],
        )
        expected_delta = expected_rot.apply(np.array([-1.0, 0.0, 0.0])) * 0.1
        expected_delta += expected_rot.apply(np.array([0.0, 0.0, 1.0])) * 0.2

        nptest.assert_allclose(goal_pos, np.array([1.0, 2.0, 3.0]) + expected_delta)
        nptest.assert_allclose(
            qt.as_float_array(goal_quat),
            qt.as_float_array(expected_quat),
            atol=1e-7,
        )

    def test_unknown_action_returns_none(self) -> None:
        action = LookUp(agent_id=self.agent_id, rotation_degrees=10.0)

        goal_pose = self.policy._compute_goal_pose_from_action(self.state, action)

        self.assertIsNone(goal_pose)


if __name__ == "__main__":
    unittest.main()
