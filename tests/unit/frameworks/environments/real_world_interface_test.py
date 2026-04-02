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
from unittest.mock import sentinel

import quaternion as qt

from tbp.monty.frameworks.actions.actions import LookUp
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.real_world_interface import (
    RealWorldEnvironmentInterface,
)
from tbp.monty.frameworks.experiments.mode import ExperimentMode
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID


class _FakeEnv:
    def __init__(self) -> None:
        self.last_result = None
        self.last_result_history = []
        self.step_calls = []

    def reset(self):
        return _obs_state_pair()

    def step(self, actions):
        self.step_calls.append(actions)
        return _obs_state_pair()

    def set_last_motor_policy_result(self, result):
        self.last_result = result
        self.last_result_history.append(result)


class _FakeEnvWithoutSetter:
    def reset(self):
        return _obs_state_pair()

    def step(self, actions):
        return _obs_state_pair()


def _obs_state_pair():
    return (
        Observations(
            {
                AgentID("agent_id_0"): AgentObservations(
                    {
                        SensorID("patch"): SensorObservation(
                            {
                                "depth": [[0.1]],
                            }
                        )
                    }
                )
            }
        ),
        ProprioceptiveState(
            {
                AgentID("agent_id_0"): AgentState(
                    sensors={
                        SensorID("patch"): SensorState(
                            position=(0.0, 0.0, 0.0),
                            rotation=qt.one,
                        )
                    },
                    position=(0.0, 0.0, 0.0),
                    rotation=qt.one,
                )
            }
        ),
    )


class RealWorldEnvironmentInterfaceTest(unittest.TestCase):
    def test_exposes_primary_target_for_experiment_contract(self) -> None:
        env = _FakeEnv()
        interface = RealWorldEnvironmentInterface(
            env=env,
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=True,
        )

        self.assertEqual(interface.primary_target["object"], "real_world_target")
        self.assertEqual(interface.primary_target["semantic_id"], 0)
        self.assertEqual(interface.semantic_id_to_label[0], "real_world_target")

    def test_step_passes_last_motor_policy_result_when_enabled(self) -> None:
        env = _FakeEnv()
        interface = RealWorldEnvironmentInterface(
            env=env,
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=True,
        )
        interface.set_last_motor_policy_result(sentinel.result)

        interface.step([])

        self.assertIs(env.last_result, sentinel.result)

    def test_step_clears_last_motor_policy_result_when_disabled(self) -> None:
        env = _FakeEnv()
        interface = RealWorldEnvironmentInterface(
            env=env,
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=False,
        )
        interface.set_last_motor_policy_result(sentinel.result)

        interface.step([LookUp(agent_id=AgentID("agent_id_0"), rotation_degrees=1.0)])

        self.assertIsNone(env.last_result)

    def test_step_noops_when_env_lacks_setter(self) -> None:
        interface = RealWorldEnvironmentInterface(
            env=_FakeEnvWithoutSetter(),
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=True,
        )
        interface.set_last_motor_policy_result(sentinel.result)

        interface.step([])

    def test_step_clears_interface_result_after_single_step(self) -> None:
        env = _FakeEnv()
        interface = RealWorldEnvironmentInterface(
            env=env,
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=True,
        )
        interface.set_last_motor_policy_result(sentinel.result)

        interface.step([])

        self.assertIsNone(interface._last_motor_policy_result)

    def test_step_does_not_reuse_stale_result_on_next_step(self) -> None:
        env = _FakeEnv()
        interface = RealWorldEnvironmentInterface(
            env=env,
            rng=sentinel.rng,
            seed=42,
            experiment_mode=ExperimentMode.TRAIN,
            transform=None,
            use_goal_pose_dispatch=True,
        )
        interface.set_last_motor_policy_result(sentinel.result)

        interface.step([])
        interface.step([])

        self.assertEqual(env.last_result_history, [sentinel.result, None])


if __name__ == "__main__":
    unittest.main()
