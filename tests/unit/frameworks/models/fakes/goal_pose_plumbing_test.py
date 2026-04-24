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

import numpy as np
import quaternion as qt

from tbp.monty.context import RuntimeContext
from tbp.monty.frameworks.actions.action_samplers import UniformlyDistributedSampler
from tbp.monty.frameworks.actions.actions import LookUp
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import Observations
from tbp.monty.frameworks.models.motor_policies import BasePolicy
from tbp.monty.frameworks.models.motor_policy_selectors import SinglePolicySelector
from tbp.monty.frameworks.models.motor_system import MotorSystem
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID
from tests.unit.frameworks.models.fakes.cmp import FakeMessage


class MotorSystemGoalPosePlumbingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.agent_id = AgentID("agent_id_0")
        self.sensor_id = SensorID("patch")
        policy = BasePolicy(
            action_sampler=UniformlyDistributedSampler(actions=[LookUp]),
            agent_id=self.agent_id,
        )
        self.motor_system = MotorSystem(policy_selector=SinglePolicySelector(policy))

    def test_last_policy_result_is_populated_without_changing_return_actions(self) -> None:
        proprioceptive_state = ProprioceptiveState(
            {
                self.agent_id: AgentState(
                    sensors={
                        self.sensor_id: SensorState(
                            position=(0.0, 0.0, 0.0),
                            rotation=qt.one,
                        )
                    },
                    position=(0.0, 0.0, 0.0),
                    rotation=qt.one,
                )
            }
        )

        actions = self.motor_system(
            ctx=RuntimeContext(rng=np.random.RandomState(0)),
            observations=Observations(),
            proprioceptive_state=proprioceptive_state,
            percept=FakeMessage(),
            goals=[],
        )

        self.assertEqual(len(actions), 1)
        self.assertIsNotNone(self.motor_system.last_policy_result)
        assert self.motor_system.last_policy_result is not None
        self.assertEqual(actions, self.motor_system.last_policy_result.actions)


if __name__ == "__main__":
    unittest.main()
