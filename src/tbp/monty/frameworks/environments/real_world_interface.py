# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

from typing import Sequence

from tbp.monty.frameworks.actions.actions import Action
from tbp.monty.frameworks.environments.embodied_data import EnvironmentInterface
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.models.abstract_monty_classes import Observations

__all__ = ["RealWorldEnvironmentInterface"]


class RealWorldEnvironmentInterface(EnvironmentInterface):
    """EnvironmentInterface for strict-blocking real-world operation.

    This interface accepts optional per-step MotorPolicyResult metadata from the
    experiment loop so goal-pose dispatch can be preferred over relative actions.
    """

    def __init__(
        self,
        *args,
        use_goal_pose_dispatch: bool = True,
        **kwargs,
    ):
        self.use_goal_pose_dispatch = use_goal_pose_dispatch
        self._last_motor_policy_result: MotorPolicyResult | None = None
        super().__init__(*args, **kwargs)

    def set_last_motor_policy_result(
        self,
        result: MotorPolicyResult | None,
    ) -> None:
        self._last_motor_policy_result = result

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[Observations, ProprioceptiveState]:
        if hasattr(self.env, "set_last_motor_policy_result"):
            if self.use_goal_pose_dispatch:
                self.env.set_last_motor_policy_result(self._last_motor_policy_result)
            else:
                self.env.set_last_motor_policy_result(None)

        observations, state = super().step(actions)
        self._last_motor_policy_result = None
        return observations, state
