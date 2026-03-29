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
from unittest.mock import MagicMock, sentinel

from tbp.monty.frameworks.experiments.object_recognition_experiments import (
    MontyObjectRecognitionExperiment,
)


class MontyObjectRecognitionExperimentPrivateTest(unittest.TestCase):
    def test_pass_last_motor_policy_result_to_env_interface_calls_optional_setter(
        self,
    ) -> None:
        experiment = MontyObjectRecognitionExperiment.__new__(
            MontyObjectRecognitionExperiment
        )
        experiment.model = MagicMock()
        experiment.model.last_motor_policy_result = sentinel.result
        experiment.env_interface = MagicMock()

        experiment._pass_last_motor_policy_result_to_env_interface()

        experiment.env_interface.set_last_motor_policy_result.assert_called_once_with(
            sentinel.result
        )

    def test_pass_last_motor_policy_result_to_env_interface_is_noop_without_setter(
        self,
    ) -> None:
        experiment = MontyObjectRecognitionExperiment.__new__(
            MontyObjectRecognitionExperiment
        )
        experiment.model = MagicMock()
        experiment.model.last_motor_policy_result = sentinel.result

        class EnvInterfaceWithoutSetter:
            pass

        experiment.env_interface = EnvInterfaceWithoutSetter()

        experiment._pass_last_motor_policy_result_to_env_interface()


if __name__ == "__main__":
    unittest.main()
