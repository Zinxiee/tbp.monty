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
from pathlib import Path

from omegaconf import OmegaConf


class RealWorldLite6ProbeConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[4]
        self.no_goal_dispatch_cfg = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/"
            "lite6_maixsense_unsupervised_probe_no_goal_dispatch.yaml"
        )
        self.probe_baseline_cfg = self.repo_root / (
            "src/tbp/monty/conf/experiment/real_world/"
            "lite6_maixsense_unsupervised_probe_baseline.yaml"
        )

    def test_no_goal_dispatch_disables_env_dispatch(self) -> None:
        cfg = OmegaConf.load(self.no_goal_dispatch_cfg)

        self.assertFalse(cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch)
        self.assertFalse(cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch)
        self.assertEqual(
            cfg.experiment.config.logging.run_name,
            "real_world_lite6_maixsense_unsupervised_probe_no_goal_dispatch",
        )

    def test_probe_baseline_enables_probe_motion_settings(self) -> None:
        cfg = OmegaConf.load(self.probe_baseline_cfg)
        env_args = cfg.experiment.config.environment.env_init_args

        self.assertTrue(cfg.experiment.config.train_env_interface_args.use_goal_pose_dispatch)
        self.assertTrue(cfg.experiment.config.eval_env_interface_args.use_goal_pose_dispatch)
        self.assertFalse(env_args.require_object_swap_confirmation)
        self.assertTrue(env_args.motion_debug_logging)
        self.assertTrue(env_args.probe_move_forward_only)
        self.assertEqual(env_args.probe_move_forward_distance_m, 0.01)
        self.assertEqual(env_args.probe_max_steps, 6)
        self.assertTrue(env_args.goal_adapter_config.debug_logging)
        self.assertEqual(
            cfg.experiment.config.logging.run_name,
            "real_world_lite6_maixsense_unsupervised_probe_baseline",
        )


if __name__ == "__main__":
    unittest.main()
