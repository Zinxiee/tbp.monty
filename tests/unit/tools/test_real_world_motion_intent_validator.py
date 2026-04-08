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
import tempfile
import unittest
from pathlib import Path


def _load_validator_module():
    root = Path(__file__).resolve().parents[3]
    module_path = root / "tools" / "real_world_motion_intent_validator.py"
    module_spec = importlib.util.spec_from_file_location(
        "real_world_motion_intent_validator",
        module_path,
    )
    module = importlib.util.module_from_spec(module_spec)
    assert module_spec is not None and module_spec.loader is not None
    module_spec.loader.exec_module(module)
    return module


_VALIDATOR_MODULE = _load_validator_module()
analyze_motion_intent = _VALIDATOR_MODULE.analyze_motion_intent
parse_log_events = _VALIDATOR_MODULE.parse_log_events


class MotionIntentValidatorTest(unittest.TestCase):
    def _write_log(self, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "log.txt"
        path.write_text(content, encoding="utf-8")
        return path

    def _write_named_log(self, name: str, content: str) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_analyze_motion_intent_passes_when_vectors_align(self) -> None:
        log_path = self._write_log(
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=False has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01], 'goal_position_m': [0.0, 0.0, -0.01]}",
                    "INFO REAL_WORLD_ADAPTER TRANSFORMED_ROBOT_GOAL | {'robot_position_m': [0.34, 0.01, 0.29]}",
                    "INFO REAL_WORLD_ADAPTER COMMAND_DISPATCHED | {'command_mm': [340.0, 10.0, 290.0]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, -0.009]}",
                ]
            )
        )

        report = analyze_motion_intent(parse_log_events(log_path))

        self.assertTrue(report["passed"])
        self.assertEqual(report["paired_steps"], 1)
        self.assertEqual(report["pass_count"], 1)
        self.assertEqual(report["failure_reason_counts"], {})
        self.assertEqual(report["steps"][0]["failed_checks"], [])
        self.assertIn("MoveForward", report["per_action"])
        self.assertEqual(report["per_action"]["MoveForward"]["paired"], 1)

    def test_analyze_motion_intent_fails_when_direction_mismatches(self) -> None:
        log_path = self._write_log(
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=False has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01]}",
                    "INFO REAL_WORLD_ADAPTER TRANSFORMED_ROBOT_GOAL | {'robot_position_m': [0.34, 0.01, 0.29]}",
                    "INFO REAL_WORLD_ADAPTER COMMAND_DISPATCHED | {'command_mm': [340.0, 10.0, 290.0]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, 0.009]}",
                ]
            )
        )

        report = analyze_motion_intent(parse_log_events(log_path))

        self.assertFalse(report["passed"])
        self.assertEqual(report["paired_steps"], 1)
        self.assertEqual(report["pass_count"], 0)
        self.assertIn("cosine", report["steps"][0]["failed_checks"])
        self.assertIn("angle", report["steps"][0]["failed_checks"])
        self.assertEqual(report["failure_reason_counts"].get("cosine"), 1)
        self.assertEqual(report["failure_reason_counts"].get("angle"), 1)

    def test_parser_accepts_legacy_rw_adapter_prefix(self) -> None:
        log_path = self._write_log(
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=False has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01]}",
                    "INFO RW_ADAPTER TRANSFORMED_ROBOT_GOAL | {'robot_position_m': [0.34, 0.01, 0.29]}",
                    "INFO RW_ADAPTER COMMAND_DISPATCHED | {'command_mm': [340.0, 10.0, 290.0]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, -0.0095]}",
                ]
            )
        )

        events = parse_log_events(log_path)
        report = analyze_motion_intent(events)

        self.assertEqual(len(events["adapter"]), 2)
        self.assertTrue(report["passed"])

    def test_report_does_not_require_adapter_when_dispatch_disabled(self) -> None:
        log_path = self._write_log(
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=False has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, -0.0095]}",
                ]
            )
        )

        report = analyze_motion_intent(parse_log_events(log_path))

        self.assertTrue(report["passed"])
        self.assertEqual(report["run_failed_checks"], [])

    def test_report_requires_adapter_when_dispatch_enabled(self) -> None:
        log_path = self._write_log(
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=True has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, -0.0095]}",
                ]
            )
        )

        report = analyze_motion_intent(parse_log_events(log_path))

        self.assertFalse(report["passed"])
        self.assertIn("missing_transformed_robot_goal", report["run_failed_checks"])
        self.assertIn("missing_command_dispatched", report["run_failed_checks"])

    def test_parse_log_events_merges_multiple_sources(self) -> None:
        rw_log = self._write_named_log(
            "log.txt",
            "\n".join(
                [
                    "INFO REAL_WORLD_INTERFACE STEP | use_goal_pose_dispatch=True has_policy_result=False has_goal_pose=False actions=[]/n",
                    "INFO RW_MOTION STEP_BEGIN | {'step_index': 0, 'selected_actions': ['MoveForward']}",
                    "INFO RW_MOTION RELATIVE_ACTION_GOAL | {'action_type': 'MoveForward', 'delta_m': [0.0, 0.0, -0.01]}",
                    "INFO RW_MOTION STEP_DELTA | {'step_index': 0, 'delta_position_m': [0.0, 0.0, -0.0095]}",
                ]
            ),
        )
        adapter_log = self._write_named_log(
            "run.log",
            "\n".join(
                [
                    "INFO REAL_WORLD_ADAPTER TRANSFORMED_ROBOT_GOAL | {'robot_position_m': [0.34, 0.01, 0.29]}",
                    "INFO REAL_WORLD_ADAPTER COMMAND_DISPATCHED | {'command_mm': [340.0, 10.0, 290.0]}",
                ]
            ),
        )

        events = parse_log_events([rw_log, adapter_log])
        report = analyze_motion_intent(events)

        self.assertEqual(len(events["adapter"]), 2)
        self.assertEqual(report["run_failed_checks"], [])
        self.assertTrue(report["passed"])


if __name__ == "__main__":
    unittest.main()
