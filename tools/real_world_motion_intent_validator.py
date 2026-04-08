# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import argparse
import ast
import collections
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

RW_MOTION_PATTERN = re.compile(r"RW_MOTION\s+([A-Z_]+)\s*\|\s*(\{.*\})")
ADAPTER_PATTERN = re.compile(
    r"(?:REAL_WORLD_ADAPTER|RW_ADAPTER)\s+([A-Z_]+)\s*\|\s*(\{.*\})"
)
INTERFACE_PATTERN = re.compile(
    r"REAL_WORLD_INTERFACE\s+STEP\s*\|\s*.*use_goal_pose_dispatch=(True|False)",
)

ALL_ACTION_TYPES = {
    "MoveForward",
    "MoveTangentially",
    "OrientHorizontal",
    "OrientVertical",
}


def _safe_dict(payload: str) -> dict[str, Any] | None:
    try:
        value = ast.literal_eval(payload)
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, dict):
        return value
    return None


def _coerce_log_files(log_file_or_files: Path | list[Path]) -> list[Path]:
    if isinstance(log_file_or_files, Path):
        return [log_file_or_files]
    return list(log_file_or_files)


def parse_log_events(log_file_or_files: Path | list[Path]) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = {
        "rw_motion": [],
        "adapter": [],
        "interface": [],
    }
    for source_file in _coerce_log_files(log_file_or_files):
        for raw_line in source_file.read_text(
            encoding="utf-8",
            errors="replace",
        ).splitlines():
            rw_match = RW_MOTION_PATTERN.search(raw_line)
            if rw_match is not None:
                event_name = rw_match.group(1)
                payload = _safe_dict(rw_match.group(2))
                if payload is not None:
                    events["rw_motion"].append(
                        {
                            "event": event_name,
                            "payload": payload,
                            "source": str(source_file),
                        }
                    )
                continue

            adapter_match = ADAPTER_PATTERN.search(raw_line)
            if adapter_match is not None:
                event_name = adapter_match.group(1)
                payload = _safe_dict(adapter_match.group(2))
                if payload is not None:
                    events["adapter"].append(
                        {
                            "event": event_name,
                            "payload": payload,
                            "source": str(source_file),
                        }
                    )
                continue

            interface_match = INTERFACE_PATTERN.search(raw_line)
            if interface_match is not None:
                dispatch_value = interface_match.group(1) == "True"
                events["interface"].append(
                    {
                        "use_goal_pose_dispatch": dispatch_value,
                        "source": str(source_file),
                    }
                )

    return events


def _select_log_files(run_dir: Path | None, log_file: Path | None) -> list[Path]:
    if log_file is not None:
        return [log_file]
    if run_dir is None:
        raise ValueError("Provide --log-file or --run-dir")

    selected: list[Path] = []
    run_dir_log = run_dir / "log.txt"
    if run_dir_log.exists():
        selected.append(run_dir_log)

    repo_root = Path(__file__).resolve().parents[1]
    outputs_root = repo_root / "outputs"
    if outputs_root.exists():
        reference_path = run_dir_log if run_dir_log.exists() else run_dir
        reference_mtime = reference_path.stat().st_mtime
        run_logs = list(outputs_root.glob("**/run.log"))
        if run_logs:
            closest = min(
                run_logs,
                key=lambda candidate: abs(candidate.stat().st_mtime - reference_mtime),
            )
            if closest.exists() and closest not in selected:
                selected.append(closest)

    if not selected:
        raise ValueError(f"No log files found for run directory: {run_dir}")
    return selected


def _vector(payload: dict[str, Any], key: str) -> np.ndarray | None:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != 3:
        return None
    vector = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(vector)):
        return None
    return vector


def _compute_step_metrics(
    intended_delta_m: np.ndarray,
    observed_delta_m: np.ndarray,
) -> dict[str, float]:
    intended_norm = float(np.linalg.norm(intended_delta_m))
    observed_norm = float(np.linalg.norm(observed_delta_m))

    if intended_norm == 0.0 or observed_norm == 0.0:
        return {
            "intended_norm_m": intended_norm,
            "observed_norm_m": observed_norm,
            "direction_cosine": 0.0,
            "angle_deg": 180.0,
            "norm_ratio": 0.0,
            "orthogonal_drift_mm": observed_norm * 1000.0,
        }

    cosine = float(np.dot(intended_delta_m, observed_delta_m) / (intended_norm * observed_norm))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cosine)))
    norm_ratio = observed_norm / intended_norm

    intended_unit = intended_delta_m / intended_norm
    parallel = float(np.dot(observed_delta_m, intended_unit)) * intended_unit
    orthogonal = observed_delta_m - parallel

    return {
        "intended_norm_m": intended_norm,
        "observed_norm_m": observed_norm,
        "direction_cosine": cosine,
        "angle_deg": angle_deg,
        "norm_ratio": float(norm_ratio),
        "orthogonal_drift_mm": float(np.linalg.norm(orthogonal) * 1000.0),
    }


def analyze_motion_intent(
    events: dict[str, list[dict[str, Any]]],
    *,
    action_type: str = "ALL",
    min_cosine: float = 0.9,
    max_angle_deg: float = 25.0,
    min_norm_ratio: float = 0.6,
    max_norm_ratio: float = 1.4,
    max_orthogonal_mm: float = 3.0,
) -> dict[str, Any]:
    rw_motion = events.get("rw_motion", [])
    adapter_events = events.get("adapter", [])
    interface_events = events.get("interface", [])

    if action_type == "ALL":
        tracked_actions = set(ALL_ACTION_TYPES)
    else:
        tracked_actions = {action_type}

    intent_events: list[dict[str, Any]] = []
    observed_by_step: dict[int, dict[str, Any]] = {}
    pending_step_index: int | None = None
    sequential_step_counter = 0

    for item in rw_motion:
        event = item.get("event")
        payload = item.get("payload", {})
        if event == "STEP_BEGIN":
            raw_step_index = payload.get("step_index")
            pending_step_index = int(raw_step_index) if isinstance(raw_step_index, int) else None
        if event == "RELATIVE_ACTION_GOAL" and payload.get("action_type") in tracked_actions:
            step_index = payload.get("step_index")
            if not isinstance(step_index, int):
                if pending_step_index is not None:
                    step_index = pending_step_index
                else:
                    step_index = sequential_step_counter
                    sequential_step_counter += 1
            intent_events.append(
                {
                    "step_index": int(step_index),
                    "action_type": payload.get("action_type"),
                    "payload": payload,
                }
            )
        if event == "STEP_DELTA":
            raw_step_index = payload.get("step_index")
            if isinstance(raw_step_index, int):
                observed_by_step[int(raw_step_index)] = payload

    paired_intents: list[dict[str, Any]] = []
    for intent in intent_events:
        step_index = int(intent["step_index"])
        observed_payload = observed_by_step.get(step_index)
        if observed_payload is None:
            continue
        paired_intents.append(
            {
                "step_index": step_index,
                "action_type": intent["action_type"],
                "intent_payload": intent["payload"],
                "observed_payload": observed_payload,
            }
        )

    paired_count = len(paired_intents)
    per_step: list[dict[str, Any]] = []
    pass_count = 0
    failure_reason_counts: collections.Counter[str] = collections.Counter()
    action_summary: dict[str, dict[str, int | float]] = {
        name: {"paired": 0, "passed": 0, "pass_rate": 0.0}
        for name in sorted(tracked_actions)
    }

    for index, pair in enumerate(paired_intents):
        action_name = str(pair["action_type"])
        intended = _vector(pair["intent_payload"], "delta_m")
        observed = _vector(pair["observed_payload"], "delta_position_m")
        if intended is None or observed is None:
            failure_reason_counts["missing_vectors"] += 1
            action_summary[action_name]["paired"] = int(action_summary[action_name]["paired"]) + 1
            per_step.append(
                {
                    "index": index,
                    "step_index": pair["step_index"],
                    "action_type": action_name,
                    "status": "missing_vectors",
                    "passed": False,
                    "failed_checks": ["missing_vectors"],
                }
            )
            continue

        metrics = _compute_step_metrics(intended, observed)
        checks = {
            "cosine": metrics["direction_cosine"] >= min_cosine,
            "angle": metrics["angle_deg"] <= max_angle_deg,
            "norm_ratio": min_norm_ratio <= metrics["norm_ratio"] <= max_norm_ratio,
            "orthogonal_drift": metrics["orthogonal_drift_mm"] <= max_orthogonal_mm,
        }
        failed_checks = [name for name, is_ok in checks.items() if not is_ok]
        passed = (
            not failed_checks
        )
        if passed:
            pass_count += 1
            action_summary[action_name]["passed"] = int(action_summary[action_name]["passed"]) + 1
        else:
            for failed_check in failed_checks:
                failure_reason_counts[failed_check] += 1

        action_summary[action_name]["paired"] = int(action_summary[action_name]["paired"]) + 1

        per_step.append(
            {
                "index": index,
                "step_index": pair["step_index"],
                "action_type": action_name,
                "status": "ok",
                "passed": passed,
                "failed_checks": failed_checks,
                "check_results": checks,
                **metrics,
            }
        )

    for action_name, summary in action_summary.items():
        paired = int(summary["paired"])
        passed = int(summary["passed"])
        summary["pass_rate"] = float(passed / paired) if paired > 0 else 0.0

    transformed_goal_count = sum(
        1 for event in adapter_events if event.get("event") == "TRANSFORMED_ROBOT_GOAL"
    )
    dispatched_count = sum(
        1 for event in adapter_events if event.get("event") == "COMMAND_DISPATCHED"
    )

    dispatch_mode_values = [
        bool(item.get("use_goal_pose_dispatch")) for item in interface_events
    ]
    dispatch_required = any(dispatch_mode_values)

    report = {
        "action_type": action_type,
        "paired_steps": paired_count,
        "intent_events": len(intent_events),
        "observed_events": len(observed_by_step),
        "adapter_transformed_goals": transformed_goal_count,
        "adapter_dispatched_commands": dispatched_count,
        "pass_count": pass_count,
        "pass_rate": float(pass_count / paired_count) if paired_count > 0 else 0.0,
        "dispatch_required": dispatch_required,
        "per_action": action_summary,
        "failure_reason_counts": dict(sorted(failure_reason_counts.items())),
        "thresholds": {
            "min_cosine": min_cosine,
            "max_angle_deg": max_angle_deg,
            "min_norm_ratio": min_norm_ratio,
            "max_norm_ratio": max_norm_ratio,
            "max_orthogonal_mm": max_orthogonal_mm,
        },
        "steps": per_step,
    }
    adapter_ok = (
        not dispatch_required
        or (
            transformed_goal_count >= paired_count
            and dispatched_count >= paired_count
        )
    )
    report["passed"] = bool(
        paired_count > 0
        and pass_count == paired_count
        and adapter_ok
    )
    run_failed_checks: list[str] = []
    if paired_count == 0:
        run_failed_checks.append("no_paired_steps")
    if dispatch_required and transformed_goal_count < paired_count:
        run_failed_checks.append("missing_transformed_robot_goal")
    if dispatch_required and dispatched_count < paired_count:
        run_failed_checks.append("missing_command_dispatched")
    report["run_failed_checks"] = run_failed_checks
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate real-world motion intent vs observed robot movement from logs."
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--log-file", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--action-type", type=str, default="ALL")
    parser.add_argument("--min-cosine", type=float, default=0.9)
    parser.add_argument("--max-angle-deg", type=float, default=25.0)
    parser.add_argument("--min-norm-ratio", type=float, default=0.6)
    parser.add_argument("--max-norm-ratio", type=float, default=1.4)
    parser.add_argument("--max-orthogonal-mm", type=float, default=3.0)
    parser.add_argument("--fail-on-threshold", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    if args.log_file is None and args.run_dir is None:
        raise SystemExit("Provide --log-file or --run-dir")

    try:
        selected_logs = _select_log_files(args.run_dir, args.log_file)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    events = parse_log_events(selected_logs)
    report = analyze_motion_intent(
        events,
        action_type=args.action_type,
        min_cosine=args.min_cosine,
        max_angle_deg=args.max_angle_deg,
        min_norm_ratio=args.min_norm_ratio,
        max_norm_ratio=args.max_norm_ratio,
        max_orthogonal_mm=args.max_orthogonal_mm,
    )
    report["log_sources"] = [str(path) for path in selected_logs]

    print(
        "Motion intent validation "
        f"{'PASSED' if report['passed'] else 'FAILED'} | "
        f"paired_steps={report['paired_steps']} pass_rate={report['pass_rate']:.2f}"
    )

    json_out = args.json_out
    if json_out is None and args.run_dir is not None:
        json_out = args.run_dir / "motion_intent_report.json"
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote report: {json_out}")

    if args.fail_on_threshold and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
