# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import inspect
import logging
import time
from typing import Any, Callable, Sequence

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.actions.actions import (
    Action,
    MoveForward,
    MoveTangentially,
    OrientHorizontal,
    OrientVertical,
    SetAgentPose,
    SetSensorPose,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.abstract_monty_classes import (
    AgentObservations,
    Observations,
    SensorObservation,
)
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import (
    AgentState,
    ProprioceptiveState,
    SensorState,
)
from tbp.monty.frameworks.sensors import SensorID

__all__ = ["RealWorldLite6A010Environment", "RealWorldSafetyStopError"]

logger = logging.getLogger(__name__)


class RealWorldSafetyStopError(RuntimeError):
    """Raised when hardware safety checks reject an attempted movement."""

    def __init__(self, reason_code: str, details: str) -> None:
        self.reason_code = reason_code
        self.details = details
        super().__init__(f"{reason_code}: {details}")


class RealWorldLite6A010Environment:
    """Hardware-backed environment for Lite6 + Maixsense A010 bring-up.

    This environment is intentionally strict for v1 deployment:
    - blocking movement loop (move -> settle -> sense)
    - world frame aligned to robot base frame
    - hard-stop behavior with explicit reason codes on failures
    """

    def __init__(
        self,
        robot_interface: Any,
        sensor_client: Any,
        observation_adapter: Any,
        goal_adapter: Any | None = None,
        agent_id: str = "agent_id_0",
        sensor_id: str = "patch",
        home_pose_mm_deg: Sequence[float] | None = None,
        settle_time_s: float = 0.25,
        home_reset_timeout_s: float = 4.0,
        home_reset_poll_s: float = 0.02,
        home_reset_position_tolerance_mm: float = 20.0,
        home_reset_orientation_tolerance_deg: float = 10.0,
        require_object_swap_confirmation: bool = True,
        object_swap_prompt: str = "Swap/position the object, then press Enter to continue: ",
        input_fn: Callable[[str], str] | None = None,
        sensor_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
        sensor_rotation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        depth_unit: int = 0,
        sensor_frame_timeout_s: float = 1.0,
        sensor_frame_max_retries: int = 2,
        sensor_frame_retry_delay_s: float = 0.05,
        depth_burst_n: int = 5,
        goal_rejection_hard_stop: bool = False,
        settle_use_goal_convergence_gate: bool = False,
        settle_convergence_timeout_s: float | None = None,
        settle_convergence_position_tolerance_mm: float | None = None,
        settle_convergence_required_consecutive_samples: int = 2,
        settle_convergence_poll_s: float = 0.02,
        goal_adapter_config: dict[str, Any] | None = None,
        motion_debug_logging: bool = False,
        probe_move_forward_only: bool = False,
        probe_move_forward_distance_m: float = 0.01,
        probe_max_steps: int = 5,
        sensed_orientation_sequence: str = "xyz",
        sensed_orientation_degrees: bool = False,
        sensed_orientation_intrinsic: bool = False,
        sensed_orientation_index_order: Sequence[int] = (0, 1, 2),
        sensed_orientation_signs: Sequence[float] = (1.0, 1.0, 1.0),
        sensed_orientation_offset_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        sensed_motion_offset_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        min_sensor_height_m: float | None = None,
    ) -> None:
        self.robot_interface = robot_interface
        self.sensor_client = sensor_client
        self.observation_adapter = observation_adapter

        # Instantiate goal_adapter from config if not provided as object.
        if goal_adapter is None and goal_adapter_config is not None:
            try:
                from multimodal_monty_meets_world.factory import create_goal_adapter

                goal_adapter = create_goal_adapter(
                    robot=robot_interface, **goal_adapter_config
                )
            except ImportError as e:
                logger.warning(
                    f"Failed to instantiate goal_adapter from config: {e}. Goal dispatch disabled."
                )

        self.goal_adapter = goal_adapter
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.home_pose_mm_deg = (
            tuple(home_pose_mm_deg) if home_pose_mm_deg is not None else None
        )
        self.settle_time_s = settle_time_s
        self.home_reset_timeout_s = max(0.0, float(home_reset_timeout_s))
        self.home_reset_poll_s = max(0.0, float(home_reset_poll_s))
        self.home_reset_position_tolerance_mm = max(
            0.0, float(home_reset_position_tolerance_mm)
        )
        self.home_reset_orientation_tolerance_deg = max(
            0.0, float(home_reset_orientation_tolerance_deg)
        )
        self.require_object_swap_confirmation = require_object_swap_confirmation
        self.object_swap_prompt = object_swap_prompt
        self.input_fn = input_fn or input
        self.sensor_translation_m = np.asarray(sensor_translation_m, dtype=float)
        self.sensor_rotation_wxyz = _as_wxyz_quaternion(sensor_rotation_wxyz)
        self.depth_unit = depth_unit
        self.sensor_frame_timeout_s = float(sensor_frame_timeout_s)
        self.sensor_frame_max_retries = max(0, int(sensor_frame_max_retries))
        self.sensor_frame_retry_delay_s = max(0.0, float(sensor_frame_retry_delay_s))
        self.depth_burst_n = max(1, int(depth_burst_n))
        self.goal_rejection_hard_stop = bool(goal_rejection_hard_stop)
        self.settle_use_goal_convergence_gate = bool(settle_use_goal_convergence_gate)
        self.settle_convergence_timeout_s = (
            None
            if settle_convergence_timeout_s is None
            else max(0.0, float(settle_convergence_timeout_s))
        )
        self.settle_convergence_position_tolerance_mm = (
            None
            if settle_convergence_position_tolerance_mm is None
            else max(0.0, float(settle_convergence_position_tolerance_mm))
        )
        self.settle_convergence_required_consecutive_samples = max(
            1, int(settle_convergence_required_consecutive_samples)
        )
        self.settle_convergence_poll_s = max(0.0, float(settle_convergence_poll_s))
        self.motion_debug_logging = bool(motion_debug_logging)
        self.probe_move_forward_only = bool(probe_move_forward_only)
        self.probe_move_forward_distance_m = max(
            0.0, float(probe_move_forward_distance_m)
        )
        self.probe_max_steps = max(0, int(probe_max_steps))
        self.sensed_orientation_sequence = str(sensed_orientation_sequence)
        self.sensed_orientation_degrees = bool(sensed_orientation_degrees)
        self.sensed_orientation_intrinsic = bool(sensed_orientation_intrinsic)
        self.sensed_orientation_index_order = tuple(
            int(index) for index in sensed_orientation_index_order
        )
        self.sensed_orientation_signs = np.asarray(
            sensed_orientation_signs, dtype=float
        )
        self.sensed_orientation_offset_wxyz = _as_wxyz_quaternion(
            sensed_orientation_offset_wxyz
        )
        self.sensed_motion_offset_wxyz = _as_wxyz_quaternion(sensed_motion_offset_wxyz)
        self.min_sensor_height_m = (
            None if min_sensor_height_m is None else float(min_sensor_height_m)
        )
        self._probe_step_count = 0
        self._last_home_command_time_s: float | None = None

        if len(self.sensed_orientation_sequence) != 3:
            raise ValueError("sensed_orientation_sequence must contain exactly 3 axes")
        if len(self.sensed_orientation_index_order) != 3:
            raise ValueError(
                "sensed_orientation_index_order must contain exactly 3 indices"
            )
        if self.sensed_orientation_signs.shape != (3,):
            raise ValueError("sensed_orientation_signs must contain exactly 3 values")

        self._last_motor_policy_result: MotorPolicyResult | None = None
        self.last_safety_event: dict[str, str] | None = None
        self._step_dispatched_command = False

    def set_last_motor_policy_result(
        self,
        result: MotorPolicyResult | None,
    ) -> None:
        self._last_motor_policy_result = result

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._probe_step_count = 0
        self._move_home_if_configured()
        if self.require_object_swap_confirmation:
            self.input_fn(self.object_swap_prompt)
        return self._observe()

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[Observations, ProprioceptiveState]:
        actions_to_execute = self._select_actions_for_step(actions)
        self._step_dispatched_command = False
        pre_position_m, _ = self._get_agent_pose_world()
        self._log_motion_debug(
            "STEP_BEGIN",
            step_index=self._probe_step_count,
            input_actions=[type(action).__name__ for action in actions],
            selected_actions=[type(action).__name__ for action in actions_to_execute],
            use_goal_pose_dispatch=self._last_motor_policy_result is not None,
            has_goal_pose=bool(
                self._last_motor_policy_result is not None
                and self._last_motor_policy_result.goal_pose is not None
            ),
            pre_position_m=np.round(pre_position_m, 6).tolist(),
        )

        if self._last_motor_policy_result is not None:
            if self._last_motor_policy_result.goal_pose is not None:
                accepted = self._dispatch_goal_pose(
                    self._last_motor_policy_result,
                    actions_to_execute,
                )
                if not accepted:
                    self._execute_actions(actions_to_execute)
            else:
                self._execute_actions(actions_to_execute)
        else:
            self._execute_actions(actions_to_execute)

        self._block_until_settled()
        post_position_m, _ = self._get_agent_pose_world()
        delta_position_m = post_position_m - pre_position_m
        self._log_motion_debug(
            "STEP_DELTA",
            step_index=self._probe_step_count,
            post_position_m=np.round(post_position_m, 6).tolist(),
            delta_position_m=np.round(delta_position_m, 6).tolist(),
            delta_mm=np.round(delta_position_m * 1000.0, 3).tolist(),
        )
        self._probe_step_count += 1
        return self._observe()

    def close(self) -> None:
        if hasattr(self.robot_interface, "graceful_stop"):
            try:
                self.robot_interface.graceful_stop()
            except Exception:
                logger.exception("graceful_stop failed during close()")
        if hasattr(self.sensor_client, "close"):
            self.sensor_client.close()
        if hasattr(self.robot_interface, "stop_listening"):
            self.robot_interface.stop_listening()

    def _move_home_if_configured(self) -> None:
        if self.home_pose_mm_deg is None:
            return

        if len(self.home_pose_mm_deg) != 6:
            self._hard_stop(
                "INVALID_HOME_POSE",
                "home_pose_mm_deg must contain exactly 6 values",
            )

        x, y, z, roll, pitch, yaw = self.home_pose_mm_deg
        self._last_home_command_time_s = time.monotonic()
        self.robot_interface.move_arm(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
        self._wait_for_home_reset_convergence()
        self._block_until_settled()

    def _dispatch_goal_pose(
        self,
        policy_result: MotorPolicyResult,
        actions: Sequence[Action],
    ) -> bool:
        if self.goal_adapter is None:
            self._hard_stop(
                "GOAL_DISPATCHER_MISSING",
                "goal_pose was produced but no goal_adapter is configured",
            )

        # Recompute the goal pose from the action using the environment's
        # clipping logic (_goal_pose_from_relative_action) instead of the
        # policy's unclipped goal_pose.  This ensures rotation is clipped to
        # max_rotation_step_deg and translation to max_translation_step_m,
        # matching the fallback (_execute_actions) path.
        relative_action = self._first_relative_action(actions)
        if relative_action is not None:
            goal_location_m, goal_quaternion_wxyz = (
                self._goal_pose_from_relative_action(relative_action)
            )
        elif policy_result.goal_pose is not None:
            # Non-relative action (e.g. SetAgentPose for jumping): use the
            # policy's goal_pose as-is.
            goal_location_m, goal_quaternion_wxyz = policy_result.goal_pose
        else:
            return False

        self._log_motion_debug(
            "DISPATCH_GOAL_POSE",
            goal_location_m=np.round(
                np.asarray(goal_location_m, dtype=float), 6
            ).tolist(),
            goal_quaternion_wxyz=np.round(
                qt.as_float_array(goal_quaternion_wxyz), 6
            ).tolist(),
            recomputed_from_action=relative_action is not None,
        )

        accepted = bool(
            self._send_world_goal_pose_with_mode(
                location_m=goal_location_m,
                rotation_quat_wxyz=goal_quaternion_wxyz,
            )
        )
        if accepted:
            self._step_dispatched_command = True
        self._log_motion_debug(
            "DISPATCH_GOAL_POSE_RESULT",
            accepted=accepted,
            rejection=(
                self._goal_adapter_rejection_details(default="ok")
                if not accepted
                else "ok"
            ),
        )
        if not accepted and self.goal_rejection_hard_stop:
            details = self._goal_adapter_rejection_details(
                default="goal_adapter rejected a motor policy result with goal_pose"
            )
            self._hard_stop(
                "GOAL_DISPATCH_REJECTED",
                details,
            )
        if not accepted and not self.goal_rejection_hard_stop:
            logger.warning(self._goal_adapter_rejection_details())
        return accepted

    def _execute_actions(self, actions: Sequence[Action]) -> None:
        for action in actions:
            self._log_motion_debug("EXECUTE_ACTION", action_type=type(action).__name__)
            if isinstance(action, (SetAgentPose, SetSensorPose)):
                self._send_world_pose(
                    location_m=np.asarray(action.location, dtype=float),
                    rotation_quat_wxyz=_as_wxyz_quaternion(action.rotation_quat),
                )
                continue

            if isinstance(
                action,
                (MoveForward, MoveTangentially, OrientHorizontal, OrientVertical),
            ):
                goal_pose = self._goal_pose_from_relative_action(action)
                self._send_world_pose(
                    location_m=goal_pose[0],
                    rotation_quat_wxyz=goal_pose[1],
                )
                continue

            self._hard_stop(
                "UNSUPPORTED_ACTION",
                f"Real-world fallback path does not support action type {type(action).__name__}",
            )

    def _goal_pose_from_relative_action(
        self,
        action: Action,
    ) -> tuple[np.ndarray, qt.quaternion]:
        current_pos, current_quat = self._get_agent_pose_world()
        agent_rot = rot.from_quat(_quat_wxyz_to_xyzw(current_quat))
        motion_rot = self._motion_rotation_from_agent_rotation(agent_rot)

        delta = np.zeros(3, dtype=float)
        goal_quat = current_quat

        if isinstance(action, MoveForward):
            agent_forward = np.array([0.0, 0.0, -1.0], dtype=float)
            delta = motion_rot.apply(agent_forward) * float(action.distance)

        elif isinstance(action, MoveTangentially):
            direction = np.asarray(action.direction, dtype=float)
            norm = float(np.linalg.norm(direction))
            if not np.isfinite(norm) or norm == 0.0:
                self._hard_stop(
                    "INVALID_ACTION_PARAMETERS",
                    "MoveTangentially direction must be finite and non-zero",
                )
            delta = (direction / norm) * float(action.distance)

        elif isinstance(action, OrientHorizontal):
            rotation_deg = self._clip_rotation_step_deg(float(action.rotation_degrees))
            z_rotation = rot.from_rotvec([0.0, 0.0, np.radians(rotation_deg)])
            new_agent_rot = agent_rot * z_rotation
            new_motion_rot = self._motion_rotation_from_agent_rotation(new_agent_rot)

            agent_left = np.array([-1.0, 0.0, 0.0], dtype=float)
            agent_forward = np.array([0.0, 0.0, -1.0], dtype=float)
            delta = new_motion_rot.apply(agent_left) * float(
                action.left_distance
            ) + new_motion_rot.apply(agent_forward) * float(action.forward_distance)

            quat_xyzw = new_agent_rot.as_quat()
            goal_quat = qt.quaternion(
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )

        elif isinstance(action, OrientVertical):
            rotation_deg = self._clip_rotation_step_deg(float(action.rotation_degrees))
            x_rotation = rot.from_rotvec([np.radians(rotation_deg), 0.0, 0.0])
            new_agent_rot = agent_rot * x_rotation
            new_motion_rot = self._motion_rotation_from_agent_rotation(new_agent_rot)

            agent_down = np.array([0.0, -1.0, 0.0], dtype=float)
            agent_forward = np.array([0.0, 0.0, -1.0], dtype=float)
            delta = new_motion_rot.apply(agent_down) * float(
                action.down_distance
            ) + new_motion_rot.apply(agent_forward) * float(action.forward_distance)

            quat_xyzw = new_agent_rot.as_quat()
            goal_quat = qt.quaternion(
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )

        raw_delta = np.asarray(delta, dtype=float)
        raw_delta_norm = float(np.linalg.norm(raw_delta))
        delta = self._clip_translation_step(raw_delta)
        clipped_delta_norm = float(np.linalg.norm(delta))
        goal_pos = current_pos + delta

        # Prevent sensor from drifting below the table boundary.  World Y
        # (index 1) points up; the table sits at Y ≈ 0.  When the surface
        # agent follows a curved object downward, OrientVertical corrections
        # can push the sensor FOV into the table rejection zone.
        if (
            self.min_sensor_height_m is not None
            and goal_pos[1] < self.min_sensor_height_m
        ):
            logger.warning(
                "Goal Y=%.4fm below min_sensor_height=%.4fm, clipping",
                goal_pos[1],
                self.min_sensor_height_m,
            )
            goal_pos[1] = self.min_sensor_height_m
        self._log_motion_debug(
            "RELATIVE_ACTION_GOAL",
            action_type=type(action).__name__,
            current_position_m=np.round(current_pos, 6).tolist(),
            raw_delta_m=np.round(raw_delta, 6).tolist(),
            raw_delta_norm_m=round(raw_delta_norm, 6),
            delta_m=np.round(delta, 6).tolist(),
            delta_norm_m=round(clipped_delta_norm, 6),
            translation_was_clipped=bool(clipped_delta_norm + 1e-12 < raw_delta_norm),
            goal_position_m=np.round(goal_pos, 6).tolist(),
            goal_quaternion_wxyz=np.round(qt.as_float_array(goal_quat), 6).tolist(),
        )
        return goal_pos, goal_quat

    def _clip_rotation_step_deg(self, rotation_deg: float) -> float:
        max_rotation_step_deg = self._goal_adapter_max_rotation_step_deg()
        clipped = float(
            np.clip(rotation_deg, -max_rotation_step_deg, max_rotation_step_deg)
        )
        if clipped != rotation_deg:
            logger.warning(
                "Rotation step clipped: %.2f° -> %.2f° (limit ±%.2f°)",
                rotation_deg,
                clipped,
                max_rotation_step_deg,
            )
        return clipped

    def _clip_translation_step(self, delta: np.ndarray) -> np.ndarray:
        max_translation_step_m = self._goal_adapter_max_translation_step_m()
        step_norm = float(np.linalg.norm(delta))
        if step_norm <= max_translation_step_m or step_norm == 0.0:
            return delta
        logger.warning(
            "Translation step clipped: %.4fm -> %.4fm (limit %.4fm)",
            step_norm,
            max_translation_step_m,
            max_translation_step_m,
        )
        return delta * (max_translation_step_m / step_norm)

    def _goal_adapter_max_translation_step_m(self) -> float:
        if self.goal_adapter is None:
            return 0.08
        safety_config = getattr(self.goal_adapter, "safety_config", None)
        if safety_config is None:
            return 0.08
        return float(getattr(safety_config, "max_translation_step_m", 0.08))

    def _goal_adapter_max_rotation_step_deg(self) -> float:
        if self.goal_adapter is None:
            return 20.0
        safety_config = getattr(self.goal_adapter, "safety_config", None)
        if safety_config is None:
            return 20.0
        return float(getattr(safety_config, "max_rotation_step_deg", 20.0))

    def _first_relative_action(self, actions: Sequence[Action]) -> Action | None:
        """Return the first relative motion action, or None."""
        for action in actions:
            if isinstance(
                action,
                (MoveForward, MoveTangentially, OrientHorizontal, OrientVertical),
            ):
                return action
        return None

    def _send_world_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> None:
        self._log_motion_debug(
            "SEND_WORLD_POSE",
            location_m=np.round(np.asarray(location_m, dtype=float), 6).tolist(),
            rotation_wxyz=np.round(qt.as_float_array(rotation_quat_wxyz), 6).tolist(),
            via_goal_adapter=self.goal_adapter is not None,
        )
        if self.goal_adapter is not None:
            accepted = self._send_world_goal_pose_with_mode(
                location_m=location_m,
                rotation_quat_wxyz=rotation_quat_wxyz,
            )
            if accepted:
                self._step_dispatched_command = True
            self._log_motion_debug(
                "SEND_WORLD_POSE_RESULT",
                accepted=accepted,
                rejection=self._goal_adapter_rejection_details(default="ok")
                if not accepted
                else "ok",
            )
            if not accepted:
                if self.goal_rejection_hard_stop:
                    self._hard_stop(
                        "GOAL_POSE_REJECTED",
                        self._goal_adapter_rejection_details(
                            default="goal_adapter rejected a direct world-pose fallback action"
                        ),
                    )
                logger.warning(
                    "%s; skipping motion command",
                    self._goal_adapter_rejection_details(
                        default="goal_adapter rejected direct world-pose fallback action"
                    ),
                )
            return

        robot_position_m, robot_quat_wxyz = self._world_pose_to_robot_pose(
            location_m=np.asarray(location_m, dtype=float),
            rotation_quat_wxyz=rotation_quat_wxyz,
        )
        roll_deg, pitch_deg, yaw_deg = _quat_to_xyz_euler_deg(robot_quat_wxyz)
        x_mm, y_mm, z_mm = (robot_position_m * 1000.0).tolist()
        self._log_motion_debug(
            "SEND_DIRECT_ROBOT_POSE",
            x_mm=round(float(x_mm), 3),
            y_mm=round(float(y_mm), 3),
            z_mm=round(float(z_mm), 3),
            roll_deg=round(float(roll_deg), 3),
            pitch_deg=round(float(pitch_deg), 3),
            yaw_deg=round(float(yaw_deg), 3),
        )
        self.robot_interface.move_arm(
            x=float(x_mm),
            y=float(y_mm),
            z=float(z_mm),
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
        )
        self._step_dispatched_command = True

    def _goal_adapter_rejection_details(
        self, default: str = "goal_adapter rejected command"
    ) -> str:
        if self.goal_adapter is None:
            return default
        details = getattr(self.goal_adapter, "last_rejection_details", None)
        if not isinstance(details, dict):
            return default

        reason_code = details.get("reason_code")
        reason_details = details.get("details")
        if reason_code and reason_details:
            return f"{reason_code}: {reason_details}"
        if reason_details:
            return str(reason_details)
        if reason_code:
            return str(reason_code)
        return default

    def _block_until_settled(self) -> None:
        if self.settle_time_s > 0:
            time.sleep(self.settle_time_s)
        if self.settle_use_goal_convergence_gate and self._step_dispatched_command:
            self._wait_for_step_command_convergence()

    def _dispatch_motor_policy_result_with_mode(
        self,
        policy_result: MotorPolicyResult,
    ) -> bool:
        dispatch_method = self.goal_adapter.dispatch_motor_policy_result
        try:
            parameters = inspect.signature(dispatch_method).parameters
            if "stop_on_rejection" in parameters:
                return bool(
                    dispatch_method(
                        policy_result,
                        stop_on_rejection=self.goal_rejection_hard_stop,
                    )
                )
        except (TypeError, ValueError):
            pass
        return bool(dispatch_method(policy_result))

    def _send_world_goal_pose_with_mode(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> bool:
        send_method = self.goal_adapter.send_world_goal_pose
        try:
            parameters = inspect.signature(send_method).parameters
            if "stop_on_rejection" in parameters:
                return bool(
                    send_method(
                        location_m=location_m,
                        rotation_quat_wxyz=rotation_quat_wxyz,
                        stop_on_rejection=self.goal_rejection_hard_stop,
                    )
                )
        except (TypeError, ValueError):
            pass

        return bool(
            send_method(
                location_m=location_m,
                rotation_quat_wxyz=rotation_quat_wxyz,
            )
        )

    def _wait_for_step_command_convergence(self) -> None:
        target_position_m = getattr(self.goal_adapter, "_last_command_position_m", None)
        if target_position_m is None:
            return

        target_position_m = np.asarray(target_position_m, dtype=float)
        if target_position_m.shape != (3,) or not np.all(
            np.isfinite(target_position_m)
        ):
            return

        timeout_s = self._resolve_settle_convergence_timeout_s()
        tolerance_mm = self._resolve_settle_convergence_tolerance_mm()
        required_samples = self._resolve_settle_convergence_required_samples()
        deadline = time.monotonic() + timeout_s
        consecutive_in_tolerance = 0

        while True:
            current_position_m = self._get_current_robot_position_m()
            if current_position_m is not None:
                error_mm = float(
                    np.linalg.norm(current_position_m - target_position_m) * 1000.0
                )
                is_within_tolerance = error_mm <= tolerance_mm
                if is_within_tolerance:
                    consecutive_in_tolerance += 1
                else:
                    consecutive_in_tolerance = 0

                if error_mm <= tolerance_mm:
                    if consecutive_in_tolerance >= required_samples:
                        self._log_motion_debug(
                            "SETTLE_POSE_TARGET_VS_SENSED",
                            target_position_m=np.round(target_position_m, 6).tolist(),
                            sensed_position_m=np.round(current_position_m, 6).tolist(),
                            gap_mm=round(error_mm, 3),
                            tolerance_mm=round(tolerance_mm, 3),
                            consecutive_samples=consecutive_in_tolerance,
                            required_consecutive_samples=required_samples,
                            outcome="reached",
                        )
                        self._log_motion_debug(
                            "SETTLE_CONVERGENCE_REACHED",
                            error_mm=round(error_mm, 3),
                            tolerance_mm=round(tolerance_mm, 3),
                            consecutive_samples=consecutive_in_tolerance,
                            required_consecutive_samples=required_samples,
                        )
                        return

            if time.monotonic() >= deadline:
                timeout_error_mm = None
                if current_position_m is not None:
                    timeout_error_mm = float(
                        np.linalg.norm(current_position_m - target_position_m) * 1000.0
                    )
                self._log_motion_debug(
                    "SETTLE_POSE_TARGET_VS_SENSED",
                    target_position_m=np.round(target_position_m, 6).tolist(),
                    sensed_position_m=(
                        np.round(current_position_m, 6).tolist()
                        if current_position_m is not None
                        else None
                    ),
                    gap_mm=(
                        round(timeout_error_mm, 3)
                        if timeout_error_mm is not None
                        else None
                    ),
                    tolerance_mm=round(tolerance_mm, 3),
                    consecutive_samples=consecutive_in_tolerance,
                    required_consecutive_samples=required_samples,
                    outcome="timeout",
                )
                self._log_motion_debug(
                    "SETTLE_CONVERGENCE_TIMEOUT",
                    timeout_s=round(timeout_s, 3),
                    tolerance_mm=round(tolerance_mm, 3),
                    last_error_mm=(
                        round(timeout_error_mm, 3)
                        if timeout_error_mm is not None
                        else None
                    ),
                    consecutive_samples=consecutive_in_tolerance,
                    required_consecutive_samples=required_samples,
                )
                self._hard_stop(
                    "SETTLE_CONVERGENCE_TIMEOUT",
                    (
                        "robot did not converge to commanded step pose: "
                        f"last_error_mm={timeout_error_mm}, "
                        f"tolerance_mm={tolerance_mm}, "
                        f"required_consecutive_samples={required_samples}, "
                        f"achieved_consecutive_samples={consecutive_in_tolerance}"
                    ),
                )

            if self.settle_convergence_poll_s > 0:
                time.sleep(self.settle_convergence_poll_s)

    def _resolve_settle_convergence_timeout_s(self) -> float:
        if self.settle_convergence_timeout_s is not None:
            return self.settle_convergence_timeout_s

        safety_config = getattr(self.goal_adapter, "safety_config", None)
        return float(getattr(safety_config, "convergence_timeout_s", 1.2))

    def _resolve_settle_convergence_tolerance_mm(self) -> float:
        if self.settle_convergence_position_tolerance_mm is not None:
            return self.settle_convergence_position_tolerance_mm

        safety_config = getattr(self.goal_adapter, "safety_config", None)
        return float(getattr(safety_config, "convergence_position_tolerance_mm", 3.0))

    def _resolve_settle_convergence_required_samples(self) -> int:
        return self.settle_convergence_required_consecutive_samples

    def _get_current_robot_position_m(self) -> np.ndarray | None:
        if not hasattr(self.robot_interface, "get_sense_state"):
            return None

        state = self.robot_interface.get_sense_state()
        end_effector = state.get("end_effector", [])
        if len(end_effector) < 3:
            return None

        return np.asarray(end_effector[:3], dtype=float) / 1000.0

    def _wait_for_home_reset_convergence(self) -> None:
        if self.home_pose_mm_deg is None:
            return

        deadline = time.monotonic() + self.home_reset_timeout_s
        home_position_mm = np.asarray(self.home_pose_mm_deg[:3], dtype=float)
        home_orientation_quat = _xyz_euler_deg_to_quat_wxyz(
            np.asarray(self.home_pose_mm_deg[3:6], dtype=float)
        )

        last_sample_timestamp_s = float("nan")
        last_position_error_mm = float("nan")
        last_orientation_error_deg = float("nan")

        while True:
            state = self.robot_interface.get_sense_state()
            end_effector = state.get("end_effector", [])
            sample_timestamp_s = float(state.get("timestamp_s", float("nan")))
            last_sample_timestamp_s = sample_timestamp_s

            if len(end_effector) >= 6:
                current_position_mm = np.asarray(end_effector[:3], dtype=float)
                position_error_mm = float(
                    np.linalg.norm(current_position_mm - home_position_mm)
                )
                current_orientation_rad = np.asarray(end_effector[3:6], dtype=float)
                current_orientation_quat = _xyz_euler_rad_to_quat_wxyz(
                    current_orientation_rad
                )
                relative_rotation = rot.from_quat(
                    _quat_wxyz_to_xyzw(current_orientation_quat)
                ).inv() * rot.from_quat(_quat_wxyz_to_xyzw(home_orientation_quat))
                orientation_error_deg = float(np.degrees(relative_rotation.magnitude()))
                last_position_error_mm = position_error_mm
                last_orientation_error_deg = orientation_error_deg

                is_fresh_enough = (
                    self._last_home_command_time_s is None
                    or sample_timestamp_s >= self._last_home_command_time_s
                )
                within_tolerance = (
                    position_error_mm <= self.home_reset_position_tolerance_mm
                    and orientation_error_deg
                    <= self.home_reset_orientation_tolerance_deg
                )
                if is_fresh_enough and within_tolerance:
                    self._log_motion_debug(
                        "HOME_RESET_CONVERGED",
                        home_position_mm=np.round(home_position_mm, 3).tolist(),
                        home_orientation_deg=np.round(
                            np.asarray(self.home_pose_mm_deg[3:6], dtype=float), 3
                        ).tolist(),
                        position_error_mm=round(position_error_mm, 3),
                        orientation_error_deg=round(orientation_error_deg, 3),
                        sample_timestamp_s=round(sample_timestamp_s, 6)
                        if np.isfinite(sample_timestamp_s)
                        else None,
                    )
                    return

            if time.monotonic() >= deadline:
                self._hard_stop(
                    "HOME_RESET_TIMEOUT",
                    (
                        "robot did not converge to home pose after reset: "
                        f"last_sample_timestamp_s={last_sample_timestamp_s}, "
                        f"last_position_error_mm={last_position_error_mm}, "
                        f"last_orientation_error_deg={last_orientation_error_deg}"
                    ),
                )

            if self.home_reset_poll_s > 0:
                time.sleep(self.home_reset_poll_s)

    def _observe(self) -> tuple[Observations, ProprioceptiveState]:
        sensor_obs = self._capture_sensor_observation()
        state = self._build_proprioceptive_state()

        observations = Observations(
            {
                self.agent_id: AgentObservations(
                    {
                        self.sensor_id: SensorObservation(sensor_obs),
                    }
                )
            }
        )
        return observations, state

    def _capture_sensor_observation(self) -> dict[str, np.ndarray]:
        # Time the frame read vs world_camera build to detect stale-frame
        # mismatch between depth pixels and the proprioception used to build
        # the world transform. Large gaps (>~50ms) plus large pose changes can
        # produce frame-to-frame world XYZ instability that looks like depth
        # noise but is actually a temporal alignment issue.
        #
        # Burst-average N consecutive captures per step. Empirically ~70% of
        # per-pixel ToF deviation is temporally independent, so √N averaging
        # meaningfully improves the SNR of downstream OLS surface-normal fits.
        # The arm is stationary during the burst (step-settle-sense loop), so
        # all N captures image the same scene.
        burst_n = max(1, int(self.depth_burst_n))

        t_burst_start = time.monotonic()
        accum: np.ndarray | None = None
        counts: np.ndarray | None = None
        last_frame_id = None
        for i in range(burst_n):
            frame = self._read_sensor_frame()
            if i == burst_n - 1:
                last_frame_id = getattr(frame, "frame_id", None)
            depth_m = _frame_to_depth_m(frame, unit=self.depth_unit)
            if depth_m.ndim != 2:
                raise ValueError(
                    f"Expected 2D depth image, got shape {depth_m.shape}"
                )
            if accum is None:
                accum = np.zeros(depth_m.shape, dtype=np.float64)
                counts = np.zeros(depth_m.shape, dtype=np.int32)
            elif depth_m.shape != accum.shape:
                raise ValueError(
                    "Depth frame shape changed during burst: "
                    f"expected {accum.shape}, got {depth_m.shape}"
                )
            # Per-pixel valid mask: exclude no-return (0) and non-finite pixels
            # so they don't drag the average toward 0. Pixels with zero valid
            # samples across the burst stay 0 so the adapter's semantic-mask
            # threshold still rejects them.
            valid = np.isfinite(depth_m) & (depth_m > 0.0)
            accum[valid] += depth_m[valid]
            counts[valid] += 1
        t_burst_end = time.monotonic()
        avg_m = np.where(counts > 0, accum / np.maximum(counts, 1), 0.0)
        world_camera = self._compute_world_camera_matrix()
        t_camera_built = time.monotonic()

        self._log_motion_debug(
            "FRAME_TIMING",
            frame_id=last_frame_id,
            burst_n=burst_n,
            burst_read_ms=round((t_burst_end - t_burst_start) * 1000.0, 2),
            read_to_camera_ms=round((t_camera_built - t_burst_end) * 1000.0, 2),
            zero_sample_pixels=int(np.sum(counts == 0)),
            total_pixels=int(counts.size),
            mean_sample_count=round(float(counts.mean()), 3),
        )

        return self.observation_adapter.from_depth_m(
            avg_m,
            world_camera=world_camera,
        )

    def _read_sensor_frame(self) -> Any:
        if hasattr(self.sensor_client, "get_frame"):
            get_frame = self.sensor_client.get_frame
            attempts = self.sensor_frame_max_retries + 1

            supports_timeout = False
            try:
                supports_timeout = (
                    "timeout_s" in inspect.signature(get_frame).parameters
                )
            except (TypeError, ValueError):
                supports_timeout = False

            last_error: Exception | None = None
            for attempt in range(attempts):
                try:
                    if supports_timeout:
                        return get_frame(timeout_s=self.sensor_frame_timeout_s)
                    return get_frame()
                except Exception as error:
                    last_error = error
                    if attempt < attempts - 1 and self.sensor_frame_retry_delay_s > 0:
                        time.sleep(self.sensor_frame_retry_delay_s)

            if last_error is not None:
                raise last_error

            self._hard_stop(
                "SENSOR_FRAME_UNAVAILABLE",
                "sensor_client.get_frame() failed without an explicit exception",
            )

        if hasattr(self.sensor_client, "read_frame"):
            return self.sensor_client.read_frame()
        if callable(self.sensor_client):
            return self.sensor_client()

        self._hard_stop(
            "SENSOR_CLIENT_INVALID",
            "sensor_client must expose get_frame/read_frame or be callable",
        )

    def _build_proprioceptive_state(self) -> ProprioceptiveState:
        agent_position_m, agent_rotation = self._get_agent_pose_world()

        sensor_rotation = agent_rotation * self.sensor_rotation_wxyz
        sensor_position_m = agent_position_m + rot.from_quat(
            _quat_wxyz_to_xyzw(agent_rotation)
        ).apply(self.sensor_translation_m)
        sensor_relative_to_agent_m = sensor_position_m - agent_position_m

        self._log_motion_debug(
            "PROPRIO_SENSOR_POSE",
            agent_position_m=np.round(agent_position_m, 6).tolist(),
            sensor_position_m=np.round(sensor_position_m, 6).tolist(),
            sensor_relative_to_agent_m=np.round(sensor_relative_to_agent_m, 6).tolist(),
            configured_sensor_translation_m=np.round(
                self.sensor_translation_m, 6
            ).tolist(),
        )

        return ProprioceptiveState(
            {
                self.agent_id: AgentState(
                    sensors={
                        self.sensor_id: SensorState(
                            position=tuple(sensor_position_m.tolist()),
                            rotation=sensor_rotation,
                        )
                    },
                    position=tuple(agent_position_m.tolist()),
                    rotation=agent_rotation,
                )
            }
        )

    def _compute_world_camera_matrix(self) -> np.ndarray:
        agent_position_m, agent_rotation = self._get_agent_pose_world()
        sensor_rotation = agent_rotation * self.sensor_rotation_wxyz
        sensor_position_m = agent_position_m + rot.from_quat(
            _quat_wxyz_to_xyzw(agent_rotation)
        ).apply(self.sensor_translation_m)

        world_camera = np.eye(4, dtype=np.float64)
        world_camera[:3, :3] = rot.from_quat(
            _quat_wxyz_to_xyzw(sensor_rotation)
        ).as_matrix()
        world_camera[:3, 3] = sensor_position_m
        return world_camera

    def _world_to_robot_transform(self) -> tuple[np.ndarray, rot]:
        world_to_robot = getattr(self.goal_adapter, "world_to_robot", None)
        if world_to_robot is None:
            return np.zeros(3, dtype=float), rot.identity()

        translation_m = np.asarray(
            getattr(world_to_robot, "translation_m", np.zeros(3, dtype=float)),
            dtype=float,
        )
        rotation_quat_wxyz = _as_wxyz_quaternion(
            getattr(world_to_robot, "rotation_quat_wxyz", qt.one)
        )
        return translation_m, rot.from_quat(_quat_wxyz_to_xyzw(rotation_quat_wxyz))

    def _link6_to_sensor_transform(self) -> tuple[np.ndarray, rot]:
        link6_to_sensor = getattr(self.goal_adapter, "link6_to_sensor", None)
        if link6_to_sensor is None:
            return np.zeros(3, dtype=float), rot.identity()

        translation_m = np.asarray(
            getattr(link6_to_sensor, "translation_m", np.zeros(3, dtype=float)),
            dtype=float,
        )
        rotation_quat_wxyz = _as_wxyz_quaternion(
            getattr(link6_to_sensor, "rotation_quat_wxyz", qt.one)
        )
        return translation_m, rot.from_quat(_quat_wxyz_to_xyzw(rotation_quat_wxyz))

    def _world_pose_to_robot_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> tuple[np.ndarray, qt.quaternion]:
        translation_m, world_to_robot_rot = self._world_to_robot_transform()
        world_rot = rot.from_quat(_quat_wxyz_to_xyzw(rotation_quat_wxyz))

        robot_position_m = world_to_robot_rot.apply(location_m) + translation_m
        robot_rot = world_to_robot_rot * world_rot
        robot_quat_xyzw = robot_rot.as_quat()
        robot_quat_wxyz = qt.quaternion(
            robot_quat_xyzw[3],
            robot_quat_xyzw[0],
            robot_quat_xyzw[1],
            robot_quat_xyzw[2],
        )
        return robot_position_m, robot_quat_wxyz

    def _parse_sensed_orientation_wxyz(
        self,
        raw_orientation: Sequence[float],
    ) -> qt.quaternion:
        orientation_values = np.asarray(raw_orientation, dtype=float)
        if orientation_values.shape != (3,):
            raise ValueError("Expected sensed orientation with exactly 3 values")

        ordered_orientation = orientation_values[
            list(self.sensed_orientation_index_order)
        ]
        signed_orientation = ordered_orientation * self.sensed_orientation_signs

        euler_sequence = (
            self.sensed_orientation_sequence.upper()
            if self.sensed_orientation_intrinsic
            else self.sensed_orientation_sequence.lower()
        )
        sensed_rotation = rot.from_euler(
            euler_sequence,
            signed_orientation,
            degrees=self.sensed_orientation_degrees,
        )
        offset_rotation = rot.from_quat(
            _quat_wxyz_to_xyzw(self.sensed_orientation_offset_wxyz)
        )
        corrected_rotation = offset_rotation * sensed_rotation

        quat_xyzw = corrected_rotation.as_quat()
        return qt.quaternion(
            quat_xyzw[3],
            quat_xyzw[0],
            quat_xyzw[1],
            quat_xyzw[2],
        )

    def _motion_rotation_from_agent_rotation(self, agent_rotation: rot) -> rot:
        motion_offset_rotation = rot.from_quat(
            _quat_wxyz_to_xyzw(self.sensed_motion_offset_wxyz)
        )
        return motion_offset_rotation * agent_rotation

    def _get_agent_pose_world(self) -> tuple[np.ndarray, qt.quaternion]:
        if hasattr(self.robot_interface, "get_sense_state"):
            state = self.robot_interface.get_sense_state()
            ee = state.get("end_effector", [])
            if len(ee) >= 6:
                robot_position_m = np.asarray(ee[:3], dtype=float) / 1000.0
                raw_orientation = np.asarray(ee[3:6], dtype=float)
                robot_rotation = self._parse_sensed_orientation_wxyz(raw_orientation)

                translation_m, world_to_robot_rot = self._world_to_robot_transform()
                robot_rot = rot.from_quat(_quat_wxyz_to_xyzw(robot_rotation))

                world_link6_position_m = world_to_robot_rot.inv().apply(
                    robot_position_m - translation_m
                )
                world_link6_rot = world_to_robot_rot.inv() * robot_rot

                link6_to_sensor_translation_m, link6_to_sensor_rot = (
                    self._link6_to_sensor_transform()
                )
                world_position_m = world_link6_position_m + world_link6_rot.apply(
                    link6_to_sensor_translation_m
                )
                world_rot = world_link6_rot * link6_to_sensor_rot
                world_quat_xyzw = world_rot.as_quat()
                agent_rotation = qt.quaternion(
                    world_quat_xyzw[3],
                    world_quat_xyzw[0],
                    world_quat_xyzw[1],
                    world_quat_xyzw[2],
                )
                self._log_motion_debug(
                    "SENSED_POSE_PARSE",
                    raw_orientation=np.round(raw_orientation, 6).tolist(),
                    sequence=self.sensed_orientation_sequence,
                    intrinsic=self.sensed_orientation_intrinsic,
                    degrees=self.sensed_orientation_degrees,
                    index_order=list(self.sensed_orientation_index_order),
                    signs=np.round(self.sensed_orientation_signs, 6).tolist(),
                    offset_wxyz=np.round(
                        qt.as_float_array(self.sensed_orientation_offset_wxyz), 6
                    ).tolist(),
                    parsed_robot_quat_wxyz=np.round(
                        qt.as_float_array(robot_rotation), 6
                    ).tolist(),
                )
                return world_position_m, agent_rotation

        return np.zeros(3, dtype=float), qt.one

    def _hard_stop(self, reason_code: str, details: str) -> None:
        logger.error("Real-world safety stop: %s | %s", reason_code, details)
        self.last_safety_event = {"reason_code": reason_code, "details": details}

        if hasattr(self.robot_interface, "stop_motion"):
            self.robot_interface.stop_motion(reason=f"{reason_code}: {details}")

        raise RealWorldSafetyStopError(reason_code=reason_code, details=details)

    def _select_actions_for_step(self, actions: Sequence[Action]) -> Sequence[Action]:
        if not self.probe_move_forward_only:
            return actions
        if self._probe_step_count >= self.probe_max_steps:
            return actions

        self._log_motion_debug(
            "PROBE_OVERRIDE_ACTIONS",
            step_index=self._probe_step_count,
            probe_distance_m=self.probe_move_forward_distance_m,
        )
        return [
            MoveForward(
                agent_id=str(self.agent_id),
                distance=self.probe_move_forward_distance_m,
            )
        ]

    def _log_motion_debug(self, event: str, **fields: Any) -> None:
        if not self.motion_debug_logging:
            return
        if fields:
            logger.info("RW_MOTION %s | %s", event, fields)
            return
        logger.info("RW_MOTION %s", event)


def _frame_to_depth_m(frame: Any, *, unit: int) -> np.ndarray:
    """Extract a depth image in meters from any supported sensor-frame type."""
    if hasattr(frame, "distance_mm_image"):
        depth_mm = np.asarray(frame.distance_mm_image(unit=unit), dtype=np.float64)
        return depth_mm * 1e-3
    if hasattr(frame, "depth"):
        if frame.depth is None:
            raise ValueError("HTTP frame does not contain a depth image")
        return np.asarray(frame.depth, dtype=np.float64) * 1e-3
    if isinstance(frame, np.ndarray):
        return np.asarray(frame, dtype=np.float64)
    raise TypeError(
        "Unsupported sensor frame type. "
        "Expected USB frame, HTTP frame, or depth ndarray."
    )


def _as_wxyz_quaternion(value: Sequence[float] | qt.quaternion) -> qt.quaternion:
    if isinstance(value, qt.quaternion):
        return value

    if len(value) != 4:
        raise ValueError("Expected quaternion with 4 elements in [w, x, y, z] format")

    w, x, y, z = [float(v) for v in value]
    return qt.quaternion(w, x, y, z)


def _quat_wxyz_to_xyzw(quaternion_wxyz: qt.quaternion) -> np.ndarray:
    q = qt.as_float_array(quaternion_wxyz)
    return np.array([q[1], q[2], q[3], q[0]], dtype=float)


def _wrap_angle_deg(values: np.ndarray) -> np.ndarray:
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _xyz_euler_rad_to_quat_wxyz(euler_rad_xyz: np.ndarray) -> qt.quaternion:
    x, y, z, w = rot.from_euler("xyz", euler_rad_xyz, degrees=False).as_quat()
    return qt.quaternion(w, x, y, z)


def _xyz_euler_deg_to_quat_wxyz(euler_deg_xyz: np.ndarray) -> qt.quaternion:
    x, y, z, w = rot.from_euler("xyz", euler_deg_xyz, degrees=True).as_quat()
    return qt.quaternion(w, x, y, z)


def _quat_to_xyz_euler_deg(
    quaternion_wxyz: qt.quaternion,
) -> tuple[float, float, float]:
    roll_deg, pitch_deg, yaw_deg = rot.from_quat(
        _quat_wxyz_to_xyzw(quaternion_wxyz)
    ).as_euler("xyz", degrees=True)
    return float(roll_deg), float(pitch_deg), float(yaw_deg)
