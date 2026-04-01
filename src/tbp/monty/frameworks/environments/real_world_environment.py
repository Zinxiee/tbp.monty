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
        require_object_swap_confirmation: bool = True,
        object_swap_prompt: str = "Swap/position the object, then press Enter to continue: ",
        input_fn: Callable[[str], str] | None = None,
        sensor_translation_m: Sequence[float] = (0.0, 0.0, 0.0),
        sensor_rotation_wxyz: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        depth_unit: int = 0,
        sensor_frame_timeout_s: float = 1.0,
        sensor_frame_max_retries: int = 2,
        sensor_frame_retry_delay_s: float = 0.05,
        goal_rejection_hard_stop: bool = False,
        goal_adapter_config: dict[str, Any] | None = None,
        motion_debug_logging: bool = False,
        probe_move_forward_only: bool = False,
        probe_move_forward_distance_m: float = 0.01,
        probe_max_steps: int = 5,
    ) -> None:
        self.robot_interface = robot_interface
        self.sensor_client = sensor_client
        self.observation_adapter = observation_adapter
        
        # Instantiate goal_adapter from config if not provided as object.
        if goal_adapter is None and goal_adapter_config is not None:
            try:
                from multimodal_monty_meets_world.factory import create_goal_adapter
                goal_adapter = create_goal_adapter(robot=robot_interface, **goal_adapter_config)
            except ImportError as e:
                logger.warning(f"Failed to instantiate goal_adapter from config: {e}. Goal dispatch disabled.")
        
        self.goal_adapter = goal_adapter
        self.agent_id = AgentID(agent_id)
        self.sensor_id = SensorID(sensor_id)
        self.home_pose_mm_deg = (
            tuple(home_pose_mm_deg) if home_pose_mm_deg is not None else None
        )
        self.settle_time_s = settle_time_s
        self.require_object_swap_confirmation = require_object_swap_confirmation
        self.object_swap_prompt = object_swap_prompt
        self.input_fn = input_fn or input
        self.sensor_translation_m = np.asarray(sensor_translation_m, dtype=float)
        self.sensor_rotation_wxyz = _as_wxyz_quaternion(sensor_rotation_wxyz)
        self.depth_unit = depth_unit
        self.sensor_frame_timeout_s = float(sensor_frame_timeout_s)
        self.sensor_frame_max_retries = max(0, int(sensor_frame_max_retries))
        self.sensor_frame_retry_delay_s = max(0.0, float(sensor_frame_retry_delay_s))
        self.goal_rejection_hard_stop = bool(goal_rejection_hard_stop)
        self.motion_debug_logging = bool(motion_debug_logging)
        self.probe_move_forward_only = bool(probe_move_forward_only)
        self.probe_move_forward_distance_m = max(0.0, float(probe_move_forward_distance_m))
        self.probe_max_steps = max(0, int(probe_max_steps))
        self._probe_step_count = 0

        self._last_motor_policy_result: MotorPolicyResult | None = None
        self.last_safety_event: dict[str, str] | None = None

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
                accepted = self._dispatch_goal_pose(self._last_motor_policy_result)
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
        self.robot_interface.move_arm(
            x=float(x),
            y=float(y),
            z=float(z),
            roll=float(roll),
            pitch=float(pitch),
            yaw=float(yaw),
        )
        self._block_until_settled()

    def _dispatch_goal_pose(self, policy_result: MotorPolicyResult) -> bool:
        if self.goal_adapter is None:
            self._hard_stop(
                "GOAL_DISPATCHER_MISSING",
                "goal_pose was produced but no goal_adapter is configured",
            )

        if policy_result.goal_pose is not None:
            goal_location_m, goal_quaternion_wxyz = policy_result.goal_pose
            self._log_motion_debug(
                "DISPATCH_GOAL_POSE",
                goal_location_m=np.round(np.asarray(goal_location_m, dtype=float), 6).tolist(),
                goal_quaternion_wxyz=np.round(qt.as_float_array(goal_quaternion_wxyz), 6).tolist(),
            )

        accepted = bool(self.goal_adapter.dispatch_motor_policy_result(policy_result))
        self._log_motion_debug(
            "DISPATCH_GOAL_POSE_RESULT",
            accepted=accepted,
            rejection=self._goal_adapter_rejection_details(default="ok") if not accepted else "ok",
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

        delta = np.zeros(3, dtype=float)
        goal_quat = current_quat

        if isinstance(action, MoveForward):
            agent_forward = np.array([0.0, 0.0, 1.0], dtype=float)
            delta = agent_rot.apply(agent_forward) * float(action.distance)

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

            agent_left = np.array([-1.0, 0.0, 0.0], dtype=float)
            agent_forward = np.array([0.0, 0.0, 1.0], dtype=float)
            delta = (
                new_agent_rot.apply(agent_left) * float(action.left_distance)
                + new_agent_rot.apply(agent_forward) * float(action.forward_distance)
            )

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

            agent_down = np.array([0.0, -1.0, 0.0], dtype=float)
            agent_forward = np.array([0.0, 0.0, 1.0], dtype=float)
            delta = (
                new_agent_rot.apply(agent_down) * float(action.down_distance)
                + new_agent_rot.apply(agent_forward) * float(action.forward_distance)
            )

            quat_xyzw = new_agent_rot.as_quat()
            goal_quat = qt.quaternion(
                quat_xyzw[3],
                quat_xyzw[0],
                quat_xyzw[1],
                quat_xyzw[2],
            )

        delta = self._clip_translation_step(delta)
        goal_pos = current_pos + delta
        self._log_motion_debug(
            "RELATIVE_ACTION_GOAL",
            action_type=type(action).__name__,
            current_position_m=np.round(current_pos, 6).tolist(),
            delta_m=np.round(delta, 6).tolist(),
            goal_position_m=np.round(goal_pos, 6).tolist(),
            goal_quaternion_wxyz=np.round(qt.as_float_array(goal_quat), 6).tolist(),
        )
        return goal_pos, goal_quat

    def _clip_rotation_step_deg(self, rotation_deg: float) -> float:
        max_rotation_step_deg = self._goal_adapter_max_rotation_step_deg()
        return float(np.clip(rotation_deg, -max_rotation_step_deg, max_rotation_step_deg))

    def _clip_translation_step(self, delta: np.ndarray) -> np.ndarray:
        max_translation_step_m = self._goal_adapter_max_translation_step_m()
        step_norm = float(np.linalg.norm(delta))
        if step_norm <= max_translation_step_m or step_norm == 0.0:
            return delta
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
            accepted = self.goal_adapter.send_world_goal_pose(
                location_m=location_m,
                rotation_quat_wxyz=rotation_quat_wxyz,
            )
            self._log_motion_debug(
                "SEND_WORLD_POSE_RESULT",
                accepted=accepted,
                rejection=self._goal_adapter_rejection_details(default="ok") if not accepted else "ok",
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

        roll_deg, pitch_deg, yaw_deg = _quat_to_xyz_euler_deg(rotation_quat_wxyz)
        x_mm, y_mm, z_mm = (location_m * 1000.0).tolist()
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

    def _goal_adapter_rejection_details(self, default: str = "goal_adapter rejected command") -> str:
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
        frame = self._read_sensor_frame()
        world_camera = self._compute_world_camera_matrix()

        if hasattr(frame, "distance_mm_image"):
            return self.observation_adapter.from_usb_frame(
                frame,
                world_camera=world_camera,
                unit=self.depth_unit,
            )
        if hasattr(frame, "depth"):
            return self.observation_adapter.from_http_frame(
                frame,
                world_camera=world_camera,
            )
        if isinstance(frame, np.ndarray):
            return self.observation_adapter.from_depth_m(
                frame,
                world_camera=world_camera,
            )

        raise TypeError(
            "Unsupported sensor frame type. Expected USB frame, HTTP frame, or depth ndarray."
        )

    def _read_sensor_frame(self) -> Any:
        if hasattr(self.sensor_client, "get_frame"):
            get_frame = self.sensor_client.get_frame
            attempts = self.sensor_frame_max_retries + 1

            supports_timeout = False
            try:
                supports_timeout = "timeout_s" in inspect.signature(get_frame).parameters
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

    def _get_agent_pose_world(self) -> tuple[np.ndarray, qt.quaternion]:
        if hasattr(self.robot_interface, "get_sense_state"):
            state = self.robot_interface.get_sense_state()
            ee = state.get("end_effector", [])
            if len(ee) >= 6:
                position_m = np.asarray(ee[:3], dtype=float) / 1000.0
                euler_rad = np.asarray(ee[3:6], dtype=float)
                agent_rotation = _xyz_euler_rad_to_quat_wxyz(euler_rad)
                return position_m, agent_rotation

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


def _xyz_euler_rad_to_quat_wxyz(euler_rad_xyz: np.ndarray) -> qt.quaternion:
    x, y, z, w = rot.from_euler("xyz", euler_rad_xyz, degrees=False).as_quat()
    return qt.quaternion(w, x, y, z)


def _quat_to_xyz_euler_deg(
    quaternion_wxyz: qt.quaternion,
) -> tuple[float, float, float]:
    roll_deg, pitch_deg, yaw_deg = rot.from_quat(
        _quat_wxyz_to_xyzw(quaternion_wxyz)
    ).as_euler("xyz", degrees=True)
    return float(roll_deg), float(pitch_deg), float(yaw_deg)
