# Copyright 2025-2026 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.actions.actions import Action, SetAgentPose, SetSensorPose
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
        goal_adapter_config: dict[str, Any] | None = None,
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

        self._last_motor_policy_result: MotorPolicyResult | None = None
        self.last_safety_event: dict[str, str] | None = None

    def set_last_motor_policy_result(
        self,
        result: MotorPolicyResult | None,
    ) -> None:
        self._last_motor_policy_result = result

    def reset(self) -> tuple[Observations, ProprioceptiveState]:
        self._move_home_if_configured()
        if self.require_object_swap_confirmation:
            self.input_fn(self.object_swap_prompt)
        return self._observe()

    def step(
        self,
        actions: Sequence[Action],
    ) -> tuple[Observations, ProprioceptiveState]:
        if self._last_motor_policy_result is not None:
            if self._last_motor_policy_result.goal_pose is not None:
                self._dispatch_goal_pose(self._last_motor_policy_result)
            else:
                self._execute_actions(actions)
        else:
            self._execute_actions(actions)

        self._block_until_settled()
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

    def _dispatch_goal_pose(self, policy_result: MotorPolicyResult) -> None:
        if self.goal_adapter is None:
            self._hard_stop(
                "GOAL_DISPATCHER_MISSING",
                "goal_pose was produced but no goal_adapter is configured",
            )

        accepted = self.goal_adapter.dispatch_motor_policy_result(policy_result)
        if not accepted:
            self._hard_stop(
                "GOAL_DISPATCH_REJECTED",
                "goal_adapter rejected a motor policy result with goal_pose",
            )

    def _execute_actions(self, actions: Sequence[Action]) -> None:
        for action in actions:
            if isinstance(action, (SetAgentPose, SetSensorPose)):
                self._send_world_pose(
                    location_m=np.asarray(action.location, dtype=float),
                    rotation_quat_wxyz=_as_wxyz_quaternion(action.rotation_quat),
                )
                continue

            self._hard_stop(
                "UNSUPPORTED_ACTION",
                f"Real-world fallback path does not support action type {type(action).__name__}",
            )

    def _send_world_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> None:
        if self.goal_adapter is not None:
            accepted = self.goal_adapter.send_world_goal_pose(
                location_m=location_m,
                rotation_quat_wxyz=rotation_quat_wxyz,
            )
            if not accepted:
                self._hard_stop(
                    "GOAL_POSE_REJECTED",
                    "goal_adapter rejected a direct world-pose fallback action",
                )
            return

        roll_deg, pitch_deg, yaw_deg = _quat_to_xyz_euler_deg(rotation_quat_wxyz)
        x_mm, y_mm, z_mm = (location_m * 1000.0).tolist()
        self.robot_interface.move_arm(
            x=float(x_mm),
            y=float(y_mm),
            z=float(z_mm),
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
        )

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
            return self.sensor_client.get_frame()
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
