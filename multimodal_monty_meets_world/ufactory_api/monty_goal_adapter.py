from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.actions.actions import Action, SetAgentPose
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult

if TYPE_CHECKING:
    from robot_interface import RobotInterface
else:
    RobotInterface = Any


@dataclass(frozen=True)
class WorldToRobotTransform:
    """Rigid transform from simulator world frame to robot base frame.

    Attributes:
        translation_m: Translation from world-origin to robot-base-origin in meters.
        rotation_quat_wxyz: Rotation from world frame into robot base frame,
            represented as ``[w, x, y, z]`` quaternion.
    """

    translation_m: np.ndarray
    rotation_quat_wxyz: qt.quaternion


class MontyGoalToRobotAdapter:
    """Convert Monty absolute goal poses into UFactory RobotInterface commands.

    This adapter consumes world-frame goal poses from ``MotorPolicyResult.goal_pose``
    and sends robot-base-frame Cartesian targets via ``RobotInterface.move_arm``.

    Conventions used by this adapter:
    - Input location units: meters.
    - Input orientation quaternion: numpy-quaternion ``[w, x, y, z]``.
    - Output position units to robot: millimeters.
    - Output orientation to robot: Euler ``xyz`` in degrees as
      ``(roll, pitch, yaw)``.

    Note:
        Euler conventions for physical robots are notoriously integration-specific.
        If roll/pitch/yaw axes appear swapped in hardware, adjust only this adapter.
    """

    def __init__(
        self,
        robot: RobotInterface,
        world_to_robot: WorldToRobotTransform,
    ) -> None:
        self.robot = robot
        self.world_to_robot = world_to_robot

    def dispatch_motor_policy_result(self, result: MotorPolicyResult) -> bool:
        """Send robot command from a policy result if an absolute goal is available.

        Behavior:
        1. Prefer ``result.goal_pose`` when present.
        2. Fall back to a ``SetAgentPose`` action if present.
        3. Return ``False`` when no absolute pose is available.
        """
        if result.goal_pose is not None:
            self.send_world_goal_pose(*result.goal_pose)
            return True

        fallback_goal = self._goal_from_actions(result.actions)
        if fallback_goal is None:
            return False

        self.send_world_goal_pose(*fallback_goal)
        return True

    def send_world_goal_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> None:
        """Convert a world-frame pose to robot command and dispatch move_arm."""
        robot_position_m, robot_quat_wxyz = self._world_pose_to_robot_pose(
            location_m=location_m,
            rotation_quat_wxyz=rotation_quat_wxyz,
        )

        x_mm, y_mm, z_mm = (robot_position_m * 1000.0).tolist()

        robot_quat_xyzw = self._quat_wxyz_to_xyzw(robot_quat_wxyz)
        roll_deg, pitch_deg, yaw_deg = rot.from_quat(robot_quat_xyzw).as_euler(
            "xyz", degrees=True
        )

        self.robot.move_arm(
            x=x_mm,
            y=y_mm,
            z=z_mm,
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
        )

    def _goal_from_actions(
        self, actions: list[Action]
    ) -> tuple[np.ndarray, qt.quaternion] | None:
        """Extract a world-frame absolute pose from action list when possible."""
        for action in actions:
            if isinstance(action, SetAgentPose):
                return np.asarray(action.location, dtype=float), action.rotation_quat

        return None

    def _world_pose_to_robot_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> tuple[np.ndarray, qt.quaternion]:
        """Apply world->robot transform to position and orientation.

        Given world-frame goal pose ``T_world_goal`` and calibration transform
        ``T_robot_world``, this computes:

            T_robot_goal = T_robot_world * T_world_goal

        where ``T_robot_world`` is defined by ``self.world_to_robot``.
        """
        world_to_robot_rot = rot.from_quat(
            self._quat_wxyz_to_xyzw(self.world_to_robot.rotation_quat_wxyz)
        )
        goal_world_rot = rot.from_quat(self._quat_wxyz_to_xyzw(rotation_quat_wxyz))

        robot_position_m = (
            world_to_robot_rot.apply(np.asarray(location_m, dtype=float))
            + self.world_to_robot.translation_m
        )
        robot_rotation = world_to_robot_rot * goal_world_rot

        quat_xyzw = robot_rotation.as_quat()
        quat_wxyz = qt.quaternion(quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])

        return robot_position_m, quat_wxyz

    @staticmethod
    def _quat_wxyz_to_xyzw(quaternion_wxyz: qt.quaternion) -> np.ndarray:
        """Convert numpy-quaternion ``[w, x, y, z]`` into scipy ``[x, y, z, w]``."""
        components_wxyz = qt.as_float_array(quaternion_wxyz)
        return np.array(
            [
                components_wxyz[1],
                components_wxyz[2],
                components_wxyz[3],
                components_wxyz[0],
            ],
            dtype=float,
        )


def identity_world_to_robot_transform() -> WorldToRobotTransform:
    """Create identity world->robot transform for initial bring-up/tests.

    Use this only when simulator world frame is intentionally aligned to robot base
    frame. For real deployments, replace with calibrated extrinsics.
    """
    return WorldToRobotTransform(
        translation_m=np.zeros(3, dtype=float),
        rotation_quat_wxyz=qt.one,
    )
