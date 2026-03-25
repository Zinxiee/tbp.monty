from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import quaternion as qt
from scipy.spatial.transform import Rotation as rot

from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult

if TYPE_CHECKING:
    from robot_interface import RobotInterface
else:
    RobotInterface = Any


@dataclass(frozen=True)
class WorldToRobotTransform:
    """Rigid transform from Monty world frame to robot base frame.

    Attributes:
        translation_m: Translation from world-origin to robot-base-origin in meters.
        rotation_quat_wxyz: Rotation from world frame into robot base frame,
            represented as ``[w, x, y, z]`` quaternion.
    """

    translation_m: np.ndarray
    rotation_quat_wxyz: qt.quaternion


@dataclass(frozen=True)
class Link6ToSensorTransform:
    """Rigid transform from Link6 frame to ToF sensor optical-center frame."""

    translation_m: np.ndarray
    rotation_quat_wxyz: qt.quaternion


@dataclass(frozen=True)
class EulerConvention:
    """Euler conversion settings for robot command serialization.

    Note:
        Lowercase axis sequences in scipy represent extrinsic rotations, matching
        the current xArm deployment assumption ("xyz" extrinsic).
    """

    sequence: str = "xyz"
    degrees: bool = True


@dataclass(frozen=True)
class SafetyConfig:
    """Pre-dispatch safety and feasibility constraints."""

    workspace_min_xyz_m: np.ndarray = field(
        default_factory=lambda: np.array([0.10, -0.40, 0.02], dtype=float)
    )
    workspace_max_xyz_m: np.ndarray = field(
        default_factory=lambda: np.array([0.70, 0.40, 0.50], dtype=float)
    )
    max_translation_step_m: float = 0.10
    max_rotation_step_deg: float = 25.0
    orientation_min_euler_deg: np.ndarray = field(
        default_factory=lambda: np.array([-180.0, -120.0, -180.0], dtype=float)
    )
    orientation_max_euler_deg: np.ndarray = field(
        default_factory=lambda: np.array([180.0, 120.0, 180.0], dtype=float)
    )
    joint_limits_rad: np.ndarray | None = None
    min_joint_limit_margin_rad: float = 0.05
    keepout_spheres_m: tuple[tuple[np.ndarray, float], ...] = ()
    convergence_timeout_s: float = 1.2
    convergence_position_tolerance_mm: float = 5.0
    min_command_interval_s: float = 0.05
    payload_mass_kg: float = 0.070 # TODO: update with actual payload mass
    payload_center_of_gravity_mm: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -60.0, 13.5], dtype=float)
    )


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
        link6_to_sensor: Link6ToSensorTransform | None = None,
        safety_config: SafetyConfig | None = None,
        euler_convention: EulerConvention | None = None,
    ) -> None:
        self.robot = robot
        self.world_to_robot = world_to_robot
        self.link6_to_sensor = link6_to_sensor or identity_link6_to_sensor_transform()
        self.safety_config = safety_config or SafetyConfig()
        self.euler_convention = euler_convention or EulerConvention()

        self._last_command_timestamp_s: float | None = None
        self._last_command_position_m: np.ndarray | None = None
        self._last_command_quat_wxyz: qt.quaternion | None = None

        self._configure_payload_from_safety_config()

    def dispatch_motor_policy_result(self, result: MotorPolicyResult) -> bool:
        """Send robot command from a policy result if an absolute goal is available.

        Behavior:
        1. Use ``result.goal_pose`` when present.
        2. Return ``False`` when no absolute pose is available.
        """
        if result.goal_pose is not None:
            return self.send_world_goal_pose(*result.goal_pose)

        return False

    def send_world_goal_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> bool:
        """Convert a world-frame pose to robot command and dispatch move_arm."""
        robot_position_m, robot_quat_wxyz = self._world_pose_to_robot_pose(
            location_m=location_m,
            rotation_quat_wxyz=rotation_quat_wxyz,
        )
        roll_deg, pitch_deg, yaw_deg = self._quat_to_command_euler_deg(robot_quat_wxyz)
        x_mm, y_mm, z_mm = (robot_position_m * 1000.0).tolist()

        safety_ok, reason = self._run_safety_checks(
            target_position_m=robot_position_m,
            target_quat_wxyz=robot_quat_wxyz,
            target_euler_deg=np.array([roll_deg, pitch_deg, yaw_deg], dtype=float),
            target_pose_mm_deg=(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg),
        )
        if not safety_ok:
            self.robot.stop_motion(reason=reason)
            return False

        self.robot.move_arm(
            x=x_mm,
            y=y_mm,
            z=z_mm,
            roll=roll_deg,
            pitch=pitch_deg,
            yaw=yaw_deg,
        )

        self._last_command_timestamp_s = time.monotonic()
        self._last_command_position_m = robot_position_m
        self._last_command_quat_wxyz = robot_quat_wxyz
        return True

    def _world_pose_to_robot_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> tuple[np.ndarray, qt.quaternion]:
        """Apply world->robot transform to position and orientation.

        Given world-frame sensor goal pose ``T_world_sensor_goal``, world->robot
        calibration ``T_robot_world``, and Link6->sensor mounting transform
        ``T_link6_sensor``, this computes:

            T_robot_link6_goal = T_robot_world * T_world_sensor_goal * inv(T_link6_sensor)

        where ``T_robot_world`` is defined by ``self.world_to_robot``.
        """
        world_to_robot_rot = rot.from_quat(
            self._quat_wxyz_to_xyzw(self.world_to_robot.rotation_quat_wxyz)
        )
        sensor_world_rot = rot.from_quat(self._quat_wxyz_to_xyzw(rotation_quat_wxyz))

        link6_to_sensor_rot = rot.from_quat(
            self._quat_wxyz_to_xyzw(self.link6_to_sensor.rotation_quat_wxyz)
        )
        sensor_to_link6_rot = link6_to_sensor_rot.inv()

        sensor_world_position_m = np.asarray(location_m, dtype=float)
        sensor_to_link6_translation_m = -sensor_to_link6_rot.apply(
            np.asarray(self.link6_to_sensor.translation_m, dtype=float)
        )
        world_link6_position_m = (
            sensor_world_position_m
            + sensor_world_rot.apply(sensor_to_link6_translation_m)
        )
        world_link6_rot = sensor_world_rot * sensor_to_link6_rot

        robot_position_m = (
            world_to_robot_rot.apply(world_link6_position_m)
            + self.world_to_robot.translation_m
        )
        robot_rotation = world_to_robot_rot * world_link6_rot

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

    def _quat_to_command_euler_deg(
        self,
        quaternion_wxyz: qt.quaternion,
    ) -> tuple[float, float, float]:
        """Convert quaternion into configured Euler triplet for robot command."""
        quat_xyzw = self._quat_wxyz_to_xyzw(quaternion_wxyz)
        roll_deg, pitch_deg, yaw_deg = rot.from_quat(quat_xyzw).as_euler(
            self.euler_convention.sequence,
            degrees=self.euler_convention.degrees,
        )
        return float(roll_deg), float(pitch_deg), float(yaw_deg)

    def _run_safety_checks(
        self,
        target_position_m: np.ndarray,
        target_quat_wxyz: qt.quaternion,
        target_euler_deg: np.ndarray,
        target_pose_mm_deg: tuple[float, float, float, float, float, float],
    ) -> tuple[bool, str]:
        """Run safety and feasibility checks before dispatching robot commands."""
        if not self._check_workspace_bounds(target_position_m):
            return False, "workspace_bounds"

        if not self._check_command_interval():
            return False, "command_interval"

        if not self._check_step_size(target_position_m):
            return False, "translation_step"

        if not self._check_rotation_step(target_quat_wxyz):
            return False, "rotation_step"

        if not self._check_orientation_sanity(target_euler_deg):
            return False, "orientation_sanity"

        if not self._check_ik_feasibility(target_pose_mm_deg):
            return False, "ik_infeasible"

        if not self.robot.is_api_healthy():
            return False, "robot_api_unhealthy"

        if not self._check_joint_limit_margin():
            return False, "joint_limit_margin"

        if not self._check_keepout_proximity(target_position_m):
            return False, "keepout_proximity"

        if not self._check_previous_command_convergence():
            return False, "convergence_timeout"

        return True, "ok"

    def _check_workspace_bounds(self, position_m: np.ndarray) -> bool:
        return bool(
            np.all(position_m >= self.safety_config.workspace_min_xyz_m)
            and np.all(position_m <= self.safety_config.workspace_max_xyz_m)
        )

    def _check_command_interval(self) -> bool:
        if self._last_command_timestamp_s is None:
            return True

        elapsed_s = time.monotonic() - self._last_command_timestamp_s
        return elapsed_s >= self.safety_config.min_command_interval_s

    def _check_step_size(self, target_position_m: np.ndarray) -> bool:
        reference_position = self._last_command_position_m
        if reference_position is None:
            return True

        step_m = np.linalg.norm(target_position_m - reference_position)
        return bool(step_m <= self.safety_config.max_translation_step_m)

    def _check_rotation_step(self, target_quat_wxyz: qt.quaternion) -> bool:
        previous_quat = self._last_command_quat_wxyz
        if previous_quat is None:
            return True

        target_rot = rot.from_quat(self._quat_wxyz_to_xyzw(target_quat_wxyz))
        previous_rot = rot.from_quat(self._quat_wxyz_to_xyzw(previous_quat))
        relative_rot = previous_rot.inv() * target_rot
        angle_deg = np.degrees(np.linalg.norm(relative_rot.as_rotvec()))
        return bool(angle_deg <= self.safety_config.max_rotation_step_deg)

    def _check_orientation_sanity(self, target_euler_deg: np.ndarray) -> bool:
        if not np.all(np.isfinite(target_euler_deg)):
            return False
        return bool(
            np.all(target_euler_deg >= self.safety_config.orientation_min_euler_deg)
            and np.all(target_euler_deg <= self.safety_config.orientation_max_euler_deg)
        )

    def _check_ik_feasibility(
        self,
        target_pose_mm_deg: tuple[float, float, float, float, float, float],
    ) -> bool:
        if not hasattr(self.robot, "is_target_pose_feasible"):
            return True

        return bool(self.robot.is_target_pose_feasible(*target_pose_mm_deg))

    def _configure_payload_from_safety_config(self) -> None:
        if not hasattr(self.robot, "configure_payload"):
            return

        self.robot.configure_payload(
            mass_kg=self.safety_config.payload_mass_kg,
            center_of_gravity_mm=self.safety_config.payload_center_of_gravity_mm,
        )

    def _check_joint_limit_margin(self) -> bool:
        if self.safety_config.joint_limits_rad is None:
            return True

        margin = self.robot.get_joint_limit_margin_rad(self.safety_config.joint_limits_rad)
        return margin >= self.safety_config.min_joint_limit_margin_rad

    def _check_keepout_proximity(self, target_position_m: np.ndarray) -> bool:
        for center_m, radius_m in self.safety_config.keepout_spheres_m:
            distance = np.linalg.norm(target_position_m - np.asarray(center_m, dtype=float))
            if distance < radius_m:
                return False

        return True

    def _check_previous_command_convergence(self) -> bool:
        if self._last_command_timestamp_s is None or self._last_command_position_m is None:
            return True

        elapsed_s = time.monotonic() - self._last_command_timestamp_s
        if elapsed_s <= self.safety_config.convergence_timeout_s:
            return True

        current_position_m = self._get_current_robot_position_m()
        if current_position_m is None:
            return False

        error_mm = np.linalg.norm(current_position_m - self._last_command_position_m) * 1000.0
        return bool(error_mm <= self.safety_config.convergence_position_tolerance_mm)

    def _get_current_robot_position_m(self) -> np.ndarray | None:
        state = self.robot.get_sense_state()
        end_effector = state.get("end_effector", [])
        if len(end_effector) < 3:
            return None

        xyz_mm = np.asarray(end_effector[:3], dtype=float)
        return xyz_mm / 1000.0


def identity_world_to_robot_transform() -> WorldToRobotTransform:
    """Create identity world->robot transform for initial bring-up/tests.

    Use this only when Monty world frame is intentionally aligned to robot base
    frame. For real deployments, replace with calibrated extrinsics.
    """
    return WorldToRobotTransform(
        translation_m=np.zeros(3, dtype=float),
        rotation_quat_wxyz=qt.one,
    )


def identity_link6_to_sensor_transform() -> Link6ToSensorTransform:
    """Create identity Link6->sensor transform for bring-up only."""
    return Link6ToSensorTransform(
        translation_m=np.zeros(3, dtype=float),
        rotation_quat_wxyz=qt.one,
    )
