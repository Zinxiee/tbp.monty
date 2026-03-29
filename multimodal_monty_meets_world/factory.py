"""Runtime object constructors for Lite6 + Maixsense A010 real-world integration.

This module provides Hydra-instantiable helper functions to construct and initialize
hardware interface objects with correct lifecycle management (e.g., listener thread startup).

Intended for use with hydra.utils.instantiate() in experiment/environment configs.
"""

from __future__ import annotations

from typing import Optional, Any
import numpy as np
import quaternion as qt

# Import with try-except for graceful degradation if modules not in path.
try:
    from ufactory_api.robot_interface import RobotInterface
    from ufactory_api.monty_goal_adapter import (
        MontyGoalToRobotAdapter,
        WorldToRobotTransform,
        Link6ToSensorTransform,
        SafetyConfig,
        EulerConvention,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import ufactory_api modules. Ensure they are in PYTHONPATH. "
        f"Error: {e}"
    )

try:
    from maixsense_a010_api.usb_frame_client import A010UsbFrameClient
    from maixsense_a010_api.monty_adapter import (
        MaixsenseMontyObservationAdapter,
        CameraIntrinsics,
    )
except ImportError as e:
    raise ImportError(
        "Failed to import maixsense_a010_api modules. Ensure they are in PYTHONPATH. "
        f"Error: {e}"
    )


def create_robot_interface(
    ip_address: str,
    start_listener: bool = True,
) -> RobotInterface:
    """Create and optionally initialize a UFactory Lite6 robot interface.
    
    Args:
        ip_address: IP address of the Lite6 (e.g., "192.168.1.159").
        start_listener: If True, start the sensory listener thread immediately.
    
    Returns:
        Initialized RobotInterface instance.
    
    Raises:
        RuntimeError: If robot connection or listener startup fails.
    """
    robot = RobotInterface(ip_address)
    if start_listener:
        robot.start_listening()
    return robot


def create_usb_frame_client(
    port: str = "/dev/sipeed",
    baudrate: int = 921600,
    timeout: float = 0.05,
    validate_checksum: bool = True,
    checksum_policy: str = "compatible",
) -> A010UsbFrameClient:
    """Create a USB frame client for Maixsense A010 sensor.
    
    Args:
        port: Serial device path (e.g., "/dev/sipeed").
        baudrate: Serial communication speed (default 921600 for A010).
        timeout: Low-level serial read timeout in seconds.
        validate_checksum: Whether to validate frame checksums.
        checksum_policy: Checksum validation strictness ("compatible" or "strict").
    
    Returns:
        Initialized A010UsbFrameClient instance.
    """
    return A010UsbFrameClient(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        validate_checksum=validate_checksum,
        checksum_policy=checksum_policy,
    )


def create_observation_adapter(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    crop_center_to_square: bool = True,
    min_valid_depth_m: float = 1e-6,
) -> MaixsenseMontyObservationAdapter:
    """Create an observation adapter for Maixsense A010 frames.
    
    Args:
        fx: Focal length in pixels (x-axis).
        fy: Focal length in pixels (y-axis).
        cx: Principal point x-coordinate in pixels.
        cy: Principal point y-coordinate in pixels.
        crop_center_to_square: If True, crop to centered square patch.
        min_valid_depth_m: Minimum valid depth in meters.
    
    Returns:
        Initialized MaixsenseMontyObservationAdapter instance.
    """
    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)
    return MaixsenseMontyObservationAdapter(
        intrinsics=intrinsics,
        crop_center_to_square=crop_center_to_square,
        min_valid_depth_m=min_valid_depth_m,
    )


def create_goal_adapter(
    robot: RobotInterface,
    world_to_robot_translation_m: Optional[list | np.ndarray] = None,
    world_to_robot_rotation_dict: Optional[dict] = None,
    link6_to_sensor_translation_m: Optional[list | np.ndarray] = None,
    link6_to_sensor_rotation_dict: Optional[dict] = None,
    workspace_min_xyz_m: Optional[list | np.ndarray] = None,
    workspace_max_xyz_m: Optional[list | np.ndarray] = None,
    max_translation_step_m: float = 0.10,
    max_rotation_step_deg: float = 25.0,
    orientation_min_euler_deg: Optional[list | np.ndarray] = None,
    orientation_max_euler_deg: Optional[list | np.ndarray] = None,
    convergence_timeout_s: float = 1.2,
    convergence_position_tolerance_mm: float = 5.0,
    payload_mass_kg: float = 0.070,
    euler_sequence: str = "xyz",
    euler_degrees: bool = True,
) -> MontyGoalToRobotAdapter:
    """Create a goal-pose-to-robot adapter with safety configuration.
    
    Args:
        robot: RobotInterface instance.
        world_to_robot_translation_m: Translation [x, y, z] in meters. Default identity.
        world_to_robot_rotation_dict: Dict with 'w', 'x', 'y', 'z' for quaternion rotation.
        link6_to_sensor_translation_m: Link6->sensor translation. Default identity.
        link6_to_sensor_rotation_dict: Link6->sensor rotation quaternion dict.
        workspace_min_xyz_m: Workspace minimum corner [x, y, z].
        workspace_max_xyz_m: Workspace maximum corner [x, y, z].
        max_translation_step_m: Max translation per command in meters.
        max_rotation_step_deg: Max rotation per command in degrees.
        orientation_min_euler_deg: Minimum Euler angles [roll, pitch, yaw].
        orientation_max_euler_deg: Maximum Euler angles [roll, pitch, yaw].
        convergence_timeout_s: Time to wait for goal convergence.
        convergence_position_tolerance_mm: Position tolerance for convergence check.
        payload_mass_kg: Estimated end-effector payload mass.
        euler_sequence: Euler angle sequence for conversion ("xyz" for extrinsic).
        euler_degrees: If True, Euler angles are in degrees.
    
    Returns:
        Initialized MontyGoalToRobotAdapter instance.
    """
    # Default world_to_robot: identity transform (world frame = robot base frame).
    if world_to_robot_translation_m is None:
        world_to_robot_translation_m = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        world_to_robot_translation_m = np.array(world_to_robot_translation_m, dtype=float)
    
    if world_to_robot_rotation_dict is None:
        world_to_robot_rotation_quat = qt.one  # identity quaternion
    else:
        w = world_to_robot_rotation_dict.get("w", 1.0)
        x = world_to_robot_rotation_dict.get("x", 0.0)
        y = world_to_robot_rotation_dict.get("y", 0.0)
        z = world_to_robot_rotation_dict.get("z", 0.0)
        world_to_robot_rotation_quat = qt.quaternion(w, x, y, z)
    
    world_to_robot = WorldToRobotTransform(
        translation_m=world_to_robot_translation_m,
        rotation_quat_wxyz=world_to_robot_rotation_quat,
    )
    
    # Default link6_to_sensor: identity transform.
    if link6_to_sensor_translation_m is None:
        link6_to_sensor_translation_m = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        link6_to_sensor_translation_m = np.array(link6_to_sensor_translation_m, dtype=float)
    
    if link6_to_sensor_rotation_dict is None:
        link6_to_sensor_rotation_quat = qt.one
    else:
        w = link6_to_sensor_rotation_dict.get("w", 1.0)
        x = link6_to_sensor_rotation_dict.get("x", 0.0)
        y = link6_to_sensor_rotation_dict.get("y", 0.0)
        z = link6_to_sensor_rotation_dict.get("z", 0.0)
        link6_to_sensor_rotation_quat = qt.quaternion(w, x, y, z)
    
    link6_to_sensor = Link6ToSensorTransform(
        translation_m=link6_to_sensor_translation_m,
        rotation_quat_wxyz=link6_to_sensor_rotation_quat,
    )
    
    # Default workspace bounds.
    if workspace_min_xyz_m is None:
        workspace_min_xyz_m = np.array([0.10, -0.40, 0.02], dtype=float)
    else:
        workspace_min_xyz_m = np.array(workspace_min_xyz_m, dtype=float)
    
    if workspace_max_xyz_m is None:
        workspace_max_xyz_m = np.array([0.70, 0.40, 0.50], dtype=float)
    else:
        workspace_max_xyz_m = np.array(workspace_max_xyz_m, dtype=float)
    
    if orientation_min_euler_deg is None:
        orientation_min_euler_deg = np.array([-180.0, -120.0, -180.0], dtype=float)
    else:
        orientation_min_euler_deg = np.array(orientation_min_euler_deg, dtype=float)
    
    if orientation_max_euler_deg is None:
        orientation_max_euler_deg = np.array([180.0, 120.0, 180.0], dtype=float)
    else:
        orientation_max_euler_deg = np.array(orientation_max_euler_deg, dtype=float)
    
    safety_config = SafetyConfig(
        workspace_min_xyz_m=workspace_min_xyz_m,
        workspace_max_xyz_m=workspace_max_xyz_m,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_deg=max_rotation_step_deg,
        orientation_min_euler_deg=orientation_min_euler_deg,
        orientation_max_euler_deg=orientation_max_euler_deg,
        convergence_timeout_s=convergence_timeout_s,
        convergence_position_tolerance_mm=convergence_position_tolerance_mm,
        payload_mass_kg=payload_mass_kg,
    )
    
    euler_convention = EulerConvention(
        sequence=euler_sequence,
        degrees=euler_degrees,
    )
    
    return MontyGoalToRobotAdapter(
        robot=robot,
        world_to_robot=world_to_robot,
        link6_to_sensor=link6_to_sensor,
        safety_config=safety_config,
        euler_convention=euler_convention,
    )
