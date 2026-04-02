"""Hardware-in-the-loop diagnostics for Lite6 + Maixsense A010.

These tests are opt-in and skipped by default.

Required env vars:
- TBP_ENABLE_HIL=1

Optional env vars:
- TBP_HIL_ALLOW_MOTION=1             # enable movement tests
- TBP_HIL_OBJECT_PRESENT=1           # enable stay-on-object checks
- TBP_HIL_ROBOT_IP=192.168.1.159
- TBP_HIL_USB_PORT=/dev/sipeed
- TBP_HIL_SETTLE_TIME_S=0.25
- TBP_HIL_STEP_DISTANCE_M=0.01
"""

from __future__ import annotations

import os
import time

import numpy as np
import numpy.testing as nptest
import pytest
import quaternion as qt

from tbp.monty.frameworks.actions.actions import MoveForward
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.environments.real_world_environment import (
    RealWorldLite6A010Environment,
)
from tbp.monty.frameworks.sensors import SensorID


def _flag(name: str, default: str = "0") -> bool:
    return str(os.getenv(name, default)).strip().lower() in {"1", "true", "yes", "on"}


def _require_hil() -> None:
    if not _flag("TBP_ENABLE_HIL"):
        pytest.skip("Set TBP_ENABLE_HIL=1 to run hardware-in-the-loop tests")


def _require_motion() -> None:
    if not _flag("TBP_HIL_ALLOW_MOTION"):
        pytest.skip("Set TBP_HIL_ALLOW_MOTION=1 to run motion HIL tests")


def _require_object_present() -> None:
    if not _flag("TBP_HIL_OBJECT_PRESENT"):
        pytest.skip("Set TBP_HIL_OBJECT_PRESENT=1 when an object is placed for proxy checks")


def _require_no_xdist(pytestconfig) -> None:
    numprocesses = pytestconfig.getoption("numprocesses", default=None)
    if numprocesses in (None, 0):
        return
    pytest.skip(
        "HIL tests require exclusive access to sensor USB port. "
        "Run with '-n 0' (disable xdist parallelism)."
    )


@pytest.fixture(scope="module")
def hil_settings(pytestconfig) -> dict[str, float | str]:
    _require_hil()
    _require_no_xdist(pytestconfig)
    return {
        "robot_ip": os.getenv("TBP_HIL_ROBOT_IP", "192.168.1.159"),
        "usb_port": os.getenv("TBP_HIL_USB_PORT", "/dev/sipeed"),
        "settle_time_s": float(os.getenv("TBP_HIL_SETTLE_TIME_S", "0.25")),
        "step_distance_m": float(os.getenv("TBP_HIL_STEP_DISTANCE_M", "0.01")),
        "sensor_timeout_s": float(os.getenv("TBP_HIL_SENSOR_TIMEOUT_S", "0.15")),
        "sensor_retries": int(os.getenv("TBP_HIL_SENSOR_RETRIES", "6")),
        "sensor_retry_delay_s": float(os.getenv("TBP_HIL_SENSOR_RETRY_DELAY_S", "0.10")),
    }


@pytest.fixture(scope="module")
def hardware_bundle(hil_settings):
    factory = pytest.importorskip("multimodal_monty_meets_world.factory")

    robot = factory.create_robot_interface(
        ip_address=str(hil_settings["robot_ip"]),
        start_listener=True,
    )
    sensor = factory.create_usb_frame_client(
        port=str(hil_settings["usb_port"]),
        baudrate=921600,
        timeout=float(hil_settings["sensor_timeout_s"]),
    )
    adapter = factory.create_observation_adapter(
        fx=230.0,
        fy=230.0,
        cx=160.0,
        cy=120.0,
        crop_center_to_square=True,
        min_valid_depth_m=1.0e-6,
    )
    goal_adapter = factory.create_goal_adapter(
        robot=robot,
        workspace_min_xyz_m=[0.135, -0.23, 0.075],
        workspace_max_xyz_m=[0.40, 0.23, 0.4],
        max_translation_step_m=0.03,
        max_rotation_step_deg=10.0,
        orientation_min_euler_deg=[-180.0, -170.0, -180.0],
        orientation_max_euler_deg=[180.0, 170.0, 180.0],
        wait_until_ready=True,
        wait_until_ready_timeout_s=2.0,
        wait_until_ready_poll_s=0.02,
        safety_profile="strict",
    )

    yield {
        "robot": robot,
        "sensor": sensor,
        "adapter": adapter,
        "goal_adapter": goal_adapter,
    }

    if hasattr(sensor, "close"):
        sensor.close()
    if hasattr(robot, "stop_listening"):
        robot.stop_listening()


def _make_env(
    *,
    hardware_bundle,
    hil_settings,
    sensor_translation_m=(0.0, 0.0, 0.0),
    sensor_rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
    goal_adapter=None,
) -> RealWorldLite6A010Environment:
    return RealWorldLite6A010Environment(
        robot_interface=hardware_bundle["robot"],
        sensor_client=hardware_bundle["sensor"],
        observation_adapter=hardware_bundle["adapter"],
        goal_adapter=goal_adapter or hardware_bundle["goal_adapter"],
        settle_time_s=float(hil_settings["settle_time_s"]),
        require_object_swap_confirmation=False,
        sensor_translation_m=sensor_translation_m,
        sensor_rotation_wxyz=sensor_rotation_wxyz,
        sensor_frame_timeout_s=float(hil_settings["sensor_timeout_s"]),
        sensor_frame_max_retries=int(hil_settings["sensor_retries"]),
        sensor_frame_retry_delay_s=float(hil_settings["sensor_retry_delay_s"]),
        goal_rejection_hard_stop=False,
    )


def _retry_step(env: RealWorldLite6A010Environment, actions, retries: int = 2):
    for attempt in range(retries + 1):
        try:
            return env.step(actions)
        except Exception as exc:  # noqa: BLE001
            name = type(exc).__name__
            message = str(exc).lower()
            is_sensor_error = name in {"SensorConnectionError", "SensorTimeoutError"} or (
                "usb communication error" in message
                or "no frame received" in message
                or "device reports readiness to read but returned no data" in message
            )
            if not is_sensor_error:
                raise
            if attempt >= retries:
                raise
            time.sleep(0.2)

    raise RuntimeError("Unreachable sensor retry branch")


def _ensure_robot_ready_for_motion(robot) -> None:
    arm = getattr(robot, "arm", None)
    if arm is not None:
        if hasattr(arm, "motion_enable"):
            arm.motion_enable(enable=True)
        if hasattr(arm, "set_mode"):
            arm.set_mode(0)
        if hasattr(arm, "set_state"):
            arm.set_state(0)
        time.sleep(0.2)

    wait_until_ready = getattr(robot, "wait_until_ready", None)
    if callable(wait_until_ready):
        assert wait_until_ready(timeout_s=2.0, poll_interval_s=0.02), (
            "Robot did not return to ready state before motion test"
        )


def _extract_depth_valid_ratio(observations) -> float:
    sensor_obs = observations[AgentID("agent_id_0")][SensorID("patch")]
    depth = np.asarray(sensor_obs["depth"], dtype=float)
    valid = np.isfinite(depth) & (depth > 0.0)
    return float(valid.mean())


def _wait_for_pose_change(
    env: RealWorldLite6A010Environment,
    initial_position_m: np.ndarray,
    *,
    min_change_m: float,
    timeout_s: float,
    poll_s: float = 0.05,
) -> float:
    start = time.monotonic()
    max_delta = 0.0
    while time.monotonic() - start <= timeout_s:
        current_position_m, _ = env._get_agent_pose_world()
        delta = float(np.linalg.norm(current_position_m - initial_position_m))
        max_delta = max(max_delta, delta)
        if delta >= min_change_m:
            return delta
        time.sleep(poll_s)
    return max_delta


def _rejection_summary(goal_adapter) -> str:
    details = getattr(goal_adapter, "last_rejection_details", None)
    if not isinstance(details, dict):
        return "<no rejection details>"
    reason_code = details.get("reason_code", "<missing reason_code>")
    reason_details = details.get("details", "<missing details>")
    return f"{reason_code}: {reason_details}"


@pytest.mark.hil
def test_hil_world_camera_matrix_is_rigid(hardware_bundle, hil_settings) -> None:
    env = _make_env(hardware_bundle=hardware_bundle, hil_settings=hil_settings)

    observations, _ = env.reset()
    world_camera = np.asarray(
        observations[AgentID("agent_id_0")][SensorID("patch")]["world_camera"],
        dtype=float,
    )

    assert world_camera.shape == (4, 4)
    rot_matrix = world_camera[:3, :3]
    nptest.assert_allclose(rot_matrix.T @ rot_matrix, np.eye(3), atol=5e-3)
    assert abs(float(np.linalg.det(rot_matrix)) - 1.0) < 5e-3


@pytest.mark.hil
def test_hil_sensor_extrinsics_affect_world_camera_translation(
    hardware_bundle,
    hil_settings,
) -> None:
    env = _make_env(
        hardware_bundle=hardware_bundle,
        hil_settings=hil_settings,
        sensor_translation_m=(0.03, -0.01, 0.02),
    )

    world_camera = env._compute_world_camera_matrix()
    proprio = env._build_proprioceptive_state()
    sensor_position = np.asarray(
        proprio[AgentID("agent_id_0")].sensors[SensorID("patch")].position,
        dtype=float,
    )

    nptest.assert_allclose(world_camera[:3, 3], sensor_position, atol=2e-3)


@pytest.mark.hil
def test_hil_rejection_details_are_populated_for_out_of_workspace_pose(
) -> None:
    factory = pytest.importorskip("multimodal_monty_meets_world.factory")

    class _DryRunRobot:
        def __init__(self) -> None:
            self.stop_reason = None

        def is_api_healthy(self):
            return True

        def wait_until_ready(self, timeout_s=2.0, poll_interval_s=0.02):  # noqa: ARG002
            return True

        def get_api_health_snapshot(self):
            return {
                "joint_code": 0,
                "position_code": 0,
                "error_code": 0,
            }

        def stop_motion(self, reason):
            self.stop_reason = reason

        def get_sense_state(self):
            return {
                "end_effector": [300.0, 0.0, 200.0, 0.0, 0.0, 0.0],
                "api_status": {"joint_code": 0, "position_code": 0},
            }

        def get_joint_limit_margin_rad(self, _joint_limits_rad):
            return 1.0

        def is_target_pose_feasible(self, *_args):
            return True

        def move_arm(self, x, y, z, roll, pitch, yaw):  # noqa: ANN001, ARG002
            return None

    dry_run_robot = _DryRunRobot()
    rejecting_adapter = factory.create_goal_adapter(
        robot=dry_run_robot,
        workspace_min_xyz_m=[0.0, 0.0, 0.0],
        workspace_max_xyz_m=[0.01, 0.01, 0.01],
        max_translation_step_m=0.001,
        max_rotation_step_deg=2.0,
        wait_until_ready=True,
        wait_until_ready_timeout_s=1.0,
        wait_until_ready_poll_s=0.02,
        safety_profile="strict",
    )

    accepted = rejecting_adapter.send_world_goal_pose(
        location_m=np.array([0.3, 0.0, 0.2], dtype=float),
        rotation_quat_wxyz=qt.one,
    )

    assert not accepted
    assert rejecting_adapter.last_rejection_details is not None
    reason_code = rejecting_adapter.last_rejection_details.get("reason_code", "")
    assert isinstance(reason_code, str)
    assert len(reason_code) > 0


@pytest.mark.hil
@pytest.mark.requires_motion
def test_hil_goal_dispatch_feasibility_sanity(
    hardware_bundle,
    hil_settings,
) -> None:
    _require_motion()
    _ensure_robot_ready_for_motion(hardware_bundle["robot"])
    env = _make_env(hardware_bundle=hardware_bundle, hil_settings=hil_settings)
    min_realized_motion_m = float(os.getenv("TBP_HIL_MIN_REALIZED_MOTION_M", "0.0015"))
    motion_timeout_s = float(os.getenv("TBP_HIL_MOTION_OBSERVE_TIMEOUT_S", "1.5"))

    env.reset()
    current_pos, current_quat = env._get_agent_pose_world()

    conservative_target = current_pos + np.array([0.003, 0.0, 0.0], dtype=float)
    accepted = env.goal_adapter.send_world_goal_pose(
        location_m=conservative_target,
        rotation_quat_wxyz=current_quat,
    )
    assert accepted, (
        "Conservative goal pose was rejected; this strongly suggests a home-pose, "
        f"workspace, or joint-limit issue. Details: {_rejection_summary(env.goal_adapter)}"
    )

    conservative_delta = _wait_for_pose_change(
        env,
        current_pos,
        min_change_m=min_realized_motion_m,
        timeout_s=motion_timeout_s,
    )
    assert conservative_delta >= min_realized_motion_m, (
        "Conservative goal pose was accepted but no measurable movement was realized. "
        "This usually indicates robot-side command rejection (e.g., SDK code=9) or mode/state mismatch. "
        f"Observed_delta_m={conservative_delta:.6f}"
    )

    current_pos, current_quat = env._get_agent_pose_world()
    probe_target = current_pos + np.array(
        [float(hil_settings["step_distance_m"]), 0.0, 0.0],
        dtype=float,
    )
    accepted = env.goal_adapter.send_world_goal_pose(
        location_m=probe_target,
        rotation_quat_wxyz=current_quat,
    )

    assert accepted, (
        "Probe-sized goal pose was rejected; the configured step is likely too large "
        f"for the current home pose/joint limits. Details: {_rejection_summary(env.goal_adapter)}"
    )

    probe_delta = _wait_for_pose_change(
        env,
        current_pos,
        min_change_m=min_realized_motion_m,
        timeout_s=motion_timeout_s,
    )
    assert probe_delta >= min_realized_motion_m, (
        "Probe-sized goal pose was accepted but no measurable movement was realized. "
        "This suggests robot firmware rejected motion after dispatch (e.g., SDK code=9). "
        f"Observed_delta_m={probe_delta:.6f}"
    )


@pytest.mark.hil
@pytest.mark.requires_motion
def test_hil_step_drift_within_configured_translation_bound(
    hardware_bundle,
    hil_settings,
) -> None:
    _require_motion()
    _ensure_robot_ready_for_motion(hardware_bundle["robot"])
    env = _make_env(hardware_bundle=hardware_bundle, hil_settings=hil_settings)
    min_realized_motion_m = float(os.getenv("TBP_HIL_MIN_REALIZED_MOTION_M", "0.0015"))
    motion_timeout_s = float(os.getenv("TBP_HIL_MOTION_OBSERVE_TIMEOUT_S", "1.5"))

    _, pre_state = env.reset()
    pre_position = np.asarray(pre_state[AgentID("agent_id_0")].position, dtype=float)

    step_distance = float(hil_settings["step_distance_m"])
    action = MoveForward(agent_id=AgentID("agent_id_0"), distance=step_distance)
    _, post_state = _retry_step(env, [action], retries=2)
    post_position = np.asarray(post_state[AgentID("agent_id_0")].position, dtype=float)

    observed_step_m = float(np.linalg.norm(post_position - pre_position))
    max_step_m = float(env.goal_adapter.safety_config.max_translation_step_m)

    assert env.goal_adapter.last_rejection_details is None
    realized_delta = _wait_for_pose_change(
        env,
        pre_position,
        min_change_m=min_realized_motion_m,
        timeout_s=motion_timeout_s,
    )
    assert realized_delta >= min_realized_motion_m, (
        "Step command produced no measurable motion despite passing dispatch checks. "
        "Likely robot-side rejection (e.g., SDK code=9). "
        f"Observed_delta_m={realized_delta:.6f}"
    )
    assert observed_step_m <= (max_step_m * 1.25 + 0.005)


@pytest.mark.hil
@pytest.mark.requires_motion
def test_hil_depth_valid_proxy_recovers_within_step_budget(
    hardware_bundle,
    hil_settings,
) -> None:
    _require_motion()
    _require_object_present()
    _ensure_robot_ready_for_motion(hardware_bundle["robot"])
    env = _make_env(hardware_bundle=hardware_bundle, hil_settings=hil_settings)

    low_threshold = float(os.getenv("TBP_HIL_LOW_VALID_RATIO", "0.02"))
    recover_threshold = float(os.getenv("TBP_HIL_RECOVER_VALID_RATIO", "0.05"))
    total_steps = int(os.getenv("TBP_HIL_PROXY_STEPS", "8"))
    recovery_budget = int(os.getenv("TBP_HIL_RECOVERY_STEPS", "3"))
    step_distance = float(hil_settings["step_distance_m"])

    observations, _ = env.reset()
    valid_ratios: list[float] = [_extract_depth_valid_ratio(observations)]

    for _ in range(total_steps):
        action = MoveForward(agent_id=AgentID("agent_id_0"), distance=step_distance)
        observations, _ = _retry_step(env, [action], retries=2)
        valid_ratios.append(_extract_depth_valid_ratio(observations))

    for index, ratio in enumerate(valid_ratios):
        if ratio < low_threshold:
            upper = min(len(valid_ratios), index + recovery_budget + 1)
            recovered = any(r >= recover_threshold for r in valid_ratios[index + 1 : upper])
            assert recovered, (
                f"valid-depth ratio dropped below {low_threshold} at step {index} "
                f"and did not recover to >= {recover_threshold} within {recovery_budget} steps"
            )
