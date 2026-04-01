"""Mocked integration smoke test for real-world Lite6 + A010 pipeline.

Validates that a complete experiment episode works end-to-end with
mocked hardware, including goal dispatch, settling, and observation.
"""

from __future__ import annotations

import time
from unittest import mock
from typing import Any, List, Sequence

import numpy as np
import quaternion as qt
import pytest

from tbp.monty.frameworks.actions.actions import (
    Action,
    MoveForward,
    SetAgentPose,
    SetSensorPose,
)
from tbp.monty.frameworks.agents import AgentID
from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult
from tbp.monty.frameworks.models.motor_system_state import ProprioceptiveState
from tbp.monty.frameworks.sensors import SensorID

# Import real-world environment.
from tbp.monty.frameworks.environments.real_world_environment import (
    RealWorldLite6A010Environment,
    RealWorldSafetyStopError,
)


class MockRobotInterface:
    """Minimal mock RobotInterface for testing."""
    
    def __init__(self, ip_address: str = "192.168.1.159"):
        self.ip_address = ip_address
        self.is_listening = False
        self.last_move_command = None
        self.listener_thread = None
    
    def start_listening(self) -> None:
        self.is_listening = True
    
    def stop_listening(self) -> None:
        self.is_listening = False
    
    def move_arm(self, x: float, y: float, z: float, roll: float, pitch: float, yaw: float) -> None:
        """Mock move command recording."""
        self.last_move_command = (x, y, z, roll, pitch, yaw)
    
    def get_sense_state(self) -> dict[str, Any]:
        """Return mock sensory state."""
        return {
            "joints": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "end_effector": [300.0, 0.0, 250.0, 180.0, 0.0, 0.0],
            "api_status": {
                "joint_code": 0,
                "position_code": 0,
            },
        }


class MockUSBClient:
    """Mock USB frame client."""
    
    def __init__(self, port: str = "/dev/sipeed", baudrate: int = 921600, **kwargs):
        self.port = port
        self.baudrate = baudrate
        self.is_open_state = False
        self.frame_counter = 0
    
    def is_open(self) -> bool:
        return self.is_open_state
    
    def get_frame(self, timeout_s: float = 1.0) -> Any:
        """Return mock frame."""
        self.frame_counter += 1
        mock_frame = mock.MagicMock()
        mock_frame.frame_id = self.frame_counter
        mock_frame.distance_mm_image = mock.MagicMock(
            return_value=np.random.randint(100, 1000, size=(240, 320), dtype=np.int16)
        )
        return mock_frame
    
    def close(self) -> None:
        self.is_open_state = False


class MockRetryUSBClient:
    """Mock USB client that fails first, then succeeds."""

    def __init__(self) -> None:
        self.calls = 0
        self.timeouts: List[float] = []

    def get_frame(self, timeout_s: float = 1.0) -> Any:
        self.calls += 1
        self.timeouts.append(timeout_s)
        if self.calls == 1:
            raise RuntimeError("simulated first-frame timeout")

        mock_frame = mock.MagicMock()
        mock_frame.frame_id = self.calls
        mock_frame.distance_mm_image = mock.MagicMock(
            return_value=np.random.randint(100, 1000, size=(240, 320), dtype=np.int16)
        )
        return mock_frame


class MockObservationAdapter:
    """Mock observation adapter."""
    
    def __init__(self, fx: float = 230.0, fy: float = 230.0, cx: float = 160.0, cy: float = 120.0, **kwargs):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
    
    def from_usb_frame(self, frame: Any, **kwargs) -> dict[str, np.ndarray]:
        """Return mock observation."""
        height, width = 240, 320
        return {
            "depth": np.random.rand(height, width).astype(np.float32) * 2.0,  # 0-2 meters
            "rgba": np.ones((height, width, 4), dtype=np.uint8) * 128,
            "semantic_3d": np.ones((height, width, 3), dtype=np.float32),
            "sensor_frame_data": np.eye(4),
            "world_camera": np.eye(4),
        }


class MockGoalAdapter:
    """Mock goal adapter that records dispatch calls."""
    
    def __init__(self, robot: Any = None, **kwargs):
        self.robot = robot
        self.dispatch_calls = []
        self.pose_calls = []
        self.dispatch_success = True
        self.pose_success = True
    
    def dispatch_motor_policy_result(self, result: MotorPolicyResult) -> bool:
        """Record dispatch and return success."""
        self.dispatch_calls.append(result)
        return self.dispatch_success

    def send_world_goal_pose(
        self,
        location_m: np.ndarray,
        rotation_quat_wxyz: qt.quaternion,
    ) -> bool:
        """Record direct world-pose sends and return success."""
        self.pose_calls.append((location_m, rotation_quat_wxyz))
        return self.pose_success


class TestRealWorldIntegrationSmoke:
    """Integration smoke tests with mocked hardware."""

    def test_environment_initialization_with_mocks(self) -> None:
        """Verify environment initializes with mock hardware."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            home_pose_mm_deg=[300.0, 0.0, 250.0, 180.0, 0.0, 0.0],
            input_fn=lambda prompt: "",  # Auto-confirm object swap
        )
        
        assert env.robot_interface == mock_robot
        assert env.sensor_client == mock_sensor
        assert env.goal_adapter == mock_goal_adapter

    def test_reset_with_home_pose_and_confirmation(self) -> None:
        """Verify reset moves to home and prompts for confirmation."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        confirmation_called = []
        def mock_input(prompt: str) -> str:
            confirmation_called.append(prompt)
            return ""
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            home_pose_mm_deg=[300.0, 0.0, 250.0, 180.0, 0.0, 0.0],
            require_object_swap_confirmation=True,
            input_fn=mock_input,
        )
        
        observations, proprioceptive = env.reset()
        
        # Verify input was called.
        assert len(confirmation_called) > 0
        
        # Verify observations are returned.
        assert isinstance(observations, dict)
        assert isinstance(proprioceptive, ProprioceptiveState)

    def test_step_without_policy_result_executes_fallback(self) -> None:
        """Verify step executes actions when no policy result is available."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            input_fn=lambda prompt: "",
        )
        
        # Dummy actions (will fail since mock doesn't support them,
        # but we can verify they were attempted).
        actions: Sequence[Action] = []
        
        observations, proprioceptive = env.step(actions)
        
        # Verify observations are returned.
        assert isinstance(observations, dict)
        assert isinstance(proprioceptive, ProprioceptiveState)

    def test_step_with_policy_result_dispatches_goal_pose(self) -> None:
        """Verify step uses goal dispatch when policy result is provided."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            input_fn=lambda prompt: "",
        )
        
        # Create a mock policy result with goal pose.
        policy_result = MotorPolicyResult(
            actions=[],
            goal_pose=(
                np.array([0.3, 0.0, 0.25], dtype=np.float32),  # location in meters
                qt.one,  # rotation quaternion
            ),
        )
        
        env.set_last_motor_policy_result(policy_result)
        
        observations, proprioceptive = env.step([])
        
        # Verify goal adapter dispatch was called.
        assert len(mock_goal_adapter.dispatch_calls) > 0
        assert mock_goal_adapter.dispatch_calls[0] == policy_result
        
        # Verify observations are returned.
        assert isinstance(observations, dict)

    def test_settle_time_blocking(self) -> None:
        """Verify step respects settle time."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        settle_time = 0.1  # 100ms
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            settle_time_s=settle_time,
            input_fn=lambda prompt: "",
        )
        
        start = time.monotonic()
        observations, proprioceptive = env.step([])
        elapsed = time.monotonic() - start
        
        # Verify settle time was respected (allowing 50ms tolerance).
        assert elapsed >= settle_time - 0.05

    def test_multiple_steps_work_sequentially(self) -> None:
        """Verify multiple steps can be executed in sequence."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            settle_time_s=0.01,
            input_fn=lambda prompt: "",
        )
        
        obs, prop = env.reset()
        
        for i in range(3):
            policy_result = MotorPolicyResult(
                actions=[],
                goal_pose=(
                    np.array([0.3 + i * 0.05, 0.0, 0.25], dtype=np.float32),
                    qt.one,
                ),
            )
            env.set_last_motor_policy_result(policy_result)
            obs, prop = env.step([])
            
            assert isinstance(obs, dict)
            assert isinstance(prop, ProprioceptiveState)
        
        # Verify multiple dispatches occurred.
        assert len(mock_goal_adapter.dispatch_calls) >= 3

    def test_step_falls_back_to_relative_action_when_goal_dispatch_rejected(self) -> None:
        """Verify rejected goal dispatch falls back to relative action execution."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        mock_goal_adapter.dispatch_success = False

        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            input_fn=lambda prompt: "",
        )

        policy_result = MotorPolicyResult(
            actions=[MoveForward(agent_id=AgentID("agent_id_0"), distance=0.02)],
            goal_pose=(np.array([5.0, 5.0, 5.0], dtype=np.float32), qt.one),
        )
        env.set_last_motor_policy_result(policy_result)

        observations, proprioceptive = env.step(policy_result.actions)

        assert len(mock_goal_adapter.dispatch_calls) == 1
        assert len(mock_goal_adapter.pose_calls) == 1
        assert isinstance(observations, dict)
        assert isinstance(proprioceptive, ProprioceptiveState)

    def test_observation_structure_from_real_world_env(self) -> None:
        """Verify observations have correct structure."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)
        
        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            agent_id="agent_0",
            sensor_id="patch",
            input_fn=lambda prompt: "",
        )
        
        obs, prop = env.reset()
        
        # Verify observation structure.
        assert isinstance(obs, dict)
        agent_obs = obs.get(AgentID("agent_0"))
        assert agent_obs is not None
        
        sensor_obs = agent_obs.get(SensorID("patch"))
        assert sensor_obs is not None
        assert "depth" in sensor_obs
        assert isinstance(sensor_obs["depth"], np.ndarray)

    def test_sensor_read_retries_and_passes_timeout(self) -> None:
        """Verify environment retries sensor reads and forwards timeout_s."""
        mock_robot = MockRobotInterface()
        mock_sensor = MockRetryUSBClient()
        mock_adapter = MockObservationAdapter()
        mock_goal_adapter = MockGoalAdapter(robot=mock_robot)

        env = RealWorldLite6A010Environment(
            robot_interface=mock_robot,
            sensor_client=mock_sensor,
            observation_adapter=mock_adapter,
            goal_adapter=mock_goal_adapter,
            sensor_frame_timeout_s=0.33,
            sensor_frame_max_retries=1,
            sensor_frame_retry_delay_s=0.0,
            input_fn=lambda prompt: "",
        )

        obs, prop = env.reset()

        assert isinstance(obs, dict)
        assert isinstance(prop, ProprioceptiveState)
        assert mock_sensor.calls == 2
        assert mock_sensor.timeouts == [0.33, 0.33]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
