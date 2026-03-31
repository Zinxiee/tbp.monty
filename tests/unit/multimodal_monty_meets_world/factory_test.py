"""Unit tests for hardware object constructors."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest
import quaternion as qt

from multimodal_monty_meets_world.factory import (
    create_goal_adapter,
    create_observation_adapter,
    create_robot_interface,
    create_usb_frame_client,
)


class TestCreateRobotInterface:
    """Test RobotInterface factory function."""

    def test_creates_robot_with_ip_address(self) -> None:
        """Verify robot is created with correct IP."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.RobotInterface"
        ) as mock_robot_class:
            mock_instance = mock.MagicMock()
            mock_robot_class.return_value = mock_instance
            
            robot = create_robot_interface(ip_address="192.168.1.100", start_listener=False)
            
            mock_robot_class.assert_called_once_with("192.168.1.100")
            assert robot == mock_instance

    def test_starts_listener_by_default(self) -> None:
        """Verify listener is started when start_listener=True."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.RobotInterface"
        ) as mock_robot_class:
            mock_instance = mock.MagicMock()
            mock_robot_class.return_value = mock_instance
            
            robot = create_robot_interface(ip_address="192.168.1.100", start_listener=True)
            
            mock_instance.start_listening.assert_called_once()

    def test_skips_listener_when_disabled(self) -> None:
        """Verify listener is not started when start_listener=False."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.RobotInterface"
        ) as mock_robot_class:
            mock_instance = mock.MagicMock()
            mock_robot_class.return_value = mock_instance
            
            robot = create_robot_interface(ip_address="192.168.1.100", start_listener=False)
            
            mock_instance.start_listening.assert_not_called()

    def test_propagates_robot_connection_errors(self) -> None:
        """Verify connection errors from RobotInterface are raised."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.RobotInterface",
            side_effect=RuntimeError("Connection failed"),
        ):
            with pytest.raises(RuntimeError):
                create_robot_interface(ip_address="invalid.ip")


class TestCreateUsbFrameClient:
    """Test USB frame client factory function."""

    def test_creates_client_with_defaults(self) -> None:
        """Verify client is created with port and baudrate."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.A010UsbFrameClient"
        ) as mock_client_class:
            mock_instance = mock.MagicMock()
            mock_client_class.return_value = mock_instance
            
            client = create_usb_frame_client()
            
            mock_client_class.assert_called_once()
            call_args = mock_client_class.call_args
            assert call_args[1]["port"] in {"/dev/sipeed", "/dev/ttyUSB0"}
            assert call_args[1]["baudrate"] == 921600
            assert call_args[1]["auto_configure_stream"] is True
            assert call_args[1]["stream_display_mode"] == 3

    def test_accepts_custom_port_and_baudrate(self) -> None:
        """Verify custom port and baudrate are passed."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.A010UsbFrameClient"
        ) as mock_client_class:
            mock_instance = mock.MagicMock()
            mock_client_class.return_value = mock_instance
            
            client = create_usb_frame_client(port="/dev/ttyUSB0", baudrate=115200)
            
            call_args = mock_client_class.call_args
            assert call_args[1]["port"] == "/dev/ttyUSB0"
            assert call_args[1]["baudrate"] == 115200

    def test_includes_checksum_settings(self) -> None:
        """Verify checksum validation parameters are passed."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.A010UsbFrameClient"
        ) as mock_client_class:
            mock_instance = mock.MagicMock()
            mock_client_class.return_value = mock_instance
            
            client = create_usb_frame_client(
                validate_checksum=False, checksum_policy="strict"
            )
            
            call_args = mock_client_class.call_args
            assert call_args[1]["validate_checksum"] is False
            assert call_args[1]["checksum_policy"] == "strict"

    def test_resolves_sipeed_symlink_to_primary_data_tty(self) -> None:
        """Verify /dev/sipeed resolves to ttyUSB0 for dual-interface devices."""
        target_port = "/dev/ttyUSB1"

        mock_ports = [
            mock.Mock(device="/dev/ttyUSB1", hwid="USB VID:PID=0403:6010 SER=ABC LOCATION=1-1:1.1"),
            mock.Mock(device="/dev/ttyUSB0", hwid="USB VID:PID=0403:6010 SER=ABC LOCATION=1-1:1.0"),
        ]

        with mock.patch("multimodal_monty_meets_world.factory.os.path.islink", return_value=True), mock.patch(
            "multimodal_monty_meets_world.factory.os.path.realpath", return_value=target_port
        ), mock.patch("multimodal_monty_meets_world.factory.os.path.exists", return_value=True), mock.patch(
            "multimodal_monty_meets_world.factory.serial.tools.list_ports.comports",
            return_value=mock_ports,
        ), mock.patch("multimodal_monty_meets_world.factory.A010UsbFrameClient") as mock_client_class:
            create_usb_frame_client(port="/dev/sipeed")

            call_args = mock_client_class.call_args
            assert call_args[1]["port"] == "/dev/ttyUSB0"


class TestCreateObservationAdapter:
    """Test observation adapter factory function."""

    def test_creates_adapter_with_intrinsics(self) -> None:
        """Verify adapter is created with focal length and principal point."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.MaixsenseMontyObservationAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_observation_adapter(fx=200.0, fy=210.0, cx=160.0, cy=120.0)
            
            mock_adapter_class.assert_called_once()
            call_args = mock_adapter_class.call_args
            intrinsics = call_args[1]["intrinsics"]
            assert intrinsics.fx == 200.0
            assert intrinsics.fy == 210.0
            assert intrinsics.cx == 160.0
            assert intrinsics.cy == 120.0

    def test_crop_and_depth_parameters(self) -> None:
        """Verify optional parameters are passed through."""
        with mock.patch(
            "multimodal_monty_meets_world.factory.MaixsenseMontyObservationAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_observation_adapter(
                fx=200.0,
                fy=200.0,
                cx=160.0,
                cy=120.0,
                crop_center_to_square=False,
                min_valid_depth_m=0.1,
            )
            
            call_args = mock_adapter_class.call_args
            assert call_args[1]["crop_center_to_square"] is False
            assert call_args[1]["min_valid_depth_m"] == 0.1


class TestCreateGoalAdapter:
    """Test goal adapter factory function."""

    def test_creates_adapter_with_robot_and_defaults(self) -> None:
        """Verify goal adapter is created with robot and identity transforms."""
        mock_robot = mock.MagicMock()
        
        with mock.patch(
            "multimodal_monty_meets_world.factory.MontyGoalToRobotAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_goal_adapter(robot=mock_robot)
            
            mock_adapter_class.assert_called_once()
            call_args = mock_adapter_class.call_args
            
            # Verify robot is passed.
            assert call_args[1]["robot"] == mock_robot
            
            # Verify world_to_robot is identity.
            world_to_robot = call_args[1]["world_to_robot"]
            assert np.allclose(world_to_robot.translation_m, [0.0, 0.0, 0.0])
            assert world_to_robot.rotation_quat_wxyz == qt.one

    def test_accepts_custom_transforms(self) -> None:
        """Verify custom transforms are used."""
        mock_robot = mock.MagicMock()
        
        translation = [0.1, 0.2, 0.3]
        rotation_dict = {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
        
        with mock.patch(
            "multimodal_monty_meets_world.factory.MontyGoalToRobotAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_goal_adapter(
                robot=mock_robot,
                world_to_robot_translation_m=translation,
                world_to_robot_rotation_dict=rotation_dict,
            )
            
            call_args = mock_adapter_class.call_args
            world_to_robot = call_args[1]["world_to_robot"]
            assert np.allclose(world_to_robot.translation_m, translation)

    def test_safety_config_parameters(self) -> None:
        """Verify safety configuration is built correctly."""
        mock_robot = mock.MagicMock()
        
        with mock.patch(
            "multimodal_monty_meets_world.factory.MontyGoalToRobotAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_goal_adapter(
                robot=mock_robot,
                workspace_min_xyz_m=[0.1, -0.3, 0.0],
                workspace_max_xyz_m=[0.8, 0.3, 0.6],
                max_translation_step_m=0.05,
                max_rotation_step_deg=15.0,
                convergence_timeout_s=2.0,
                min_command_interval_s=0.01,
                wait_for_min_command_interval=False,
                wait_until_ready=False,
                wait_until_ready_timeout_s=1.5,
                wait_until_ready_poll_s=0.05,
                safety_profile="very_relaxed",
                payload_mass_kg=0.1,
            )
            
            call_args = mock_adapter_class.call_args
            safety_config = call_args[1]["safety_config"]
            
            assert np.allclose(safety_config.workspace_min_xyz_m, [0.1, -0.3, 0.0])
            assert np.allclose(safety_config.workspace_max_xyz_m, [0.8, 0.3, 0.6])
            assert safety_config.max_translation_step_m == 0.05
            assert safety_config.max_rotation_step_deg == 15.0
            assert safety_config.convergence_timeout_s == 2.0
            assert safety_config.min_command_interval_s == 0.01
            assert safety_config.wait_for_min_command_interval is False
            assert safety_config.wait_until_ready is False
            assert safety_config.wait_until_ready_timeout_s == 1.5
            assert safety_config.wait_until_ready_poll_s == 0.05
            assert safety_config.safety_profile == "very_relaxed"
            assert safety_config.payload_mass_kg == 0.1

    def test_euler_convention(self) -> None:
        """Verify Euler convention settings are applied."""
        mock_robot = mock.MagicMock()
        
        with mock.patch(
            "multimodal_monty_meets_world.factory.MontyGoalToRobotAdapter"
        ) as mock_adapter_class:
            mock_instance = mock.MagicMock()
            mock_adapter_class.return_value = mock_instance
            
            adapter = create_goal_adapter(
                robot=mock_robot,
                euler_sequence="zyx",
                euler_degrees=False,
            )
            
            call_args = mock_adapter_class.call_args
            euler_convention = call_args[1]["euler_convention"]
            
            assert euler_convention.sequence == "zyx"
            assert euler_convention.degrees is False
