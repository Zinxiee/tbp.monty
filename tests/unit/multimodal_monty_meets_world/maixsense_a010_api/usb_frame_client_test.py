"""Unit tests for A010UsbFrameClient wrapper."""

from __future__ import annotations

import time
from unittest import mock

import pytest

from multimodal_monty_meets_world.maixsense_a010_api.models import A010UsbFrame
from multimodal_monty_meets_world.maixsense_a010_api.usb_frame_client import (
    A010UsbFrameClient,
    SensorConnectionError,
    SensorTimeoutError,
)


class TestA010UsbFrameClient:
    """Test strict blocking USB frame client."""

    def test_init_stores_parameters(self) -> None:
        """Verify initialization stores port and baudrate."""
        client = A010UsbFrameClient(port="/dev/test", baudrate=115200)
        assert client._usb_client._port == "/dev/test"
        assert client._usb_client._baudrate == 115200

    def test_is_open_reflects_connection_state(self) -> None:
        """Verify is_open tracks USB connection."""
        client = A010UsbFrameClient()
        assert not client.is_open()
        # Don't actually connect; just verify the property works.

    def test_get_frame_lazy_connects(self) -> None:
        """Verify first get_frame call attempts connection."""
        client = A010UsbFrameClient(port="/dev/test", baudrate=115200)
        
        with mock.patch.object(
            client._usb_client, "connect", side_effect=Exception("Connection failed")
        ):
            with pytest.raises(SensorConnectionError) as exc_info:
                client.get_frame(timeout_s=0.1)
            assert "Failed to connect" in str(exc_info.value)

    def test_get_frame_returns_first_frame_on_success(self) -> None:
        """Verify get_frame returns frame when available."""
        client = A010UsbFrameClient()
        
        # Create a mock frame.
        mock_frame = mock.MagicMock(spec=A010UsbFrame)
        mock_frame.frame_id = 1
        
        # Mock iter_frames to return one frame.
        with mock.patch.object(
            client._usb_client, "connect"
        ) as mock_connect, mock.patch.object(
            client._usb_client, "iter_frames", return_value=iter([mock_frame])
        ) as mock_iter:
            frame = client.get_frame(timeout_s=1.0)
            assert frame == mock_frame
            mock_connect.assert_called_once()

    def test_get_frame_raises_timeout_on_empty_stream(self) -> None:
        """Verify timeout exception when no frames received."""
        client = A010UsbFrameClient()
        
        # Mock iter_frames to yield nothing (timeout reached).
        with mock.patch.object(
            client._usb_client, "connect"
        ), mock.patch.object(
            client._usb_client, "iter_frames", return_value=iter([])
        ):
            with pytest.raises(SensorTimeoutError) as exc_info:
                client.get_frame(timeout_s=0.1)
            assert "No frame received" in str(exc_info.value)
            assert "0.100" in str(exc_info.value)  # Timeout value in message

    def test_get_frame_propagates_usb_errors(self) -> None:
        """Verify USB communication errors are caught and reported."""
        client = A010UsbFrameClient()
        
        # Mock iter_frames to raise an error during iteration.
        with mock.patch.object(
            client._usb_client, "connect"
        ), mock.patch.object(
            client._usb_client,
            "iter_frames",
            side_effect=RuntimeError("USB communication error"),
        ):
            with pytest.raises(SensorConnectionError) as exc_info:
                client.get_frame(timeout_s=1.0)
            assert "USB communication error" in str(exc_info.value)

    def test_close_stops_connection(self) -> None:
        """Verify close releases USB resources."""
        client = A010UsbFrameClient()
        
        # Mark as connected, then close.
        client._connected = True
        with mock.patch.object(client._usb_client, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()
            assert not client._connected

    def test_close_is_idempotent(self) -> None:
        """Verify close can be called multiple times safely."""
        client = A010UsbFrameClient()
        client._connected = True
        
        with mock.patch.object(client._usb_client, "close"):
            client.close()
            client.close()  # Should not raise.
            assert not client._connected

    def test_context_manager(self) -> None:
        """Verify context manager enter/exit."""
        client = A010UsbFrameClient()
        client._connected = True
        
        with mock.patch.object(client._usb_client, "close") as mock_close:
            with client as ctx:
                assert ctx is client
            mock_close.assert_called_once()

    def test_get_frame_timeout_scales_with_parameter(self) -> None:
        """Verify timeout parameter is used in iter_frames call."""
        client = A010UsbFrameClient()
        mock_frame = mock.MagicMock(spec=A010UsbFrame)
        
        with mock.patch.object(
            client._usb_client, "connect"
        ), mock.patch.object(
            client._usb_client, "iter_frames"
        ) as mock_iter:
            mock_iter.return_value = iter([mock_frame])
            
            client.get_frame(timeout_s=0.5)
            
            # Verify timeout_s was passed to iter_frames.
            call_args = mock_iter.call_args
            assert call_args[1]["timeout_s"] == 0.5

    def test_connection_error_includes_device_info(self) -> None:
        """Verify connection errors include device path for debugging."""
        client = A010UsbFrameClient(port="/dev/custom_device")
        
        with mock.patch.object(
            client._usb_client, "connect", side_effect=OSError("Permission denied")
        ):
            with pytest.raises(SensorConnectionError) as exc_info:
                client.get_frame(timeout_s=0.1)
            assert "/dev/custom_device" in str(exc_info.value)

    def test_get_frame_multiple_successful_calls(self) -> None:
        """Verify multiple frame acquisitions work without re-creating connection."""
        client = A010UsbFrameClient()
        
        frame1 = mock.MagicMock(spec=A010UsbFrame, frame_id=1)
        frame2 = mock.MagicMock(spec=A010UsbFrame, frame_id=2)
        
        with mock.patch.object(
            client._usb_client, "connect"
        ) as mock_connect, mock.patch.object(
            client._usb_client, "iter_frames"
        ) as mock_iter:
            # First call returns frame1, second returns frame2.
            mock_iter.side_effect = [iter([frame1]), iter([frame2])]
            
            f1 = client.get_frame(timeout_s=0.5)
            f2 = client.get_frame(timeout_s=0.5)
            
            assert f1 == frame1
            assert f2 == frame2
            # Connect should only be called once (lazy init).
            mock_connect.assert_called_once()

    def test_auto_configure_stream_runs_once(self) -> None:
        """Verify stream startup command is sent only once across calls."""
        client = A010UsbFrameClient(auto_configure_stream=True, stream_display_mode=3)
        frame1 = mock.MagicMock(spec=A010UsbFrame, frame_id=1)
        frame2 = mock.MagicMock(spec=A010UsbFrame, frame_id=2)

        with mock.patch.object(client._usb_client, "connect") as mock_connect, mock.patch.object(
            client._usb_client, "set_display_mode"
        ) as mock_set_display, mock.patch.object(
            client._usb_client, "iter_frames"
        ) as mock_iter:
            mock_iter.side_effect = [iter([frame1]), iter([frame2])]

            client.get_frame(timeout_s=0.5)
            client.get_frame(timeout_s=0.5)

            mock_connect.assert_called_once()
            mock_set_display.assert_called_once_with(3)

    def test_close_resets_stream_configured_state(self) -> None:
        """Verify close clears stream configured marker for reconnect."""
        client = A010UsbFrameClient(auto_configure_stream=True, stream_display_mode=3)
        frame = mock.MagicMock(spec=A010UsbFrame, frame_id=1)

        with mock.patch.object(client._usb_client, "connect"), mock.patch.object(
            client._usb_client, "set_display_mode"
        ), mock.patch.object(client._usb_client, "iter_frames", return_value=iter([frame])):
            client.get_frame(timeout_s=0.2)

        assert client._stream_configured is True

        client._connected = True
        with mock.patch.object(client._usb_client, "close"):
            client.close()

        assert client._stream_configured is False
