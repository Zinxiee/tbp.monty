"""Strict synchronous frame acquisition wrapper for Maixsense A010 USB.

Provides a simple `get_frame()` method for step-wise sensor acquisition with
explicit timeout handling and lifecycle management. Designed for integration
with strict blocking control loops (e.g., real-world robot experiments).
"""

from __future__ import annotations

import time
from typing import Optional

from .models import A010UsbFrame
from .usb_client import MaixsenseA010USB


class SensorTimeoutError(Exception):
    """Raised when sensor frame acquisition exceeds timeout."""
    pass


class SensorConnectionError(Exception):
    """Raised when sensor connection fails or is unhealthy."""
    pass


class A010UsbFrameClient:
    """Strict blocking frame client for Maixsense A010 USB.
    
    Wraps MaixsenseA010USB to provide a simple `get_frame()` API suitable for
    step-wise sensor acquisition in control loops. Handles connect/disconnect
    lifecycle and raises explicit exceptions for timeout and error conditions.
    
    Example:
        client = A010UsbFrameClient(port="/dev/sipeed", baudrate=921600)
        frame = client.get_frame(timeout_s=0.5)  # raises SensorTimeoutError if no frame
        client.close()
    """
    
    def __init__(
        self,
        port: str = "/dev/sipeed",
        baudrate: int = 921600,
        timeout: float = 0.05,
        validate_checksum: bool = True,
        checksum_policy: str = "compatible",
    ):
        """Initialize USB frame client.
        
        Args:
            port: Serial port device path (e.g., "/dev/sipeed").
            baudrate: Serial communication speed (default 921600).
            timeout: Serial read timeout in seconds for low-level polling.
            validate_checksum: Whether to validate frame checksums.
            checksum_policy: Checksum validation policy ("compatible" or "strict").
        """
        self._usb_client = MaixsenseA010USB(
            port=port,
            baudrate=baudrate,
            timeout=timeout,
            validate_checksum=validate_checksum,
            checksum_policy=checksum_policy,
        )
        self._connected = False
    
    def is_open(self) -> bool:
        """Check if USB connection is open."""
        return self._usb_client.is_open
    
    def get_frame(
        self,
        timeout_s: float = 1.0,
    ) -> A010UsbFrame:
        """Acquire next sensor frame with timeout.
        
        Ensures connection is open on first use. Polls for frames until timeout
        is exceeded or a frame is received.
        
        Args:
            timeout_s: Maximum time to wait for frame in seconds.
        
        Returns:
            A010UsbFrame containing depth, amplitude, and phase data.
        
        Raises:
            SensorConnectionError: If USB connection cannot be established.
            SensorTimeoutError: If no frame is received within timeout_s.
        """
        # Lazily connect on first call.
        if not self._connected:
            try:
                self._usb_client.connect()
                self._connected = True
            except Exception as e:
                raise SensorConnectionError(
                    f"Failed to connect to Maixsense A010 USB on "
                    f"{self._usb_client._port}: {e}"
                ) from e
        
        # Poll for frames until timeout.
        start_time = time.monotonic()
        deadline = start_time + timeout_s
        
        try:
            for frame in self._usb_client.iter_frames(
                timeout_s=timeout_s,
                keep_latest_only=True,
            ):
                return frame
        except Exception as e:
            raise SensorConnectionError(
                f"USB communication error during frame acquisition: {e}"
            ) from e
        
        # No frame received within timeout.
        elapsed = time.monotonic() - start_time
        raise SensorTimeoutError(
            f"No frame received from Maixsense A010 within {timeout_s:.3f}s "
            f"(elapsed: {elapsed:.3f}s). Check USB connection and configuration."
        )
    
    def close(self) -> None:
        """Close USB connection and release resources."""
        if self._connected:
            try:
                self._usb_client.close()
            except Exception as e:
                pass  # Log but don't re-raise during cleanup.
            finally:
                self._connected = False
    
    def __enter__(self) -> A010UsbFrameClient:
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
