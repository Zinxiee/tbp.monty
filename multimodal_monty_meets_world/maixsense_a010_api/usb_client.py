from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Generator, List, Optional

import serial
import serial.tools.list_ports

from .models import A010UsbFrame, DisplayMode
from .protocol import UsbFrameParser


BAUD_INDEX_TO_RATE = {
    0: 9600,
    1: 57600,
    2: 115200,
    3: 230400,
    4: 460800,
    5: 921600,
    6: 1000000,
    7: 2000000,
    8: 3000000,
}


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    hwid: str


class MaixsenseA010USB:
    def __init__(
        self,
        port: str = "/dev/sipeed",
        baudrate: int = 921600,
        timeout: float = 0.05,
        validate_checksum: bool = True,
        checksum_policy: str = "compatible",
    ) -> None:
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout
        self._validate_checksum = validate_checksum
        self._checksum_policy = checksum_policy
        self._serial: Optional[serial.Serial] = None
        self._parser = UsbFrameParser(
            validate_checksum=validate_checksum,
            checksum_policy=checksum_policy,
        )
        self._last_frame_id: Optional[int] = None

    @staticmethod
    def list_serial_ports() -> List[SerialPortInfo]:
        return [
            SerialPortInfo(device=p.device, description=p.description, hwid=p.hwid)
            for p in serial.tools.list_ports.comports()
        ]

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    def connect(self) -> None:
        if self.is_open:
            return
        self._serial = serial.Serial(self._port, self._baudrate, timeout=self._timeout)

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    def __enter__(self) -> "MaixsenseA010USB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_open(self) -> serial.Serial:
        if not self.is_open or self._serial is None:
            raise RuntimeError("Serial connection is not open")
        return self._serial

    def send_raw(self, data: bytes) -> int:
        ser = self._ensure_open()
        return ser.write(data)

    def send_command(self, command: str) -> int:
        if not command.endswith("\r"):
            command = command + "\r"
        return self.send_raw(command.encode("ascii"))

    def read(self, size: int = 4096) -> bytes:
        ser = self._ensure_open()
        waiting = ser.in_waiting
        if waiting <= 0:
            return ser.read(size)
        return ser.read(min(size, waiting))

    def read_text_response(self, timeout_s: float = 0.3) -> str:
        deadline = time.monotonic() + timeout_s
        chunks: List[bytes] = []
        while time.monotonic() < deadline:
            chunk = self.read(1024)
            if chunk:
                chunks.append(chunk)
                if b"\n" in chunk or b"OK" in chunk or b"ERR" in chunk:
                    break
            else:
                time.sleep(0.005)
        return b"".join(chunks).decode("utf-8", errors="ignore")

    def poll_frames(self, keep_latest_only: bool = True) -> List[A010UsbFrame]:
        chunk = self.read(8192)
        if not chunk:
            return []

        frames = self._parser.feed(chunk)
        if keep_latest_only:
            unique: List[A010UsbFrame] = []
            for frame in frames:
                if self._last_frame_id is not None and frame.frame_id == self._last_frame_id:
                    continue
                self._last_frame_id = frame.frame_id
                unique.append(frame)
            return unique
        return frames

    def iter_frames(
        self,
        timeout_s: Optional[float] = None,
        keep_latest_only: bool = True,
    ) -> Generator[A010UsbFrame, None, None]:
        start = time.monotonic()
        while True:
            if timeout_s is not None and (time.monotonic() - start) >= timeout_s:
                return

            frames = self.poll_frames(keep_latest_only=keep_latest_only)
            if not frames:
                time.sleep(0.001)
                continue

            for frame in frames:
                yield frame

    def configure_stream(
        self,
        *,
        fps: Optional[int] = None,
        display_mode: Optional[int] = None,
        anti_mmi: Optional[bool] = None,
        binning: Optional[int] = None,
        unit: Optional[int] = None,
        isp: Optional[bool] = None,
        ae: Optional[bool] = None,
        ev: Optional[int] = None,
    ) -> None:
        if isp is not None:
            self.set_isp(isp)
        if display_mode is not None:
            self.set_display_mode(display_mode)
        if anti_mmi is not None:
            self.set_anti_mmi(anti_mmi)
        if binning is not None:
            self.set_binning(binning)
        if unit is not None:
            self.set_unit(unit)
        if fps is not None:
            self.set_fps(fps)
        if ae is not None:
            self.set_auto_exposure(ae)
        if ev is not None:
            self.set_exposure_value(ev)

    def set_isp(self, enabled: bool) -> None:
        self.send_command(f"AT+ISP={1 if enabled else 0}")

    def set_display_mode(self, mode: int) -> None:
        self.send_command(f"AT+DISP={int(mode)}")

    def set_display(self, lcd: bool = False, usb: bool = True, uart: bool = False) -> None:
        mode = (
            (DisplayMode.LCD if lcd else 0)
            | (DisplayMode.USB if usb else 0)
            | (DisplayMode.UART if uart else 0)
        )
        self.set_display_mode(int(mode))

    def set_anti_mmi(self, enabled: bool) -> None:
        self.send_command(f"AT+ANTIMMI={1 if enabled else 0}")

    def set_binning(self, factor: int) -> None:
        if factor not in (1, 2, 4):
            raise ValueError("Binning factor must be one of: 1, 2, 4")
        self.send_command(f"AT+BINN={factor}")

    def set_fps(self, fps: int) -> None:
        if fps < 1 or fps > 30:
            raise ValueError("FPS must be in [1, 30]")
        self.send_command(f"AT+FPS={fps}")

    def set_unit(self, unit: int) -> None:
        if unit < 0 or unit > 10:
            raise ValueError("Unit must be in [0, 10]")
        self.send_command(f"AT+UNIT={unit}")

    def set_auto_exposure(self, enabled: bool) -> None:
        self.send_command(f"AT+AE={1 if enabled else 0}")

    def set_exposure_value(self, ev: int) -> None:
        if ev < 0 or ev > 40000:
            raise ValueError("EV must be in [0, 40000]")
        self.send_command(f"AT+EV={ev}")

    def set_baud_index(self, index: int) -> int:
        if index not in BAUD_INDEX_TO_RATE:
            raise ValueError(f"Unsupported baud index: {index}")
        self.send_command(f"AT+BAUD={index}")
        return BAUD_INDEX_TO_RATE[index]

    def apply_baud_index(self, index: int, wait_s: float = 0.15) -> int:
        new_baud = self.set_baud_index(index)
        time.sleep(wait_s)
        ser = self._ensure_open()
        ser.baudrate = new_baud
        self._baudrate = new_baud
        return new_baud
