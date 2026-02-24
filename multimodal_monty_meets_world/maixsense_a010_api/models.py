from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


class DisplayMode(IntEnum):
    OFF = 0
    LCD = 1
    USB = 2
    UART = 4


@dataclass(frozen=True)
class LensCoefficients:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass(frozen=True)
class FrameConfig:
    trigger_mode: int = 1
    deep_mode: int = 1
    deep_shift: int = 255
    ir_mode: int = 1
    status_mode: int = 2
    status_mask: int = 7
    rgb_mode: int = 1
    rgb_res: int = 0
    expose_time: int = 0


@dataclass
class A010UsbFrame:
    command: int
    output_mode: int
    sensor_temp: int
    driver_temp: int
    exposure_time: int
    err_code: int
    reserved1: int
    rows: int
    cols: int
    frame_id: int
    isp_version: int
    reserved3: int
    payload: bytes
    checksum: int
    tail: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.rows, self.cols)

    def depth_index_image(self) -> np.ndarray:
        arr = np.frombuffer(self.payload, dtype=np.uint8)
        if arr.size != self.rows * self.cols:
            raise ValueError(
                f"Payload length mismatch: got {arr.size}, expected {self.rows * self.cols}"
            )
        return arr.reshape((self.rows, self.cols))

    def distance_mm_image(self, unit: int = 0) -> np.ndarray:
        idx = self.depth_index_image().astype(np.float32)
        if unit == 0:
            dist = idx / 5.1
            return np.square(dist)
        return idx * float(unit)


@dataclass
class HttpFrame:
    frame_id: int
    stamp_msec: int
    config: FrameConfig
    depth: Optional[np.ndarray]
    ir: Optional[np.ndarray]
    status: Optional[np.ndarray]
    rgb: Optional[np.ndarray]
