from __future__ import annotations

import struct
from typing import Optional

import requests

from .models import FrameConfig, LensCoefficients, HttpFrame
from .protocol import decode_http_frame, frame_config_encode


class MaixsenseA010HTTP:
    def __init__(self, host: str = "192.168.233.1", port: int = 80, timeout_s: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout_s = timeout_s

    @property
    def _base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def set_config(self, config: FrameConfig) -> bool:
        payload = frame_config_encode(config)
        r = requests.post(f"{self._base_url}/set_cfg", payload, timeout=self.timeout_s)
        return r.status_code == requests.codes.ok

    def get_raw_frame(self) -> bytes:
        r = requests.get(f"{self._base_url}/getdeep", timeout=self.timeout_s)
        r.raise_for_status()
        return r.content

    def get_frame(self) -> HttpFrame:
        return decode_http_frame(self.get_raw_frame())

    def get_lens_coefficients(self) -> Optional[LensCoefficients]:
        r = requests.get(f"{self._base_url}/getinfo", timeout=self.timeout_s)
        r.raise_for_status()
        info = r.content
        if len(info) < 57:
            return None
        fx, fy, cx, cy = struct.unpack("<ffff", info[41:57])
        return LensCoefficients(fx=fx, fy=fy, cx=cx, cy=cy)
