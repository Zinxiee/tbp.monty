from __future__ import annotations

import struct
from typing import List, Optional

import numpy as np

from .models import A010UsbFrame, FrameConfig, HttpFrame


FRAME_HEAD = b"\x00\xFF"
ALLOWED_TAILS = (0xCC, 0xDD)


def frame_config_encode(config: FrameConfig) -> bytes:
    return struct.pack(
        "<BBBBBBBBi",
        config.trigger_mode,
        config.deep_mode,
        config.deep_shift,
        config.ir_mode,
        config.status_mode,
        config.status_mask,
        config.rgb_mode,
        config.rgb_res,
        config.expose_time,
    )


def frame_config_decode(frame_config: bytes) -> FrameConfig:
    fields = struct.unpack("<BBBBBBBBi", frame_config)
    return FrameConfig(
        trigger_mode=fields[0],
        deep_mode=fields[1],
        deep_shift=fields[2],
        ir_mode=fields[3],
        status_mode=fields[4],
        status_mask=fields[5],
        rgb_mode=fields[6],
        rgb_res=fields[7],
        expose_time=fields[8],
    )


class UsbFrameParser:
    def __init__(
        self,
        little_endian: bool = True,
        validate_checksum: bool = True,
        checksum_policy: str = "compatible",
    ) -> None:
        self._buf = bytearray()
        self._endian = "<" if little_endian else ">"
        self._validate_checksum = validate_checksum
        if checksum_policy not in {"compatible", "strict", "off"}:
            raise ValueError("checksum_policy must be one of: compatible, strict, off")
        self._checksum_policy = checksum_policy

    def feed(self, data: bytes) -> List[A010UsbFrame]:
        self._buf.extend(data)
        frames: List[A010UsbFrame] = []

        while True:
            idx = self._buf.find(FRAME_HEAD)
            if idx < 0:
                if len(self._buf) > 8192:
                    self._buf.clear()
                break

            if idx > 0:
                del self._buf[:idx]

            if len(self._buf) < 4:
                break

            data_len = struct.unpack(self._endian + "H", self._buf[2:4])[0]
            frame_len = 2 + 2 + data_len + 2
            if len(self._buf) < frame_len:
                break

            frame = bytes(self._buf[:frame_len])
            del self._buf[:frame_len]

            tail = frame[-1]
            checksum = frame[-2]
            if tail not in ALLOWED_TAILS:
                continue

            calc = sum(frame[:-2]) & 0xFF
            checksum_ok = calc == checksum

            if self._checksum_policy == "strict":
                if self._validate_checksum and not checksum_ok:
                    continue
            elif self._checksum_policy == "compatible":
                if self._validate_checksum and (tail != 0xDD and not checksum_ok):
                    continue
            elif self._checksum_policy == "off":
                pass

            if data_len < 16 or len(frame) < 22:
                continue

            rows = frame[14]
            cols = frame[15]
            payload_len = data_len - 16
            data_start = 20
            data_end = data_start + payload_len
            if data_end > len(frame) - 2:
                continue

            parsed = A010UsbFrame(
                command=frame[4],
                output_mode=frame[5],
                sensor_temp=frame[6],
                driver_temp=frame[7],
                exposure_time=struct.unpack(self._endian + "I", frame[8:12])[0],
                err_code=frame[12],
                reserved1=frame[13],
                rows=rows,
                cols=cols,
                frame_id=struct.unpack(self._endian + "H", frame[16:18])[0],
                isp_version=frame[18],
                reserved3=frame[19],
                payload=frame[data_start:data_end],
                checksum=checksum,
                tail=tail,
            )
            frames.append(parsed)

        return frames


def _decode_http_payload(frame_data: bytes, cfg: FrameConfig):
    deep_data_size, rgb_data_size = struct.unpack("<ii", frame_data[:8])
    frame_payload = frame_data[8:]

    depth_size = (320 * 240 * 2) >> cfg.deep_mode
    depth_img = frame_payload[:depth_size] if depth_size else None
    frame_payload = frame_payload[depth_size:]

    ir_size = (320 * 240 * 2) >> cfg.ir_mode
    ir_img = frame_payload[:ir_size] if ir_size else None
    frame_payload = frame_payload[ir_size:]

    status_size = (320 * 240 // 8) * (
        16 if cfg.status_mode == 0 else 2 if cfg.status_mode == 1 else 8 if cfg.status_mode == 2 else 1
    )
    status_img = frame_payload[:status_size] if status_size else None
    frame_payload = frame_payload[status_size:]

    if deep_data_size != (depth_size + ir_size + status_size):
        raise ValueError("Invalid deep payload size")

    rgb_size = len(frame_payload)
    if rgb_data_size != rgb_size:
        raise ValueError("Invalid RGB payload size")
    rgb_img = frame_payload[:rgb_size] if rgb_size else None

    return depth_img, ir_img, status_img, rgb_img


def decode_http_frame(frame_data: bytes) -> HttpFrame:
    frame_id, stamp_msec = struct.unpack("<QQ", frame_data[0:16])
    cfg = frame_config_decode(frame_data[16:28])
    depth_b, ir_b, status_b, rgb_b = _decode_http_payload(frame_data[28:], cfg)

    depth = (
        np.frombuffer(depth_b, dtype=np.uint16 if cfg.deep_mode == 0 else np.uint8).reshape((240, 320))
        if depth_b
        else None
    )
    ir = (
        np.frombuffer(ir_b, dtype=np.uint16 if cfg.ir_mode == 0 else np.uint8).reshape((240, 320))
        if ir_b
        else None
    )
    status = (
        np.frombuffer(status_b, dtype=np.uint16 if cfg.status_mode == 0 else np.uint8).reshape((240, 320))
        if status_b
        else None
    )

    rgb: Optional[np.ndarray] = None
    if rgb_b:
        if cfg.rgb_mode == 1:
            try:
                import cv2

                jpeg = cv2.imdecode(np.frombuffer(rgb_b, dtype=np.uint8), cv2.IMREAD_COLOR)
                if jpeg is not None:
                    rgb = cv2.cvtColor(jpeg, cv2.COLOR_BGR2RGB)
            except Exception:
                rgb = None
        else:
            try:
                rgb = np.frombuffer(rgb_b, dtype=np.uint8).reshape((480, 640, 3))
            except Exception:
                rgb = None

    return HttpFrame(
        frame_id=frame_id,
        stamp_msec=stamp_msec,
        config=cfg,
        depth=depth,
        ir=ir,
        status=status,
        rgb=rgb,
    )
