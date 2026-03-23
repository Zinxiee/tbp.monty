from __future__ import annotations

"""Phase-1 Maixsense -> CameraSM wiring example.

This script demonstrates the integration path where Maixsense depth frames are
adapted to CameraSM-compatible observations via `MaixsenseMontyObservationAdapter`.

It intentionally avoids constructing a real RuntimeContext or CameraSM instance,
because those are experiment-specific. Instead, it prints a concise summary of the
adapted observation payload you can pass into `camera_sm.step(ctx, observation)`.
"""

import argparse
import time

import numpy as np

from maixsense_a010_api import (
    MaixsenseA010HTTP,
    MaixsenseA010USB,
    create_adapter_from_http_calibration,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Maixsense -> CameraSM adapter demo")
    parser.add_argument("--port", default="/dev/sipeed")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--http-host", default="192.168.233.1")
    parser.add_argument("--http-port", type=int, default=80)
    parser.add_argument("--unit", type=int, default=0)
    args = parser.parse_args()

    http_client = MaixsenseA010HTTP(args.http_host, args.http_port)
    adapter = create_adapter_from_http_calibration(http_client)

    with MaixsenseA010USB(port=args.port, baudrate=args.baud) as sensor:
        sensor.set_display(usb=True, lcd=True, uart=False)

        start = time.monotonic()
        count = 0
        for frame in sensor.iter_frames(timeout_s=args.seconds):
            count += 1
            observation = adapter.from_usb_frame(
                frame,
                world_camera=np.eye(4),
                unit=args.unit,
            )

            # At runtime, this is where you call:
            # observed_state = camera_sm.step(ctx, observation)
            print(
                "frame=%s depth=%s semantic_3d=%s sensor_frame_data=%s"
                % (
                    frame.frame_id,
                    observation["depth"].shape,
                    observation["semantic_3d"].shape,
                    observation["sensor_frame_data"].shape,
                )
            )

        elapsed = max(1e-6, time.monotonic() - start)
        print("Adapted %d frames in %.2fs (%.2f FPS)" % (count, elapsed, count / elapsed))


if __name__ == "__main__":
    main()
