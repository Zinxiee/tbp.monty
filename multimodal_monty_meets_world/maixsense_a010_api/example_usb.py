from __future__ import annotations

import argparse
import time

from maixsense_a010_api import MaixsenseA010USB


def main() -> None:
    parser = argparse.ArgumentParser(description="Maixsense A010 USB stream example")
    parser.add_argument("--port", default="/dev/sipeed")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=19)
    args = parser.parse_args()

    with MaixsenseA010USB(port=args.port, baudrate=args.baud) as sensor:
        sensor.set_display(usb=True, lcd=True, uart=False)
        sensor.set_fps(args.fps)
        sensor.set_unit(0)

        start = time.monotonic()
        count = 0
        for frame in sensor.iter_frames(timeout_s=args.seconds):
            count += 1
            img = frame.depth_index_image()
            center = img[img.shape[0] // 2, img.shape[1] // 2]
            print(
                f"frame={frame.frame_id} shape={frame.shape} center_idx={center} "
                f"sensor_temp={frame.sensor_temp} driver_temp={frame.driver_temp}"
            )

        elapsed = max(1e-6, time.monotonic() - start)
        print(f"Received {count} frames in {elapsed:.2f}s ({count / elapsed:.2f} FPS)")


if __name__ == "__main__":
    main()
