from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .usb_client import MaixsenseA010USB


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MaixSense A010 high-level USB capture utility"
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port, e.g. /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=921600, help="Serial baudrate")
    parser.add_argument(
        "--checksum-policy",
        choices=["compatible", "strict", "off"],
        default="compatible",
        help="USB frame checksum acceptance policy",
    )
    parser.add_argument("--seconds", type=float, default=10.0, help="Capture duration in seconds")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")

    parser.add_argument("--fps", type=int, default=19, help="AT+FPS value")
    parser.add_argument("--binning", type=int, choices=[1, 2, 4], default=1, help="AT+BINN value")
    parser.add_argument("--unit", type=int, default=0, help="AT+UNIT value (0-10)")
    parser.add_argument("--display", type=int, default=3, help="AT+DISP bitmask (1=LCD,2=USB,4=UART)")

    parser.add_argument("--isp", type=int, choices=[0, 1], default=1, help="AT+ISP")
    parser.add_argument("--anti-mmi", type=int, choices=[0, 1], default=1, help="AT+ANTIMMI")
    parser.add_argument("--ae", type=int, choices=[0, 1], default=1, help="AT+AE")
    parser.add_argument("--ev", type=int, default=0, help="AT+EV")

    parser.add_argument("--cmd", action="append", default=[], help="Additional raw AT command (repeatable)")
    parser.add_argument("--skip-config", action="store_true", help="Do not send stream/config commands")

    parser.add_argument("--output-dir", default="./a010_capture", help="Directory for outputs")
    parser.add_argument("--save-npy", action="store_true", help="Save each frame payload as .npy")
    parser.add_argument("--save-png", action="store_true", help="Save each frame index image as .png")
    parser.add_argument("--save-csv", action="store_true", help="Write capture metadata CSV")
    return parser.parse_args()


def _ensure_output_dir(path: str) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_png(path: Path, img: np.ndarray) -> bool:
    try:
        from PIL import Image

        Image.fromarray(img).save(path)
        return True
    except Exception:
        return False


def run_capture(args: argparse.Namespace) -> None:
    output_dir = _ensure_output_dir(args.output_dir)

    csv_writer: Optional[csv.writer] = None
    csv_fp = None
    if args.save_csv:
        csv_fp = (output_dir / "frames.csv").open("w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_fp)
        csv_writer.writerow(
            [
                "host_time_ms",
                "frame_id",
                "rows",
                "cols",
                "payload_len",
                "sensor_temp",
                "driver_temp",
                "exposure_time",
                "output_mode",
                "err_code",
                "center_idx",
                "center_mm",
            ]
        )

    png_available = True
    frame_count = 0
    start = time.monotonic()

    try:
        with MaixsenseA010USB(
            port=args.port,
            baudrate=args.baud,
            checksum_policy=args.checksum_policy,
        ) as sensor:
            if not args.skip_config:
                sensor.configure_stream(
                    fps=args.fps,
                    display_mode=args.display,
                    anti_mmi=bool(args.anti_mmi),
                    binning=args.binning,
                    unit=args.unit,
                    isp=bool(args.isp),
                    ae=bool(args.ae),
                    ev=args.ev,
                )
                for cmd in args.cmd:
                    sensor.send_command(cmd)

            for frame in sensor.iter_frames(timeout_s=args.seconds):
                frame_count += 1
                idx_img = frame.depth_index_image()
                mm_img = frame.distance_mm_image(unit=args.unit)
                center_r = frame.rows // 2
                center_c = frame.cols // 2
                center_idx = int(idx_img[center_r, center_c])
                center_mm = float(mm_img[center_r, center_c])

                ts_ms = int(time.time() * 1000)
                stem = f"f{frame.frame_id:06d}_{ts_ms}"

                if args.save_npy:
                    np.save(output_dir / f"{stem}.npy", idx_img)
                if args.save_png and png_available:
                    png_available = _save_png(output_dir / f"{stem}.png", idx_img)

                if csv_writer is not None:
                    csv_writer.writerow(
                        [
                            ts_ms,
                            frame.frame_id,
                            frame.rows,
                            frame.cols,
                            len(frame.payload),
                            frame.sensor_temp,
                            frame.driver_temp,
                            frame.exposure_time,
                            frame.output_mode,
                            frame.err_code,
                            center_idx,
                            round(center_mm, 3),
                        ]
                    )

                print(
                    f"frame={frame.frame_id} shape={frame.shape} center_idx={center_idx} center_mm={center_mm:.2f}"
                )

                if args.max_frames > 0 and frame_count >= args.max_frames:
                    break

    finally:
        if csv_fp is not None:
            csv_fp.close()

    elapsed = max(1e-6, time.monotonic() - start)
    print(f"Captured {frame_count} frames in {elapsed:.2f}s ({frame_count/elapsed:.2f} FPS)")
    if args.save_png and not png_available:
        print("PNG export was requested but Pillow is not available; .png files were not written.")


def main() -> None:
    args = _parse_args()
    run_capture(args)


if __name__ == "__main__":
    main()
