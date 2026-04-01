"""
ZED 2i RGBD Capture Script

Captures RGB-D images from a ZED 2i stereo camera with optional live streaming
and sensor metadata export including spatial location and orientation.

BASIC USAGE:
    # Capture single frame to specified directory
    python capture_rgbd.py --output-dir ./my_captures

    # Capture with custom filename prefix
    python capture_rgbd.py --output-dir ./data --basename frame_001

STREAMING OPTIONS:
    # Local OpenCV preview window (press 'c' to capture, 'q' to quit)
    python capture_rgbd.py --output-dir ./captures --stream

    # Network MJPEG stream (view at http://localhost:8080/)
    python capture_rgbd.py --output-dir ./captures --stream-network

    # Both local and network streaming on custom port
    python capture_rgbd.py --output-dir ./captures --stream --stream-network --stream-port 9000

METADATA OPTIONS:
    # Capture with pose data (position + orientation in WORLD frame)
    python capture_rgbd.py --output-dir ./captures --metadata pose

    # Capture with all sensor data (IMU, magnetometer, barometer, temperature)
    python capture_rgbd.py --output-dir ./captures --metadata sensors

    # Capture with all available metadata
    python capture_rgbd.py --output-dir ./captures --metadata all

    # No metadata
    python capture_rgbd.py --output-dir ./captures --metadata none

CAMERA CONFIGURATION:
    # High resolution with quality depth mode
    python capture_rgbd.py --output-dir ./captures --resolution HD1080 --depth-mode QUALITY

    # Fast capture with lower resolution
    python capture_rgbd.py --output-dir ./captures --resolution VGA --depth-mode PERFORMANCE --fps 60

    # Depth units in millimeters instead of meters
    python capture_rgbd.py --output-dir ./captures --units MILLIMETER

COMPLETE EXAMPLE:
    # Network stream + high quality + full metadata
    python capture_rgbd.py \\
        --output-dir ./production_captures \\
        --basename robot_view \\
        --stream-network \\
        --stream-port 8080 \\
        --resolution HD1080 \\
        --depth-mode NEURAL_PLUS \\
        --fps 30 \\
        --units METER \\
        --metadata all

OUTPUT FILES:
    {basename}_rgb.png          - RGB image from left camera
    {basename}_depth.npy        - Raw depth map (NumPy array)
    {basename}_depth_vis.png    - Colorized depth visualization
    {basename}_metadata.json    - Camera info, pose, and sensor data (if enabled)

METADATA STRUCTURE:
    {
        "capture_time": "2026-02-20T15:30:45.123456",
        "camera": {
            "serial_number": 12345678,
            "model": "ZED 2i",
            "resolution": {"width": 1280, "height": 720},
            "fps": 30
        },
        "pose": {
            "translation_xyz": [x, y, z],
            "orientation": {
                "quaternion_xyzw": [x, y, z, w],
                "euler_angles": {
                    "roll_deg": ..., "pitch_deg": ..., "yaw_deg": ...,
                    "roll_rad": ..., "pitch_rad": ..., "yaw_rad": ...
                }
            },
            "pose_confidence": 100
        },
        "sensors": {
            "imu": {"linear_acceleration": [...], "angular_velocity": [...]},
            "magnetometer": {"magnetic_field_calibrated": [...]},
            "barometer": {"pressure": ...},
            "temperature": {...}
        }
    }
"""

import argparse
import json
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pyzed.sl as sl


RESOLUTION_MAP = {
    "HD2K": sl.RESOLUTION.HD2K,
    "HD1200": sl.RESOLUTION.HD1200,
    "HD1080": sl.RESOLUTION.HD1080,
    "HD720": sl.RESOLUTION.HD720,
    "SVGA": sl.RESOLUTION.SVGA,
    "VGA": sl.RESOLUTION.VGA,
}

DEPTH_MODE_MAP = {
    "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
    "QUALITY": sl.DEPTH_MODE.QUALITY,
    "ULTRA": sl.DEPTH_MODE.ULTRA,
    "NEURAL": sl.DEPTH_MODE.NEURAL,
    "NEURAL_PLUS": sl.DEPTH_MODE.NEURAL_PLUS,
}

UNIT_MAP = {
    "METER": sl.UNIT.METER,
    "MILLIMETER": sl.UNIT.MILLIMETER,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one RGBD frame from a ZED 2i and save optional metadata."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="",
        help="Base filename for saved outputs. Defaults to timestamp.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=list(RESOLUTION_MAP.keys()),
        default="HD720",
        help="Camera resolution.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS.",
    )
    parser.add_argument(
        "--depth-mode",
        type=str,
        choices=list(DEPTH_MODE_MAP.keys()),
        default="NEURAL",
        help="Depth quality/performance mode.",
    )
    parser.add_argument(
        "--units",
        type=str,
        choices=list(UNIT_MAP.keys()),
        default="METER",
        help="Depth units used in outputs and metadata.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Show live preview stream before capture. Press 'c' to capture, 'q' to quit.",
    )
    parser.add_argument(
        "--stream-network",
        action="store_true",
        help="Enable MJPEG network stream on --stream-port (view at http://localhost:PORT).",
    )
    parser.add_argument(
        "--stream-port",
        type=int,
        default=8080,
        help="Port for network streaming server.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        choices=["none", "pose", "sensors", "all"],
        default="none",
        help="Metadata to include in JSON. 'pose' includes camera spatial location.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print stage-by-stage diagnostics for camera open/grab/close.",
    )
    return parser.parse_args()


def zed_timestamp_to_iso(ts: sl.Timestamp) -> str:
    try:
        return datetime.fromtimestamp(ts.get_microseconds() / 1_000_000.0).isoformat()
    except Exception:
        return datetime.now().isoformat()


def to_list(value):
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return value


def depth_to_vis(depth_array: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_array) & (depth_array > 0)
    if not np.any(valid):
        return np.zeros((*depth_array.shape, 3), dtype=np.uint8)
    low = np.percentile(depth_array[valid], 2)
    high = np.percentile(depth_array[valid], 98)
    if high <= low:
        high = low + 1e-6
    scaled = np.zeros(depth_array.shape, dtype=np.float32)
    scaled[valid] = np.clip((depth_array[valid] - low) / (high - low), 0.0, 1.0)
    vis = np.zeros(depth_array.shape, dtype=np.uint8)
    vis[valid] = np.rint(scaled[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)


def grab_once(zed: sl.Camera, left: sl.Mat, depth: sl.Mat, runtime: sl.RuntimeParameters) -> bool:
    if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
        return False
    zed.retrieve_image(left, sl.VIEW.LEFT)
    zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
    return True


def quaternion_to_euler(quat: list) -> dict:
    """Convert quaternion [x, y, z, w] to Euler angles (roll, pitch, yaw) in degrees."""
    x, y, z, w = quat
    
    # Roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    # Pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.copysign(np.pi / 2, sinp)
    else:
        pitch = np.arcsin(sinp)
    
    # Yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return {
        "roll_deg": float(np.degrees(roll)),
        "pitch_deg": float(np.degrees(pitch)),
        "yaw_deg": float(np.degrees(yaw)),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "yaw_rad": float(yaw),
    }


def collect_pose(zed: sl.Camera):
    pose = sl.Pose()
    status = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
    if status != sl.POSITIONAL_TRACKING_STATE.OK:
        return {"status": str(status)}

    translation = sl.Translation()
    pose.get_translation(translation)

    orientation = pose.get_orientation()
    quat = to_list(orientation.get())
    euler = quaternion_to_euler(quat)
    
    return {
        "status": str(status),
        "timestamp": zed_timestamp_to_iso(pose.timestamp),
        "translation_xyz": to_list(translation.get()),
        "orientation": {
            "quaternion_xyzw": quat,
            "euler_angles": euler,
        },
        "pose_confidence": int(pose.pose_confidence),
    }


def collect_sensors(zed: sl.Camera):
    sensors = sl.SensorsData()
    status = zed.get_sensors_data(sensors, sl.TIME_REFERENCE.CURRENT)
    if status != sl.ERROR_CODE.SUCCESS:
        return {"status": str(status)}

    imu = sensors.get_imu_data()
    magnetic = sensors.get_magnetometer_data()
    barometer = sensors.get_barometer_data()
    temperature = sensors.get_temperature_data()

    payload = {
        "status": str(status),
        "timestamp": zed_timestamp_to_iso(sensors.get_timestamp(sl.TIME_REFERENCE.CURRENT)),
        "imu": {
            "linear_acceleration": to_list(imu.get_linear_acceleration()),
            "angular_velocity": to_list(imu.get_angular_velocity()),
        },
        "magnetometer": {
            "magnetic_field_calibrated": to_list(magnetic.get_magnetic_field_calibrated()),
        },
        "barometer": {
            "pressure": float(barometer.pressure) if hasattr(barometer, "pressure") else None,
        },
        "temperature": {},
    }

    for sensor_name in [
        "IMU",
        "BAROMETER",
        "ONBOARD_LEFT",
        "ONBOARD_RIGHT",
    ]:
        try:
            sensor_enum = getattr(sl.SENSOR_LOCATION, sensor_name)
            payload["temperature"][sensor_name] = float(temperature.get(sensor_enum))
        except Exception:
            payload["temperature"][sensor_name] = None

    return payload


class MJPEGStreamHandler(BaseHTTPRequestHandler):
    """HTTP handler for MJPEG streaming."""
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            try:
                while True:
                    if self.server.current_frame is None:
                        continue
                    _, jpeg = cv2.imencode('.jpg', self.server.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.wfile.write(b"--jpgboundary\r\n")
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(len(jpeg)))
                    self.end_headers()
                    self.wfile.write(jpeg.tobytes())
                    self.wfile.write(b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)
    
    def log_message(self, format, *args):
        pass  # Suppress request logging


class MJPEGServer(HTTPServer):
    """HTTP server with shared frame buffer for MJPEG streaming."""
    
    def __init__(self, server_address, handler_class):
        super().__init__(server_address, handler_class)
        self.current_frame: Optional[np.ndarray] = None


def start_mjpeg_server(port: int) -> MJPEGServer:
    """Start MJPEG server in background thread."""
    server = MJPEGServer(('0.0.0.0', port), MJPEGStreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"MJPEG stream available at http://localhost:{port}/")
    return server


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.basename or datetime.now().strftime("rgbd_%Y%m%d_%H%M%S")

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = RESOLUTION_MAP[args.resolution]
    init.camera_fps = args.fps
    init.depth_mode = DEPTH_MODE_MAP[args.depth_mode]
    init.coordinate_units = UNIT_MAP[args.units]
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP

    if args.debug:
        print("[DEBUG] Opening ZED camera...")
    open_status = zed.open(init)
    if open_status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open camera: {open_status}")
    if args.debug:
        print("[DEBUG] ZED camera opened.")

    positional_tracking_enabled = False
    tracking_requested = args.metadata in {"pose", "all"}
    mjpeg_server = None
    left = None
    depth = None

    try:
        if tracking_requested:
            if args.debug:
                print("[DEBUG] Enabling positional tracking...")
            track_params = sl.PositionalTrackingParameters()
            track_status = zed.enable_positional_tracking(track_params)
            if track_status == sl.ERROR_CODE.SUCCESS:
                positional_tracking_enabled = True
                if args.debug:
                    print("[DEBUG] Positional tracking enabled.")
            else:
                print(f"Warning: positional tracking not available: {track_status}")

        runtime = sl.RuntimeParameters()
        left = sl.Mat()
        depth = sl.Mat()

        if args.stream_network:
            mjpeg_server = start_mjpeg_server(args.stream_port)

        captured = False

        if args.stream or args.stream_network:
            display_local = args.stream
            if display_local:
                print("Streaming preview. Press 'c' to capture, 'q' to quit.")
            else:
                print("Network streaming active. Press Ctrl+C or send 'c' to capture.")

            while True:
                if not grab_once(zed, left, depth, runtime):
                    continue

                left_rgba = left.get_data()
                left_bgr = cv2.cvtColor(left_rgba, cv2.COLOR_BGRA2BGR)
                depth_arr = depth.get_data().copy()
                depth_vis = depth_to_vis(depth_arr)
                preview = np.hstack([left_bgr, depth_vis])

                if mjpeg_server:
                    mjpeg_server.current_frame = preview

                if display_local:
                    cv2.imshow("ZED RGBD Stream (Left | Depth)", preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):
                        captured = True
                        break
                    if key == ord("q"):
                        break
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("c"):
                        captured = True
                        break
        else:
            if args.debug:
                print("[DEBUG] Grabbing single frame...")
            captured = grab_once(zed, left, depth, runtime)

        if not captured:
            print("No frame captured.")
            return

        left_rgba = left.get_data()
        left_bgr = cv2.cvtColor(left_rgba, cv2.COLOR_BGRA2BGR)
        depth_arr = depth.get_data().copy()
        depth_vis = depth_to_vis(depth_arr)

        rgb_path = output_dir / f"{stem}_rgb.png"
        depth_npy_path = output_dir / f"{stem}_depth.npy"
        depth_vis_path = output_dir / f"{stem}_depth_vis.png"

        cv2.imwrite(str(rgb_path), left_bgr)
        np.save(depth_npy_path, depth_arr)
        cv2.imwrite(str(depth_vis_path), depth_vis)

        print(f"Saved RGB image: {rgb_path}")
        print(f"Saved depth map (NumPy): {depth_npy_path}")
        print(f"Saved depth visualization: {depth_vis_path}")

        if args.metadata != "none":
            camera_info = zed.get_camera_information()
            metadata = {
                "capture_time": datetime.now().isoformat(),
                "camera": {
                    "serial_number": int(camera_info.serial_number),
                    "model": str(camera_info.camera_model),
                    "firmware_version": int(camera_info.camera_configuration.firmware_version),
                    "resolution": {
                        "width": int(camera_info.camera_configuration.resolution.width),
                        "height": int(camera_info.camera_configuration.resolution.height),
                    },
                    "fps": int(camera_info.camera_configuration.fps),
                },
                "units": args.units,
                "files": {
                    "rgb": str(rgb_path),
                    "depth_npy": str(depth_npy_path),
                    "depth_vis": str(depth_vis_path),
                },
            }

            if args.metadata in {"pose", "all"}:
                metadata["pose"] = collect_pose(zed)
            if args.metadata in {"sensors", "all"}:
                metadata["sensors"] = collect_sensors(zed)

            metadata_path = output_dir / f"{stem}_metadata.json"
            with metadata_path.open("w", encoding="utf-8") as file:
                json.dump(metadata, file, indent=2)
            print(f"Saved metadata: {metadata_path}")
    finally:
        if args.debug:
            print("[DEBUG] Cleaning up resources...")
        cv2.destroyAllWindows()
        if mjpeg_server:
            mjpeg_server.shutdown()
            mjpeg_server.server_close()
        if positional_tracking_enabled:
            zed.disable_positional_tracking()
        if left is not None:
            left.free(sl.MEM.CPU)
        if depth is not None:
            depth.free(sl.MEM.CPU)
        zed.close()
        # Give USB stack/SDK a brief moment to fully release camera handle.
        time.sleep(0.25)
        if args.debug:
            print("[DEBUG] Cleanup complete.")


if __name__ == "__main__":
    main()