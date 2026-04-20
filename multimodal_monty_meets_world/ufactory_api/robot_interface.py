import threading
import time
import copy
import numpy as np
from xarm.wrapper import XArmAPI

class RobotInterface:
    def __init__(self, ip_address):
        self.arm = XArmAPI(ip_address)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        
        # Shared memory for the "Subconscious"
        self._latest_data = {
            "joints": [],
            "end_effector": [],
            "timestamp_s": time.monotonic(),
            "api_status": {
                "joint_code": -1,
                "position_code": -1,
            },
        }
        
        # Threading tools
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def __getstate__(self):
        # Drop live hardware handles that don't survive pickling (XArmAPI C
        # bindings, threading.Lock, threading.Thread). Lets the enclosing
        # experiment config / environment be serialized to disk.
        state = self.__dict__.copy()
        state["arm"] = None
        state["_lock"] = None
        state["_thread"] = None
        state["_running"] = False
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def start_listening(self):
        """Wakes up the subconscious sensory thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        print("Subconscious (Sensor) Thread Started...")

    def stop_listening(self):
        self._running = False
        if self._thread:
            self._thread.join()

    def _monitor_loop(self):
        """The loop that runs in the background forever."""
        while self._running:
            # We use a Lock to make sure we don't interrupt a Move command
            with self._lock:
                # 1. Read Joints
                code_j, angles = self.arm.get_servo_angle(is_radian=True)
                # 2. Read Cartesian Position
                code_p, pos = self.arm.get_position(is_radian=True)

            if code_j == 0 and code_p == 0:
                # Update the shared memory
                self._latest_data = {
                    "joints": angles,
                    "end_effector": pos,
                    "timestamp_s": time.monotonic(),
                    "api_status": {
                        "joint_code": code_j,
                        "position_code": code_p,
                    },
                }
            else:
                self._latest_data["api_status"] = {
                    "joint_code": code_j,
                    "position_code": code_p,
                }
                self._latest_data["timestamp_s"] = time.monotonic()
            
            # Sleep slightly to prevent CPU overheating (100Hz)
            time.sleep(0.01)

    def get_sense_state(self):
        """
        Monty calls this. It returns data INSTANTLY.
        No waiting for the hardware.
        """
        return copy.deepcopy(self._latest_data)

    def move_arm(self, x, y, z, roll, pitch, yaw):
        """
        Send a Cartesian move command safely.
        XYZ fixed angle coordinate system with extrinsic rotations in roll-pitch-yaw order.

        Args:
            x: X position in millimeters, in robot base frame.
            y: Y position in millimeters, in robot base frame.
            z: Z position in millimeters, in robot base frame.
            roll: Rotation about X axis in degrees.
            pitch: Rotation about Y axis in degrees.
            yaw: Rotation about Z axis in degrees.
        """
        print(f"Commanding Move to: {x}, {y}, {z}")
        with self._lock:
            # wait=False is CRITICAL here. 
            # It tells the robot: "Start moving, but give me control of Python back immediately."
            self.arm.set_position(
                x=x,
                y=y,
                z=z,
                roll=roll,
                pitch=pitch,
                yaw=yaw,
                speed=100,
                wait=False,
                is_radian=False,
            )

    def is_api_healthy(self):
        """Return True when the arm appears ready to accept commands."""
        status = self.get_api_health_snapshot()
        if status.get("joint_code", -1) != 0 or status.get("position_code", -1) != 0:
            return False

        error_code = status.get("error_code", 0)
        if error_code not in (0, None):
            return False

        return True

    def get_api_health_snapshot(self):
        """Return latest robot API health status for diagnostics and gating."""
        api_status = self._latest_data.get("api_status", {})
        return {
            "joint_code": api_status.get("joint_code", -1),
            "position_code": api_status.get("position_code", -1),
            "error_code": getattr(self.arm, "error_code", 0),
        }

    def wait_until_ready(self, timeout_s=2.0, poll_interval_s=0.02):
        """Poll robot health until motion API appears ready or timeout elapses."""
        timeout_s = max(0.0, float(timeout_s))
        poll_interval_s = max(0.0, float(poll_interval_s))

        deadline = time.monotonic() + timeout_s
        while True:
            if self.is_api_healthy():
                return True

            if time.monotonic() >= deadline:
                return False

            if poll_interval_s > 0.0:
                time.sleep(poll_interval_s)

    def get_joint_limit_margin_rad(self, joint_limits_rad):
        """Return minimum absolute distance to nearest configured joint limit."""
        joints = np.asarray(self._latest_data.get("joints", []), dtype=float)
        limits = np.asarray(joint_limits_rad, dtype=float)

        if joints.size == 0 or limits.ndim != 2 or limits.shape[1] != 2:
            return float("inf")

        check_count = min(joints.shape[0], limits.shape[0])
        lower = limits[:check_count, 0]
        upper = limits[:check_count, 1]
        active_joints = joints[:check_count]

        margin_to_lower = active_joints - lower
        margin_to_upper = upper - active_joints
        return float(np.min(np.minimum(margin_to_lower, margin_to_upper)))

    def configure_payload(self, mass_kg, center_of_gravity_mm):
        """Set payload metadata for safer dynamics and limit handling."""
        cog_mm = np.asarray(center_of_gravity_mm, dtype=float).tolist()
        with self._lock:
            return_code = self.arm.set_tcp_load(weight=mass_kg, center_of_gravity=cog_mm)
        return return_code == 0

    def is_target_pose_feasible(self, x, y, z, roll, pitch, yaw):
        """Best-effort IK feasibility query for a Cartesian target.

        Returns True when no explicit IK API is available, allowing deployment to
        proceed while still supporting feasibility checks when the SDK exposes one.
        """
        if not hasattr(self.arm, "get_inverse_kinematics"):
            return True

        try:
            with self._lock:
                code, _angles = self.arm.get_inverse_kinematics(
                    pose=[x, y, z, roll, pitch, yaw],
                    input_is_radian=False,
                    return_is_radian=True,
                )
            return code == 0
        except TypeError:
            # Keep compatibility with SDK variants that use a different signature.
            return True

    def stop_motion(self, reason):
        """Stop or pause motion when safety checks reject a command."""
        print(f"Safety stop requested: {reason}")
        with self._lock:
            if hasattr(self.arm, "emergency_stop"):
                self.arm.emergency_stop()
            else:
                self.arm.set_state(4)

    def graceful_stop(self):
        """Pause motion and drain queued commands without requiring manual reset.

        Uses ``set_state(4)`` (paused) rather than ``emergency_stop()`` so the arm
        can be resumed without operator intervention. Intended for normal shutdown
        and KeyboardInterrupt paths where queued commands must be cancelled
        immediately.
        """
        with self._lock:
            try:
                self.arm.set_state(4)
            except Exception as exc:
                print(f"graceful_stop: set_state(4) failed: {exc}")