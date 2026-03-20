import threading
import time
import copy
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
            "end_effector": []
        }
        
        # Threading tools
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

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
                    "end_effector": pos
                }
            
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
            self.arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=100, wait=False)