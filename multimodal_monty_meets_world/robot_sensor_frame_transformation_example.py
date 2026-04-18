import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from ufactory_api.robot_interface import RobotInterface


def make_T(t_xyz, R_3x3):
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_3x3
    T[:3, 3] = np.asarray(t_xyz, dtype=float)
    return T


def inv_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=float)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


def quat_wxyz_to_R(q_wxyz):
    w, x, y, z = q_wxyz
    return R.from_quat([x, y, z, w]).as_matrix()


def print_pose(label, T_ref_frame):
    t = T_ref_frame[:3, 3]
    Rm = T_ref_frame[:3, :3]
    print(f"\n{label}")
    print("  origin [m]:", np.round(t, 6).tolist())
    print("  x-axis in ref:", np.round(Rm[:, 0], 6).tolist())
    print("  y-axis in ref:", np.round(Rm[:, 1], 6).tolist())
    print("  z-axis in ref:", np.round(Rm[:, 2], 6).tolist())


# -----------------------------
# 1) Read live Link6 pose in BASE frame
# -----------------------------
robot = RobotInterface("192.168.1.159")
robot.start_listening()
time.sleep(0.1)  # allow listener to populate at least one sample
state = robot.get_sense_state()

x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad = state["end_effector"][:6]
t_base_link6 = np.array([x_mm, y_mm, z_mm], dtype=float) / 1000.0
R_base_link6 = R.from_euler("xyz", [roll_rad, pitch_rad, yaw_rad], degrees=False).as_matrix()
T_base_link6 = make_T(t_base_link6, R_base_link6)

# -----------------------------
# 2) Config transforms from your YAML
# -----------------------------
# world_to_robot == world_to_base (robot base frame)
world_to_robot_translation_m = np.array([0.0, 0.0, 0.0], dtype=float)
world_to_robot_quat_wxyz = [0.5, 0.5, -0.5, -0.5]  # w,x,y,z
R_base_world = quat_wxyz_to_R(world_to_robot_quat_wxyz)
T_base_world = make_T(world_to_robot_translation_m, R_base_world)

# link6_to_sensor from YAML
link6_to_sensor_translation_m = np.array([0.060, 0.0, 0.0135], dtype=float)
link6_to_sensor_quat_wxyz = [0.0, 0.0, 1.0, 0.0]  # w,x,y,z
R_link6_sensor = quat_wxyz_to_R(link6_to_sensor_quat_wxyz)
T_link6_sensor = make_T(link6_to_sensor_translation_m, R_link6_sensor)

# -----------------------------
# 3) Compose BASE-frame transforms
# -----------------------------
T_base_base = np.eye(4, dtype=float)
T_base_sensor = T_base_link6 @ T_link6_sensor

# -----------------------------
# 4) Convert everything into WORLD frame
# -----------------------------
T_world_base = inv_T(T_base_world)
T_world_world = np.eye(4, dtype=float)
T_world_link6 = T_world_base @ T_base_link6
T_world_sensor = T_world_base @ T_base_sensor

# -----------------------------
# 5) Print all frames in both references
# -----------------------------
print("\n=== Frames expressed in ROBOT BASE frame ===")
print_pose("BASE in BASE (T_base_base)", T_base_base)
print_pose("WORLD in BASE (T_base_world)", T_base_world)
print_pose("LINK6 in BASE (T_base_link6)", T_base_link6)
print_pose("SENSOR in BASE (T_base_sensor)", T_base_sensor)

print("\n=== Frames expressed in WORLD / MONTY frame ===")
print_pose("WORLD in WORLD (T_world_world)", T_world_world)
print_pose("BASE in WORLD (T_world_base)", T_world_base)
print_pose("LINK6 in WORLD (T_world_link6)", T_world_link6)
print_pose("SENSOR in WORLD (T_world_sensor)", T_world_sensor)

# Optional: compare against env logs
print("\nExpected match with env logs:")
print("  PROPRIO_SENSOR_POSE.sensor_position_m / WORLD_CAMERA t ~= T_world_sensor[:3,3]")
print("  value:", np.round(T_world_sensor[:3, 3], 6).tolist())

robot.stop_listening()