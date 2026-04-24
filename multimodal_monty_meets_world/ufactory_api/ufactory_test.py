import time
import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult

from monty_goal_adapter import (
    EulerConvention,
    Link6ToSensorTransform,
    MontyGoalToRobotAdapter,
    SafetyConfig,
    identity_world_to_robot_transform,
)
from robot_interface import RobotInterface

robot = RobotInterface('192.168.1.159')

# Measured Link6 -> ToF optical-center offset from deployment calibration.
link6_to_sensor = Link6ToSensorTransform(
    translation_m=np.array([0.0, -0.060, 0.0135], dtype=float),
    rotation_quat_wxyz=qt.one,
)

safety_config = SafetyConfig(
    workspace_min_xyz_m=np.array([0.10, -0.35, 0.02], dtype=float),
    workspace_max_xyz_m=np.array([0.70, 0.35, 0.45], dtype=float),
    max_translation_step_m=0.08,
    max_rotation_step_deg=20.0,
    orientation_min_euler_deg=np.array([-180.0, -120.0, -180.0], dtype=float),
    orientation_max_euler_deg=np.array([180.0, 120.0, 180.0], dtype=float),
    payload_mass_kg=0.056,
    payload_center_of_gravity_mm=np.array([0.0, -60.0, 13.5], dtype=float),
)

goal_adapter = MontyGoalToRobotAdapter(
    robot=robot,
    world_to_robot=identity_world_to_robot_transform(),
    link6_to_sensor=link6_to_sensor,
    safety_config=safety_config,
    euler_convention=EulerConvention(sequence="xyz", degrees=True),
)
robot.start_listening()

try:
    while True:
        # STEP A: PERCEIVE (Instant)
        # This takes 0.00001 seconds because the data is already in memory
        current_state = robot.get_sense_state()
        
        # Feed this into Monty...
        print(f"Sensory Input: {current_state['end_effector']}")
        
        # STEP B: DECIDE & ACT
        # Example: command from policy result in world coordinates.
        # In production, this object comes directly from the motor policy.
        world_goal_result = MotorPolicyResult(
            goal_pose=(np.array([0.30, 0.0, 0.20], dtype=float), qt.one)
        )

        # Adapter converts world-frame meters + quaternion to
        # robot-base millimeters + roll/pitch/yaw degrees.
        if not goal_adapter.dispatch_motor_policy_result(world_goal_result):
            print("No safe absolute goal to dispatch on this step.")

        time.sleep(0.5) # Representing "Thinking" time

except KeyboardInterrupt:
    robot.stop_listening()
    robot.arm.disconnect()