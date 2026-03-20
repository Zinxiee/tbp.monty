import time
import numpy as np
import quaternion as qt

from tbp.monty.frameworks.models.motor_policies import MotorPolicyResult

from monty_goal_adapter import (
    MontyGoalToRobotAdapter,
    identity_world_to_robot_transform,
)
from robot_interface import RobotInterface

robot = RobotInterface('192.168.1.159')
goal_adapter = MontyGoalToRobotAdapter(
    robot=robot,
    world_to_robot=identity_world_to_robot_transform(),
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
            # Fallback behavior for results without absolute goals.
            robot.move_arm(300, 0, 200, 180, 0, 0)

        time.sleep(0.5) # Representing "Thinking" time

except KeyboardInterrupt:
    robot.stop_listening()
    robot.arm.disconnect()