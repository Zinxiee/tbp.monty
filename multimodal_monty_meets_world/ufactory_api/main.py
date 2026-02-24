import time
from robot_interface import RobotInterface

robot = RobotInterface('192.168.1.159')
robot.start_listening()

try:
    while True:
        # STEP A: PERCEIVE (Instant)
        # This takes 0.00001 seconds because the data is already in memory
        current_state = robot.get_sense_state()
        
        # Feed this into Monty...
        print(f"Sensory Input: {current_state['end_effector']}")
        
        # STEP B: DECIDE & ACT
        # Let's say Monty decides to move slightly
        # We send the command and IMMEDIATELY loop back to sensing
        # We do NOT wait for the move to finish
        robot.move_arm(300, 0, 200, 180, 0, 0) # Monty commands a new position

        time.sleep(0.5) # Representing "Thinking" time

except KeyboardInterrupt:
    robot.stop_listening()
    robot.arm.disconnect()