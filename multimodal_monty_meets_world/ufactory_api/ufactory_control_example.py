# arm.connect(port='192.168.1.159')
from xarm.wrapper import XArmAPI

def command_move_to_target(target):
    x, y, z = target['command_mm']
    roll, pitch, yaw = target['command_euler_deg']
    arm.set_position(x=x, y=y, z=z, roll=roll, pitch=pitch, yaw=yaw, speed=100, wait=True)


# 1. Connect to the robot
arm = XArmAPI('192.168.1.159') 

# 2. Clear errors and enable motion
arm.motion_enable(enable=True)
arm.set_mode(0)  # Mode 0 = Position Control Mode
arm.set_state(state=0) # State 0 = Ready

# print("Moving to Home Position...")
# arm.set_position(x=200, y=1.7, z=199, roll=180, pitch=0, yaw=0, speed=100, wait=True)
# arm.move_gohome(wait=True)

# 3. Move to a specific coordinate (x, y, z, roll, pitch, yaw)
# Units are in millimeters (mm) and degrees
print("Moving to Target...")
target = {'command_mm': [190.473, 63.502, 169.464], 'command_euler_deg': [172.946, -27.441, 32.384]}
command_move_to_target(target)

# 5. Disconnect safely
arm.disconnect()
print("Done.")