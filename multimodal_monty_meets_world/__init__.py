"""Multimodal Monty Meets World hardware integration.

Hardware interface modules for real-world robot and sensor control:
- ufactory_api: UFactory Lite6 robot interface and goal-pose adapter
- maixsense_a010_api: Maixsense A010 ToF sensor control and observation
- factory: Hydra-compatible constructors for hardware objects
"""

from .factory import (
    create_goal_adapter,
    create_observation_adapter,
    create_robot_interface,
    create_usb_frame_client,
)

__all__ = [
    "create_goal_adapter",
    "create_observation_adapter",
    "create_robot_interface",
    "create_usb_frame_client",
]
