from dataclasses import dataclass


@dataclass
class TaskSpacePose:
    """
    Describe a target end-effector pose in task space.

    Inputs:
        x: Target x position in meters.
        y: Target y position in meters.
        z: Target z position in meters.
        roll: Target roll angle in radians.
        pitch: Target pitch angle in radians.
        yaw: Target yaw angle in radians.

    Returns:
        None
    """
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float