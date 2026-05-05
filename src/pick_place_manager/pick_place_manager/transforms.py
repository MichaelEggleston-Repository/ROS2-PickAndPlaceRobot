import numpy as np
from scipy.spatial.transform import Rotation
from pick_place_interfaces.msg import TaskSpacePose

def transform_from_translation_quaternion(
    translation_xyz: list[float],
    quaternion_xyzw: list[float],
) -> np.ndarray:
    rotation_matrix = Rotation.from_quat(quaternion_xyzw).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_xyz

    return transform

def transform_from_translation_rpy(
    translation_xyz: list[float],
    rpy: list[float],
) -> np.ndarray:
    rotation_matrix = Rotation.from_euler("xyz", rpy).as_matrix()

    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = translation_xyz

    return transform

def translation_quaternion_from_transform(transform: np.ndarray) -> tuple[list[float], list[float]]:
    translation_xyz = transform[:3, 3].tolist()
    quaternion_xyzw = Rotation.from_matrix(transform[:3, :3]).as_quat().tolist()

    return translation_xyz, quaternion_xyzw

def translation_rpy_from_transform(transform: np.ndarray) -> tuple[list[float], list[float]]:
    translation_xyz = transform[:3, 3].tolist()
    rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz").tolist()

    return translation_xyz, rpy

def task_space_pose_from_translation_rpy(
    translation_xyz: list[float],
    rpy: list[float],
) -> TaskSpacePose:
    pose = TaskSpacePose()

    pose.x = translation_xyz[0]
    pose.y = translation_xyz[1]
    pose.z = translation_xyz[2]
    pose.roll = rpy[0]
    pose.pitch = rpy[1]
    pose.yaw = rpy[2]

    return pose

def task_space_pose_from_translation_quaternion(
    translation_xyz: list[float],
    quaternion_xyzw: list[float],
) -> TaskSpacePose:
    # TaskSpacePose uses flat RPY fields, not nested geometry_msgs fields.
    # Convert quaternion to RPY so we can fill x, y, z, roll, pitch, yaw directly.
    rpy = Rotation.from_quat(quaternion_xyzw).as_euler("xyz").tolist()
    return task_space_pose_from_translation_rpy(translation_xyz, rpy)