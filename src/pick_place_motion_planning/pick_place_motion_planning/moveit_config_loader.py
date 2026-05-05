# pick_place_robot/moveit_config_loader.py
import os
from typing import Any

import xacro
import yaml
from ament_index_python.packages import get_package_share_directory

MOVEIT_CONFIG_PACKAGE = "pick_place_moveit_config"
ROBOT_DESCRIPTION_PACKAGE = "pick_place_description"


def load_yaml(package_name: str, relative_path: str) -> dict[str, Any]:
    """
    Load a YAML file from a ROS 2 package share directory.

    Inputs:
        package_name: Name of the ROS 2 package that owns the file.
        relative_path: File path relative to the package share directory.

    Returns:
        dict[str, Any]: Parsed YAML content as a dictionary.
    """
    package_path = get_package_share_directory(package_name)
    absolute_path = os.path.join(package_path, relative_path)

    try:
        with open(absolute_path, "r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
    except OSError as exc:
        raise RuntimeError(f"Failed to read YAML file: {absolute_path}") from exc

    if data is None:
        raise RuntimeError(f"YAML file was empty: {absolute_path}")

    if not isinstance(data, dict):
        raise RuntimeError(
            f"Expected a dictionary at the top level of YAML file: {absolute_path}"
        )

    return data


def load_text(package_name: str, relative_path: str) -> str:
    """
    Load a text file from a ROS 2 package share directory.

    Inputs:
        package_name: Name of the ROS 2 package that owns the file.
        relative_path: File path relative to the package share directory.

    Returns:
        str: Full file contents as text.
    """
    package_path = get_package_share_directory(package_name)
    absolute_path = os.path.join(package_path, relative_path)

    try:
        with open(absolute_path, "r", encoding="utf-8") as file:
            return file.read()
    except OSError as exc:
        raise RuntimeError(f"Failed to read text file: {absolute_path}") from exc


def build_robot_description(
    enable_calibration: bool = False,
) -> str:
    """
    Build the Panda robot description from xacro.

    Inputs:
        enable_calibration: If True, include the calibration tag on panda_hand_tcp.

    Returns:
        str: Fully processed robot description XML.
    """
    package_path = get_package_share_directory(ROBOT_DESCRIPTION_PACKAGE)
    panda_urdf_xacro = os.path.join(package_path, "urdf", "panda.urdf.xacro")

    return xacro.process_file(
        panda_urdf_xacro,
        mappings={
            "enable_calibration": "true" if enable_calibration else "false",
        },
    ).toxml()


def build_moveit_config_dict(
    enable_calibration: bool = False,
) -> dict[str, Any]:
    """
    Build the full MoveIt configuration dictionary for MoveItPy.

    Inputs:
        enable_calibration: If True, include the calibration tag on panda_hand_tcp.

    Returns:
        dict[str, Any]: MoveIt parameters ready to pass into MoveItPy.
    """
    robot_description = build_robot_description(
        enable_calibration=enable_calibration,
    )

    robot_description_semantic = load_text(
        MOVEIT_CONFIG_PACKAGE,
        os.path.join("srdf", "panda.srdf"),
    )

    robot_description_kinematics = load_yaml(
        MOVEIT_CONFIG_PACKAGE,
        os.path.join("config", "kinematics.yaml"),
    )

    joint_limits = load_yaml(
        MOVEIT_CONFIG_PACKAGE,
        os.path.join("config", "joint_limits.yaml"),
    )

    ompl_planning = load_yaml(
        MOVEIT_CONFIG_PACKAGE,
        os.path.join("config", "ompl_planning.yaml"),
    )

    moveit_controller_manager = load_yaml(
        MOVEIT_CONFIG_PACKAGE,
        os.path.join("config", "moveit_controller_manager.yaml"),
    )

    return {
        "robot_description": robot_description,
        "robot_description_semantic": robot_description_semantic,
        "robot_description_kinematics": robot_description_kinematics,
        "robot_description_planning": {
            **joint_limits,
            "cartesian_limits": {
                "max_trans_vel": 1.0,
                "max_trans_acc": 2.25,
                "max_trans_dec": -5.0,
                "max_rot_vel": 1.57,
            },
        },
        "planning_pipelines": {
            "pipeline_names": ["ompl", "pilz_industrial_motion_planner"],
        },
        "default_planning_pipeline": "ompl",
        "plan_request_params": {
            "planning_pipeline": "ompl",
            "planner_id": "RRTConnectkConfigDefault",
            "planning_attempts": 10,
            "planning_time": 5.0,
            "max_velocity_scaling_factor": 1.0,
            "max_acceleration_scaling_factor": 1.0,
        },
        "ompl": {
            "planning_plugins": ["ompl_interface/OMPLPlanner"],
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
                "default_planning_request_adapters/CheckForStackedConstraints",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
            "start_state_max_bounds_error": 0.31416,
            **ompl_planning,
        },
        "pilz_industrial_motion_planner": {
            "planning_plugins": ["pilz_industrial_motion_planner/CommandPlanner"],
            "request_adapters": [
                "default_planning_request_adapters/ResolveConstraintFrames",
                "default_planning_request_adapters/ValidateWorkspaceBounds",
                "default_planning_request_adapters/CheckStartStateBounds",
                "default_planning_request_adapters/CheckStartStateCollision",
            ],
            "response_adapters": [
                "default_planning_response_adapters/AddTimeOptimalParameterization",
                "default_planning_response_adapters/ValidateSolution",
                "default_planning_response_adapters/DisplayMotionPath",
            ],
        },
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "monitor_dynamics": False,
        "joint_state_topic": "/joint_states",
        "allow_trajectory_execution": True,
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.6,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": moveit_controller_manager,
    }