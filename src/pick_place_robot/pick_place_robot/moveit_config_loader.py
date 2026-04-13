import os
from typing import Any

import yaml
from ament_index_python.packages import get_package_share_directory

MOVEIT_CONFIG_PACKAGE = "pick_place_robot_moveit_config"
ROBOT_DESCRIPTION_PACKAGE = "pick_place_robot"

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

def build_moveit_config_dict() -> dict[str, Any]:
    """
    Build the full MoveIt configuration dictionary for MoveItPy.

    Inputs:
        None

    Returns:
        dict[str, Any]: MoveIt parameters ready to pass into MoveItPy.
    """
    robot_description = load_text(
        ROBOT_DESCRIPTION_PACKAGE,
        os.path.join("urdf", "panda.urdf"),
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
        "robot_description_planning": joint_limits,
        "planning_pipelines": {
            "pipeline_names": ["ompl"],
        },
        "default_planning_pipeline": "ompl",
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
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
        "allow_trajectory_execution": True,
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": moveit_controller_manager,
    }