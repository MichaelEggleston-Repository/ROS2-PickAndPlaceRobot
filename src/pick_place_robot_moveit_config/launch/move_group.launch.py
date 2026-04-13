#!/usr/bin/env -S ros2 launch
"""Configure and start move_group for the project-owned Panda MoveIt setup."""
from os import path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    declared_arguments = generate_declared_arguments()

    moveit_config_package = "pick_place_robot_moveit_config"
    robot_description_package = "pick_place_robot"

    enable_rviz = LaunchConfiguration("enable_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    robot_description = {
        "robot_description": load_text(
            robot_description_package,
            path.join("urdf", "panda.urdf"),
        )
    }

    robot_description_semantic = {
        "robot_description_semantic": load_text(
            moveit_config_package,
            path.join("srdf", "panda.srdf"),
        )
    }

    robot_description_kinematics = {
        "robot_description_kinematics": {
            "arm": {
                "kinematics_solver": "kdl_kinematics_plugin/KDLKinematicsPlugin",
                "kinematics_solver_search_resolution": 0.0025,
                "kinematics_solver_timeout": 0.05,
                "kinematics_solver_attempts": 5,
            }
        }
    }

    joint_limits = {
        "robot_description_planning": load_yaml(
            moveit_config_package,
            path.join("config", "joint_limits.yaml"),
        )
    }

    planning_pipeline = {
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
            "planner_configs": {
                "RRTConnectkConfigDefault": {
                    "type": "geometric::RRTConnect",
                    "range": 0.0,
                },
            },
            "arm": {
                "projection_evaluator": "joints(panda_joint1,panda_joint2)",
                "planner_configs": ["RRTConnectkConfigDefault"],
                "longest_valid_segment_fraction": 0.005,
            },
        },
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    moveit_controller_manager_yaml = load_yaml(
        moveit_config_package,
        path.join("config", "moveit_controller_manager.yaml"),
    )
    moveit_controller_manager = {
        "moveit_controller_manager": (
            "moveit_simple_controller_manager/MoveItSimpleControllerManager"
        ),
        "moveit_simple_controller_manager": moveit_controller_manager_yaml,
    }

    trajectory_execution = {
        "allow_trajectory_execution": True,
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    nodes = [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="log",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                robot_description,
                {
                    "publish_frequency": 50.0,
                    "frame_prefix": "",
                    "use_sim_time": use_sim_time,
                },
            ],
        ),

        Node(
            package="moveit_ros_move_group",
            executable="move_group",
            output="log",
            arguments=["--ros-args", "--log-level", log_level],
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                joint_limits,
                planning_pipeline,
                trajectory_execution,
                planning_scene_monitor_parameters,
                moveit_controller_manager,
                {"use_sim_time": use_sim_time},
            ],
        ),

        Node(
            package="rviz2",
            executable="rviz2",
            output="log",
            arguments=[
                "--display-config",
                rviz_config,
                "--ros-args",
                "--log-level",
                log_level,
            ],
            parameters=[
                robot_description,
                robot_description_semantic,
                robot_description_kinematics,
                planning_pipeline,
                joint_limits,
                {"use_sim_time": use_sim_time},
            ],
            condition=IfCondition(enable_rviz),
        ),
    ]

    return LaunchDescription(declared_arguments + nodes)

def load_yaml(package_name: str, file_path: str):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)

    import yaml

    try:
        with open(absolute_file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def load_text(package_name: str, file_path: str) -> str:
    package_path = get_package_share_directory(package_name)
    absolute_file_path = path.join(package_path, file_path)

    with open(absolute_file_path, "r", encoding="utf-8") as file:
        return file.read()

def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    return [
        DeclareLaunchArgument(
            "enable_rviz",
            default_value="false",
            description="Flag to enable RViz2.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=path.join(
                get_package_share_directory("pick_place_robot_moveit_config"),
                "rviz",
                "moveit.rviz",
            ),
            description="Path to the RViz2 configuration file.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="If true, use simulated clock.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="warn",
            description="Log level applied to launched ROS 2 nodes.",
        ),
    ]