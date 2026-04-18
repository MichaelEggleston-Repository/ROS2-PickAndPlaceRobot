#!/usr/bin/env -S ros2 launch
"""Configure and start move_group for the project-owned Panda MoveIt setup."""
from os import path
from typing import List

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from pick_place_robot.moveit_config_loader import build_moveit_config_dict

def launch_setup(context, *args, **kwargs):
    """
    Build the MoveIt launch actions after resolving launch arguments.

    Inputs:
        context: Launch runtime context used to resolve substitutions.
        args: Unused positional launch arguments.
        kwargs: Unused keyword launch arguments.

    Returns:
        list: Launch actions created from the resolved launch arguments.
    """
    enable_rviz = LaunchConfiguration("enable_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    log_level = LaunchConfiguration("log_level")

    enable_calibration_value = LaunchConfiguration(
        "enable_calibration"
    ).perform(context)

    moveit_config = build_moveit_config_dict(
        enable_calibration=(enable_calibration_value == "true"),
    )

    move_group_parameters = [
        moveit_config,
        {"use_sim_time": use_sim_time},
    ]

    rviz_parameters = [
        {
            "robot_description": moveit_config["robot_description"],
            "robot_description_semantic": moveit_config[
                "robot_description_semantic"
            ],
            "robot_description_kinematics": moveit_config[
                "robot_description_kinematics"
            ],
            "robot_description_planning": moveit_config[
                "robot_description_planning"
            ],
            "planning_pipelines": moveit_config["planning_pipelines"],
            "default_planning_pipeline": moveit_config[
                "default_planning_pipeline"
            ],
            "ompl": moveit_config["ompl"],
            "use_sim_time": use_sim_time,
        },
    ]

    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="log",
        arguments=["--ros-args", "--log-level", log_level],
        parameters=move_group_parameters,
    )

    rviz_node = Node(
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
        parameters=rviz_parameters,
        condition=IfCondition(enable_rviz),
    )

    return [
        move_group_node,
        rviz_node,
    ]

def generate_launch_description():
    """
    Generate the MoveIt launch description for the Panda robot.

    Inputs:
        None

    Returns:
        LaunchDescription: Full launch description for move_group and optional RViz.
    """
    declared_arguments = generate_declared_arguments()

    return LaunchDescription(
        declared_arguments
        + [
            OpaqueFunction(function=launch_setup),
        ]
    )

def generate_declared_arguments() -> List[DeclareLaunchArgument]:
    """
    Build the declared launch arguments for the MoveIt launch file.

    Inputs:
        None

    Returns:
        List[DeclareLaunchArgument]: Declared launch arguments.
    """
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
            default_value="true",
            description="If true, use simulated clock.",
        ),
        DeclareLaunchArgument(
            "log_level",
            default_value="warn",
            description="Log level applied to launched ROS 2 nodes.",
        ),
        DeclareLaunchArgument(
            "enable_calibration",
            default_value="false",
            description="If true, include the calibration tag on panda_hand_tcp.",
        ),
    ]