# src/pick_place_bringup/launch/system_bringup.launch.py
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """
    Generate the shared system bringup launch description.

    Inputs:
        None

    Returns:
        LaunchDescription: Launch description for the shared system bringup.
    """
    enable_calibration_arg = DeclareLaunchArgument(
        "enable_calibration",
        default_value="false",
        description="If true, load the calibration world and calibration tag.",
    )

    run_calibration_collection_arg = DeclareLaunchArgument(
        "run_calibration_collection",
        default_value="false",
        description="If true, run the calibration data collection node.",
    )

    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("pick_place_simulation"),
                "launch",
                "sim.launch.py",
            )
        ),
        launch_arguments={
            "enable_calibration": LaunchConfiguration("enable_calibration"),
        }.items(),
    )

    planner_node = Node(
        package="pick_place_motion_planning",
        executable="panda_moveit_planner",
        name="panda_moveit_planner_node",
        output="screen",
        parameters=[
            {
                "enable_calibration": ParameterValue(
                    LaunchConfiguration("enable_calibration"),
                    value_type=bool,
                )
            }
        ],
    )

    coordinator_node = Node(
        package="pick_place_robot",
        executable="panda_coordinator",
        name="panda_coordinator_node",
        output="screen",
    )

    camera_acquisition_node = Node(
        package="pick_place_vision",
        executable="camera_acquisition",
        name="camera_acquisition_node",
        output="screen",
    )

    calibration_data_collection_node = Node(
        package="pick_place_calibration",
        executable="calibration_data_collection",
        name="calibration_data_collection_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_calibration_collection")),
    )

    return LaunchDescription([
        enable_calibration_arg,
        run_calibration_collection_arg,
        simulation_launch,
        planner_node,
        coordinator_node,
        camera_acquisition_node,
        calibration_data_collection_node,
    ])
