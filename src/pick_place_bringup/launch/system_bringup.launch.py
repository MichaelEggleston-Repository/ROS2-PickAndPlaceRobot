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

    debug_visualization_arg = DeclareLaunchArgument(
        "debug_visualization_enabled",
        default_value="false",
        description="If true, show OpenCV debug overlay windows in the perception node.",
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

    pick_place_manager_node = Node(
        package="pick_place_manager",
        executable="pick_place_manager",
        name="pick_place_manager",
        output="screen",
        parameters=[
            {
                "calibration_file": "/home/michael/Projects/ROS2-PickAndPlaceRobot/calibration_data/session_20260501_132539/eye_to_hand_calibration.yaml",
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

    object_detection_node = Node(
        package="pick_place_vision",
        executable="object_perception",
        name="object_perception",
        output="screen",
        parameters=[
            {
                "debug_visualization_enabled": ParameterValue(
                    LaunchConfiguration("debug_visualization_enabled"),
                    value_type=bool,
                )
            }
        ],
    )

    calibration_data_collection_node = Node(
        package="pick_place_calibration",
        executable="calibration_data_collection",
        name="calibration_data_collection_node",
        output="screen",
        condition=IfCondition(LaunchConfiguration("run_calibration_collection")),
    )

    # ---- PDDL task planner -----------------------------------------------
    # goal_type:     "stack" | "arrange"
    # goal_sequence: comma-separated cube IDs in goal order
    #                stack   → bottom-to-top, e.g. "red_cube,green_cube,blue_cube"
    #                arrange → slot_1..slot_3, e.g. "blue_cube,green_cube,red_cube"
    # slot_N_x/y/z_surface: absolute robot-base-frame position of each conveyor slot
    pddl_planner_node = Node(
        package="pick_place_pddl",
        executable="pddl_planner_node",
        name="pddl_planner_node",
        output="screen",
    )

    # Update manager node to include PDDL and slot position parameters.
    # Overrides the earlier pick_place_manager_node definition so we redefine it here.
    pick_place_manager_node = Node(
        package="pick_place_manager",
        executable="pick_place_manager",
        name="pick_place_manager",
        output="screen",
        parameters=[
            {
                "calibration_file": "/home/michael/Projects/ROS2-PickAndPlaceRobot/calibration_data/session_20260501_132539/eye_to_hand_calibration.yaml",
                # Execution mode: "pddl" runs the full task planner;
                # "test" runs the simple single-cube pick-and-place loop.
                "execution_mode": "pddl",
                # PDDL goal specification.
                "goal_type": "stack",
                "goal_sequence": "red_cube,green_cube,blue_cube",
                # Conveyor slot absolute positions in the robot base frame.
                # Adjust these to match your physical (or simulated) setup.
                "slot_1_x": 0.50,
                "slot_1_y": -0.20,
                "slot_2_x": 0.50,
                "slot_2_y":  0.00,
                "slot_3_x": 0.50,
                "slot_3_y":  0.20,
            }
        ],
    )

    return LaunchDescription([
        enable_calibration_arg,
        run_calibration_collection_arg,
        debug_visualization_arg,
        pddl_planner_node,
        pick_place_manager_node,
        simulation_launch,
        planner_node,
        coordinator_node,
        camera_acquisition_node,
        object_detection_node,
        calibration_data_collection_node,
    ])
