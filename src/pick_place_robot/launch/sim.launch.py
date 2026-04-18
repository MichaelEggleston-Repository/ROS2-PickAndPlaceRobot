from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
import xacro


def launch_setup(context, *args, **kwargs):
    """
    Build the robot-dependent launch actions after resolving launch arguments.

    Inputs:
        context: Launch runtime context used to resolve substitutions.
        args: Unused positional launch arguments.
        kwargs: Unused keyword launch arguments.

    Returns:
        list: Launch actions created from the resolved launch arguments.
    """
    pkg_share = get_package_share_directory("pick_place_robot")
    panda_urdf_xacro = os.path.join(pkg_share, "urdf", "panda.urdf.xacro")

    enable_calibration_value = LaunchConfiguration(
        "enable_calibration"
    ).perform(context)

    robot_description = xacro.process_file(
        panda_urdf_xacro,
        mappings={"enable_calibration": enable_calibration_value},
    ).toxml()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": robot_description,
                "use_sim_time": True,
            }
        ],
    )

    remove_existing_panda = ExecuteProcess(
        cmd=[
            "gz",
            "service",
            "-s",
            "/world/my_world/remove",
            "--reqtype",
            "gz.msgs.Entity",
            "--reptype",
            "gz.msgs.Boolean",
            "--timeout",
            "2000",
            "--req",
            'name: "panda", type: MODEL',
        ],
        output="screen",
        shell=False,
    )

    spawn_panda = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_sim",
            "create",
            "-world",
            "my_world",
            "-name",
            "panda",
            "-string",
            robot_description,
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.0",
            "-R",
            "0.0",
            "-P",
            "0.0",
            "-Y",
            "0.0",
        ],
        output="screen",
    )

    spawn_joint_state_broadcaster = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "controller_manager",
            "spawner",
            "joint_state_broadcaster",
            "-c",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    spawn_arm_controller = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "controller_manager",
            "spawner",
            "joint_trajectory_controller",
            "-c",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    spawn_gripper_controller = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "controller_manager",
            "spawner",
            "gripper_trajectory_controller",
            "-c",
            "/controller_manager",
            "--controller-manager-timeout",
            "60",
            "--switch-timeout",
            "60",
        ],
        output="screen",
    )

    spawn_panda_after_remove = RegisterEventHandler(
        OnProcessExit(
            target_action=remove_existing_panda,
            on_exit=[spawn_panda],
        )
    )

    spawn_joint_state_broadcaster_after_robot = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_panda,
            on_exit=[spawn_joint_state_broadcaster],
        )
    )

    spawn_arm_controller_after_jsb = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_joint_state_broadcaster,
            on_exit=[spawn_arm_controller],
        )
    )

    spawn_gripper_controller_after_arm = RegisterEventHandler(
        OnProcessExit(
            target_action=spawn_arm_controller,
            on_exit=[spawn_gripper_controller],
        )
    )

    return [
        robot_state_publisher,
        remove_existing_panda,
        spawn_panda_after_remove,
        spawn_joint_state_broadcaster_after_robot,
        spawn_arm_controller_after_jsb,
        spawn_gripper_controller_after_arm,
    ]


def generate_launch_description():
    """
    Generate the simulation launch description for the Panda robot and camera bridges.

    Inputs:
        None

    Returns:
        LaunchDescription: Full launch description for the simulated setup.
    """
    pkg_share = get_package_share_directory("pick_place_robot")
    world = os.path.join(pkg_share, "worlds", "my_world.sdf")

    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    clock_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
        ],
        output="screen",
    )

    camera_info_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_bridge",
            "parameter_bridge",
            "/conveyor_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    rgb_image_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_image",
            "image_bridge",
            "/conveyor_camera/image",
        ],
        output="screen",
    )

    depth_image_bridge = ExecuteProcess(
        cmd=[
            "ros2",
            "run",
            "ros_gz_image",
            "image_bridge",
            "/conveyor_camera/depth_image",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "enable_calibration",
                default_value="false",
                description="If true, include the calibration tag on panda_hand_tcp.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(gz_launch),
                launch_arguments={"gz_args": f"-r {world}"}.items(),
            ),
            clock_bridge,
            camera_info_bridge,
            rgb_image_bridge,
            depth_image_bridge,
            OpaqueFunction(function=launch_setup),
        ]
    )