from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory("pick_place_robot")
    world = os.path.join(pkg_share, "worlds", "my_world.sdf")
    panda_urdf = os.path.join(pkg_share, "urdf", "panda.urdf")

    ros_gz_sim_share = get_package_share_directory("ros_gz_sim")
    gz_launch = os.path.join(ros_gz_sim_share, "launch", "gz_sim.launch.py")

    with open(panda_urdf, "r", encoding="utf-8") as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    camera_info_bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_bridge", "parameter_bridge",
            "/conveyor_camera/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo",
        ],
        output="screen",
    )

    rgb_image_bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_image", "image_bridge",
            "/conveyor_camera/image",
        ],
        output="screen",
    )

    depth_image_bridge = ExecuteProcess(
        cmd=[
            "ros2", "run", "ros_gz_image", "image_bridge",
            "/conveyor_camera/depth_image",
        ],
        output="screen",
    )

    spawn_panda = TimerAction(
        period=5.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "run", "ros_gz_sim", "create",
                    "-world", "my_world",
                    "-name", "panda",
                    "-file", panda_urdf,
                    "-x", "0.0",
                    "-y", "0.0",
                    "-z", "0.0",
                    "-R", "0.0",
                    "-P", "0.0",
                    "-Y", "0.0",
                ],
                output="screen",
            )
        ],
    )

    spawn_joint_state_broadcaster = TimerAction(
        period=12.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "run", "controller_manager", "spawner",
                    "joint_state_broadcaster",
                    "-c", "/controller_manager",
                    "--controller-manager-timeout", "60",
                    "--switch-timeout", "60",
                    "--activate-as-group",
                ],
                output="screen",
            )
        ],
    )

    spawn_arm_controller = TimerAction(
        period=20.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "run", "controller_manager", "spawner",
                    "joint_trajectory_controller",
                    "-c", "/controller_manager",
                    "--controller-manager-timeout", "60",
                    "--switch-timeout", "60",
                ],
                output="screen",
            )
        ],
    )

    spawn_gripper_controller = TimerAction(
        period=24.0,
        actions=[
            ExecuteProcess(
                cmd=[
                    "ros2", "run", "controller_manager", "spawner",
                    "gripper_trajectory_controller",
                    "-c", "/controller_manager",
                    "--controller-manager-timeout", "60",
                    "--switch-timeout", "60",
                ],
                output="screen",
            )
        ],
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_launch),
            launch_arguments={"gz_args": f"-r {world}"}.items(),
        ),
        robot_state_publisher,
        camera_info_bridge,
        rgb_image_bridge,
        depth_image_bridge,
        spawn_panda,
        spawn_joint_state_broadcaster,
        spawn_arm_controller,
        spawn_gripper_controller,
    ])