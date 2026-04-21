import rclpy
from rclpy.node import Node
import select
import sys
from dataclasses import dataclass

from pick_place_robot.panda_arm_control import PandaArmControl
from pick_place_robot.panda_moveit_planner import PandaMoveItPlanner
from pick_place_robot.panda_scene_planning import PandaPlanningScene
from pick_place_robot.task_space_pose import TaskSpacePose

@dataclass
class CalibrationSequenceStep:
    """
    Describe one ordered calibration route step.

    Inputs:
        name: Human-readable step name for logging.
        pose: Target TCP pose for this route step.
        is_image_pose: True if this step is intended for camera visibility or sample capture.

    Returns:
        None
    """
    name: str
    pose: TaskSpacePose
    is_image_pose: bool

class CalibrationPoseSweepNode(Node):
    def __init__(self):
        """
        Create a calibration pose sweep node for robot-guided camera visibility checks.

        Inputs:
            None

        Returns:
            None
        """
        super().__init__("calibration_pose_sweep_node")

        # Keep calibration motion logic separate from the main pick-and-place coordinator.
        self._arm = PandaArmControl(self)
        self._planner = PandaMoveItPlanner(self)
        self._scene = PandaPlanningScene(self)

        self.get_logger().info("Calibration pose sweep node started.")

        self._calibration_sequence = self.create_calibration_sequence()

        self.get_logger().info(
            f"Loaded {len(self._calibration_sequence)} sequenced calibration steps."
        )

        self._pose_failure_counts: dict[str, int] = {}
        self._pose_success_counts: dict[str, int] = {}

    def wait_for_control_interfaces(self) -> bool:
        """
        Wait for the required robot control interfaces to become available.

        Inputs:
            None

        Returns:
            bool: True if the required interfaces are available, otherwise False.
        """
        arm_ready = self._arm.wait_for_server()

        if not arm_ready:
            self.get_logger().error(
                "Calibration pose sweep setup failed because the arm server is unavailable."
            )
            return False

        self.get_logger().info(
            "Calibration pose sweep setup succeeded. Arm control is ready."
        )
        return True
    
    def create_calibration_sequence(self) -> list[CalibrationSequenceStep]:
        """
        Create the ordered calibration route validated manually in RViz.

        Inputs:
            None

        Returns:
            list[CalibrationSequenceStep]:
                Ordered calibration route steps for execution.
        """
        return [
            CalibrationSequenceStep(
                name="intermediate_1",
                pose=TaskSpacePose(
                    x=0.427,
                    y=0.003,
                    z=0.517,
                    roll=-3.142,
                    pitch=-0.000,
                    yaw=-0.007,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="image_1",
                pose=TaskSpacePose(
                    x=0.438,
                    y=0.107,
                    z=0.905,
                    roll=-0.520,
                    pitch=0.062,
                    yaw=0.006,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_2",
                pose=TaskSpacePose(
                    x=0.478,
                    y=0.073,
                    z=1.106,
                    roll=-0.452,
                    pitch=-0.163,
                    yaw=-0.202,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_3",
                pose=TaskSpacePose(
                    x=0.670,
                    y=0.105,
                    z=0.833,
                    roll=-0.539,
                    pitch=-0.198,
                    yaw=0.104,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="intermediate_2",
                pose=TaskSpacePose(
                    x=0.507,
                    y=0.368,
                    z=0.991,
                    roll=-0.475,
                    pitch=-0.181,
                    yaw=0.342,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="intermediate_3",
                pose=TaskSpacePose(
                    x=0.587,
                    y=0.217,
                    z=0.992,
                    roll=0.462,
                    pitch=-0.192,
                    yaw=0.661,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="image_4",
                pose=TaskSpacePose(
                    x=0.612,
                    y=0.131,
                    z=0.992,
                    roll=0.462,
                    pitch=-0.193,
                    yaw=0.517,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="intermediate_4",
                pose=TaskSpacePose(
                    x=0.437,
                    y=0.155,
                    z=1.154,
                    roll=0.328,
                    pitch=0.196,
                    yaw=0.840,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="image_5",
                pose=TaskSpacePose(
                    x=0.462,
                    y=0.049,
                    z=1.154,
                    roll=0.329,
                    pitch=0.197,
                    yaw=0.604,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_6",
                pose=TaskSpacePose(
                    x=0.461,
                    y=-0.064,
                    z=1.153,
                    roll=-0.326,
                    pitch=0.206,
                    yaw=-0.614,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_7",
                pose=TaskSpacePose(
                    x=0.546,
                    y=0.019,
                    z=1.039,
                    roll=-0.486,
                    pitch=-0.380,
                    yaw=-0.225,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_8",
                pose=TaskSpacePose(
                    x=0.630,
                    y=-0.028,
                    z=1.023,
                    roll=-0.442,
                    pitch=0.011,
                    yaw=-0.422,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_9",
                pose=TaskSpacePose(
                    x=0.607,
                    y=0.169,
                    z=1.023,
                    roll=-0.442,
                    pitch=0.011,
                    yaw=-0.105,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_10",
                pose=TaskSpacePose(
                    x=0.589,
                    y=-0.090,
                    z=0.975,
                    roll=-0.522,
                    pitch=-0.238,
                    yaw=-0.328,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_11",
                pose=TaskSpacePose(
                    x=0.433,
                    y=-0.085,
                    z=1.018,
                    roll=-0.497,
                    pitch=0.214,
                    yaw=-0.605,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_12",
                pose=TaskSpacePose(
                    x=0.537,
                    y=-0.069,
                    z=0.905,
                    roll=-0.521,
                    pitch=-0.037,
                    yaw=-0.266,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_13",
                pose=TaskSpacePose(
                    x=0.535,
                    y=0.080,
                    z=0.905,
                    roll=-0.521,
                    pitch=-0.037,
                    yaw=0.010,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="intermediate_before_image_14",
                pose=TaskSpacePose(
                    x=0.620,
                    y=0.074,
                    z=0.820,
                    roll=-0.505,
                    pitch=0.230,
                    yaw=0.012,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="image_14",
                pose=TaskSpacePose(
                    x=0.667,
                    y=0.074,
                    z=0.744,
                    roll=-0.505,
                    pitch=0.230,
                    yaw=0.012,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_15",
                pose=TaskSpacePose(
                    x=0.691,
                    y=0.221,
                    z=0.695,
                    roll=-0.572,
                    pitch=0.529,
                    yaw=0.056,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_16",
                pose=TaskSpacePose(
                    x=0.570,
                    y=0.223,
                    z=0.775,
                    roll=-0.496,
                    pitch=-0.030,
                    yaw=0.389,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_17",
                pose=TaskSpacePose(
                    x=0.592,
                    y=-0.156,
                    z=0.775,
                    roll=-0.496,
                    pitch=-0.030,
                    yaw=-0.241,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_18",
                pose=TaskSpacePose(
                    x=0.553,
                    y=-0.137,
                    z=0.910,
                    roll=-0.598,
                    pitch=-0.473,
                    yaw=-0.214,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_19",
                pose=TaskSpacePose(
                    x=0.712,
                    y=0.023,
                    z=0.886,
                    roll=-0.519,
                    pitch=0.114,
                    yaw=-0.224,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="image_20",
                pose=TaskSpacePose(
                    x=0.696,
                    y=-0.151,
                    z=0.887,
                    roll=0.466,
                    pitch=0.100,
                    yaw=0.066,
                ),
                is_image_pose=True,
            ),
            CalibrationSequenceStep(
                name="intermediate_5",
                pose=TaskSpacePose(
                    x=0.379,
                    y=-0.123,
                    z=1.142,
                    roll=0.444,
                    pitch=-0.353,
                    yaw=0.075,
                ),
                is_image_pose=False,
            ),
            CalibrationSequenceStep(
                name="image_21",
                pose=TaskSpacePose(
                    x=0.496,
                    y=-0.081,
                    z=1.107,
                    roll=0.430,
                    pitch=0.169,
                    yaw=0.315,
                ),
                is_image_pose=True,
            ),
        ]

    def create_candidate_calibration_poses(self) -> list[TaskSpacePose]:
        """
        Create an initial set of candidate TCP poses for calibration target visibility checks.

        Inputs:
            None

        Returns:
            list[TaskSpacePose]: Candidate task-space poses to test in the camera view.
        """
        # Start with a small manual sweep above the conveyor so pose tuning stays simple.
        return [
            TaskSpacePose(
                x=0.571,
                y=-0.229,
                z=0.862,
                roll=-0.483,
                pitch=0.003,
                yaw=-0.914,
            ),
            TaskSpacePose(
                x=0.40,
                y=-0.05,
                z=0.55,
                roll=3.14159,
                pitch=0.0,
                yaw=-0.2,
            ),
            TaskSpacePose(
                x=0.45,
                y=0.00,
                z=0.55,
                roll=3.14159,
                pitch=0.0,
                yaw=0.0,
            ),
            TaskSpacePose(
                x=0.40,
                y=0.05,
                z=0.55,
                roll=3.14159,
                pitch=0.0,
                yaw=0.2,
            ),
            TaskSpacePose(
                x=0.35,
                y=0.10,
                z=0.55,
                roll=3.14159,
                pitch=0.0,
                yaw=0.4,
            ),
        ]
    
    def execute_calibration_sequence(
        self,
        speed_scale: float = 0.25,
    ) -> bool:
        """
        Execute the ordered calibration route step by step.

        Inputs:
            speed_scale: Motion speed scale factor used during execution.

        Returns:
            bool: True if the full sequence completed successfully, otherwise False.
        """
        if not self._calibration_sequence:
            self.get_logger().error(
                "Cannot execute calibration sequence because no steps were loaded."
            )
            return False

        self.get_logger().info(
            f"Starting calibration sequence with "
            f"{len(self._calibration_sequence)} steps."
        )

        for index, step in enumerate(self._calibration_sequence, start=1):
            step_type = "image" if step.is_image_pose else "intermediate"

            self.get_logger().info(
                f"Executing calibration step {index}/{len(self._calibration_sequence)}: "
                f"{step.name} ({step_type})"
            )

            motion_succeeded = self.move_to_pose(
                step.pose,
                speed_scale=speed_scale,
                pose_name=step.name,
            )

            if not motion_succeeded:
                self.get_logger().error(
                    f"Calibration sequence failed at step '{step.name}'."
                )
                self._pose_failure_counts[step.name] = (
                    self._pose_failure_counts.get(step.name, 0) + 1
                )

                self.get_logger().warn(
                    f"Pose statistics so far for '{step.name}': "
                    f"successes={self._pose_success_counts.get(step.name, 0)}, "
                    f"failures={self._pose_failure_counts.get(step.name, 0)}"
                )
                return False

            if step.is_image_pose:
                self.get_logger().info(
                    f"Reached image pose '{step.name}'."
                )
            
            self._pose_success_counts[step.name] = (
                self._pose_success_counts.get(step.name, 0) + 1
            )

        self.get_logger().info(
            "Calibration sequence completed successfully."
        )
        return True

    def run_startup_sequence(self) -> bool:
        """
        Move the Panda arm to a known-safe home pose before calibration testing.

        Inputs:
            None

        Returns:
            bool: True if the startup motion succeeded, otherwise False.
        """
        self.get_logger().info(
            "Starting calibration pose sweep startup sequence..."
        )

        self._scene.republish_static_environment()
        rclpy.spin_once(self, timeout_sec=0.2)

        if not self._arm.move_home():
            self.get_logger().error(
                "Calibration pose sweep startup sequence failed during arm home motion."
            )
            return False

        self.get_logger().info(
            "Calibration pose sweep startup sequence completed successfully."
        )
        return True

    def move_to_pose(
        self,
        target_pose: TaskSpacePose,
        speed_scale: float = 0.25,
        pose_name: str = "unnamed_pose",
    ) -> bool:
        """
        Plan and execute motion to one calibration pose.

        Inputs:
            target_pose: The target TCP pose for calibration testing.
            speed_scale: Motion speed scale factor used during execution.

        Returns:
            bool: True if planning and execution both succeeded, otherwise False.
        """
        self.get_logger().info(
            "Planning calibration pose: "
            f"x={target_pose.x:.3f}, "
            f"y={target_pose.y:.3f}, "
            f"z={target_pose.z:.3f}, "
            f"roll={target_pose.roll:.3f}, "
            f"pitch={target_pose.pitch:.3f}, "
            f"yaw={target_pose.yaw:.3f}"
        )

        result = self._planner.plan_to_task_pose(target_pose)

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")

        if not result.success:
            self.get_logger().error(
            f"Calibration pose '{pose_name}' failed because planning did not succeed."
        )
            return False

        if result.joint_trajectory is None:
            self.get_logger().error(
                "Calibration pose motion failed because no joint trajectory was returned."
            )
            return False

        motion_succeeded = self._arm.move_to_joint_trajectory(
            result.joint_trajectory,
            speed_scale=speed_scale,
        )

        if not motion_succeeded:
            self.get_logger().error(
                f"Calibration pose '{pose_name}' failed during arm trajectory execution."
            )
            return False

        self.get_logger().info("Calibration pose motion completed successfully.")
        return True

    def move_to_first_candidate_pose(self) -> bool:
        """
        Plan and execute motion to the first candidate calibration pose.

        Inputs:
            None

        Returns:
            bool: True if planning and execution both succeeded, otherwise False.
        """
        if not self._candidate_poses:
            self.get_logger().error(
                "Cannot move to a calibration pose because no candidate poses were loaded."
            )
            return False

        target_pose = self._candidate_poses[0]

        self.get_logger().info(
            "Testing first candidate calibration pose: "
            f"x={target_pose.x:.3f}, "
            f"y={target_pose.y:.3f}, "
            f"z={target_pose.z:.3f}, "
            f"roll={target_pose.roll:.3f}, "
            f"pitch={target_pose.pitch:.3f}, "
            f"yaw={target_pose.yaw:.3f}"
        )

        result = self._planner.plan_to_task_pose(target_pose)

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")

        if not result.success:
            self.get_logger().error(
                "Calibration pose test failed because planning did not succeed."
            )
            return False

        if result.joint_trajectory is None:
            self.get_logger().error(
                "Calibration pose test failed because no joint trajectory was returned."
            )
            return False

        self.get_logger().info(
            f"Executing first calibration pose trajectory with "
            f"{len(result.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            result.joint_trajectory,
            speed_scale=0.25,
        )

        if not motion_succeeded:
            self.get_logger().error(
                "Calibration pose test failed during arm trajectory execution."
            )
            return False

        self.get_logger().info(
            "First candidate calibration pose completed successfully."
        )
        return True


def main(args=None):
    """
    Start the calibration pose sweep node and keep it available for manual homing
    and sequenced calibration execution.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)

    node = CalibrationPoseSweepNode()

    try:
        interfaces_ready = node.wait_for_control_interfaces()

        if not interfaces_ready:
            raise SystemExit(1)

        startup_succeeded = node.run_startup_sequence()

        if not startup_succeeded:
            raise SystemExit(1)

        node.get_logger().info(
            "Calibration pose sweep node is ready for manual RViz pose testing."
        )
        node.get_logger().info(
            "Press Enter to command home, or type 's' and press Enter to run the saved calibration sequence."
        )

        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            if not sys.stdin.isatty():
                continue

            ready_streams, _, _ = select.select([sys.stdin], [], [], 0.0)

            if not ready_streams:
                continue

            line = sys.stdin.readline().strip().lower()

            if line == "":
                node.get_logger().info(
                    "Manual home request received from terminal input."
                )

                if not node.run_startup_sequence():
                    node.get_logger().error(
                        "Manual home request failed during startup sequence."
                    )
                else:
                    node.get_logger().info(
                        "Manual home request completed successfully."
                    )

                continue

            if line == "s":
                node.get_logger().info(
                    "Manual calibration sequence request received from terminal input."
                )

                if not node.execute_calibration_sequence(speed_scale=1.0):
                    node.get_logger().error(
                        "Manual calibration sequence request failed."
                    )
                else:
                    node.get_logger().info(
                        "Manual calibration sequence request completed successfully."
                    )

                continue

            node.get_logger().info(
                f"Unknown command '{line}'. Press Enter for home or type 's' to run the sequence."
            )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()