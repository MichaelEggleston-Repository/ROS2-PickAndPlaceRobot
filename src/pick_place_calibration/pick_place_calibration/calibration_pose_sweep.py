import rclpy
from rclpy.node import Node
from dataclasses import dataclass
from pick_place_interfaces.msg import TaskSpacePose as TaskSpacePoseMsg
from pick_place_interfaces.srv import ExecuteTaskPose

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
    pose: TaskSpacePoseMsg
    is_image_pose: bool

class CalibrationPoseSweep:
    def __init__(self, node: Node):
        """
        Create a reusable calibration pose sweep helper.

        Inputs:
            node: Parent ROS node that owns logging, subscriptions, and clients.

        Returns:
            None
        """
        self._node = node

        self._execute_task_pose_client = self._node.create_client(
            ExecuteTaskPose,
            "execute_task_pose",
        )

        self._calibration_sequence = self.create_calibration_sequence()

        self._pose_failure_counts: dict[str, int] = {}
        self._pose_success_counts: dict[str, int] = {}

        self._node.get_logger().info("Calibration pose sweep helper ready.")
        self._node.get_logger().info(
            f"Loaded {len(self._calibration_sequence)} sequenced calibration steps."
        )

    def wait_for_execute_task_pose_service(self) -> bool:
        """
        Wait for the coordinator execute-task-pose service to become available.

        Inputs:
            None

        Returns:
            bool: True if the service became available, otherwise False.
        """
        self._node.get_logger().info("Waiting for execute_task_pose service...")

        while rclpy.ok():
            if self._execute_task_pose_client.wait_for_service(timeout_sec=1.0):
                self._node.get_logger().info(
                    "execute_task_pose service is available."
                )
                return True

            self._node.get_logger().info(
                "execute_task_pose service not available yet, waiting again..."
            )

        return False
    
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
                pose=TaskSpacePoseMsg(
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
    
    def get_calibration_sequence(self) -> list[CalibrationSequenceStep]:
        """
        Return the ordered calibration sequence.

        Inputs:
            None

        Returns:
            list[CalibrationSequenceStep]: Loaded calibration sequence steps.
        """
        return self._calibration_sequence

    def move_to_sequence_step(
        self,
        step: CalibrationSequenceStep,
        speed_scale: float = 0.25,
    ) -> bool:
        """
        Request motion to one calibration sequence step through the coordinator.

        Inputs:
            step: The calibration sequence step to execute.
            speed_scale: Motion speed scale factor for the requested move.

        Returns:
            bool: True if the move request succeeded, otherwise False.
        """
        self._node.get_logger().info(
            f"Calibration pose sweep requested move to step '{step.name}'."
        )

        if not self._execute_task_pose_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().error(
                "execute_task_pose service is no longer available."
            )
            return False

        request = ExecuteTaskPose.Request()
        request.pose = step.pose
        request.speed_scale = speed_scale

        self._node.get_logger().info(
            f"Sending execute_task_pose request for step '{step.name}'."
        )

        future = self._execute_task_pose_client.call_async(request)

        self._node.get_logger().info(
            f"Waiting for execute_task_pose response for step '{step.name}'."
        )

        rclpy.spin_until_future_complete(self._node, future)

        if future.exception() is not None:
            self._node.get_logger().error(
                f"execute_task_pose future for step '{step.name}' raised: {future.exception()}"
            )

        self._node.get_logger().info(
            f"Finished waiting for execute_task_pose response for step '{step.name}'. "
            f"done={future.done()}"
        )

        if not future.done():
            self._node.get_logger().warn(
                f"Move request for step '{step.name}' did not complete."
            )
            self._pose_failure_counts[step.name] = (
                self._pose_failure_counts.get(step.name, 0) + 1
            )
            return False

        response = future.result()

        if response is None:
            self._node.get_logger().warn(
                f"Move request for step '{step.name}' returned no response."
            )
            self._pose_failure_counts[step.name] = (
                self._pose_failure_counts.get(step.name, 0) + 1
            )
            return False

        self._node.get_logger().info(
            f"Move request result for '{step.name}': "
            f"success={response.success}, message='{response.message}'"
        )

        if not response.success:
            self._pose_failure_counts[step.name] = (
                self._pose_failure_counts.get(step.name, 0) + 1
            )
            return False

        self._pose_success_counts[step.name] = (
            self._pose_success_counts.get(step.name, 0) + 1
        )
        return True