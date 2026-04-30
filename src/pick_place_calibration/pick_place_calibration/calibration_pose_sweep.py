import json
from pathlib import Path
import time

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from dataclasses import dataclass
from pick_place_interfaces.msg import TaskSpacePose as TaskSpacePoseMsg
from pick_place_interfaces.srv import ExecuteTaskPose, ExecuteHome

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

        self._execute_home_position_client = self._node.create_client(
            ExecuteHome,
            "execute_home_position",
        )

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

    def wait_for_execute_home_service(self) -> bool:
        """
        Wait for the coordinator execute-home service to become available.

        Inputs:
            None

        Returns:
            bool: True if the service became available, otherwise False.
        """
        self._node.get_logger().info("Waiting for execute_home_position service...")

        while rclpy.ok():
            if self._execute_home_position_client.wait_for_service(timeout_sec=1.0):
                self._node.get_logger().info(
                    "execute_home_position service is available."
                )
                return True

            self._node.get_logger().info(
                "execute_home_position service not available yet, waiting again..."
            )

        return False

    def get_pose_sequence_file_path(self) -> Path:
        """
        Resolve the calibration pose sequence JSON file path.

        Inputs:
            None

        Returns:
            Path: Absolute path to the calibration pose sequence JSON file.
        """
        package_share_directory = Path(
            get_package_share_directory("pick_place_calibration")
        )
        return package_share_directory / "calibration_pose_sequence.json"

    def create_sequence_step_from_dict(
        self,
        step_data: dict,
    ) -> CalibrationSequenceStep:
        """
        Convert one JSON pose entry into a calibration sequence step.

        Inputs:
            step_data: One parsed JSON step dictionary.

        Returns:
            CalibrationSequenceStep: Converted calibration step.
        """
        translation = step_data["translation"]
        rpy_radians = step_data["rpy_radians"]

        pose = TaskSpacePoseMsg()
        pose.x = float(translation[0])
        pose.y = float(translation[1])
        pose.z = float(translation[2])
        pose.roll = float(rpy_radians[0])
        pose.pitch = float(rpy_radians[1])
        pose.yaw = float(rpy_radians[2])

        return CalibrationSequenceStep(
            name=str(step_data["name"]),
            pose=pose,
            is_image_pose=step_data["type"] == "image",
        )

    def load_calibration_sequence_from_file(self) -> list[CalibrationSequenceStep]:
        """
        Load the ordered calibration route from the JSON pose sequence file.

        Inputs:
            None

        Returns:
            list[CalibrationSequenceStep]:
                Ordered calibration route steps for execution.
        """
        pose_sequence_path = self.get_pose_sequence_file_path()

        if not pose_sequence_path.exists():
            raise FileNotFoundError(
                f"Calibration pose sequence file was not found: {pose_sequence_path}"
            )

        with open(pose_sequence_path, "r", encoding="utf-8") as pose_sequence_file:
            pose_sequence_data = json.load(pose_sequence_file)

        steps = pose_sequence_data.get("steps", [])

        if not steps:
            raise ValueError(
                f"Calibration pose sequence file contains no steps: {pose_sequence_path}"
            )

        calibration_sequence = [
            self.create_sequence_step_from_dict(step_data)
            for step_data in steps
        ]

        self._node.get_logger().info(
            f"Loaded calibration pose sequence from: {pose_sequence_path}"
        )

        return calibration_sequence
    
    def create_calibration_sequence(self) -> list[CalibrationSequenceStep]:
        """
        Create the ordered calibration route from the saved JSON pose sequence.

        Inputs:
            None

        Returns:
            list[CalibrationSequenceStep]:
                Ordered calibration route steps for execution.
        """
        return self.load_calibration_sequence_from_file()
    
    def get_calibration_sequence(self) -> list[CalibrationSequenceStep]:
        """
        Return the ordered calibration sequence.

        Inputs:
            None

        Returns:
            list[CalibrationSequenceStep]: Loaded calibration sequence steps.
        """
        return self._calibration_sequence

    def call_execute_home_service(self) -> ExecuteHome.Response | None:
        """
        Request a planned move to home through the coordinator.

        Inputs:
            None

        Returns:
            ExecuteHome.Response | None: Service response if successful, otherwise None.
        """
        if not self._execute_home_position_client.wait_for_service(timeout_sec=1.0):
            self._node.get_logger().error(
                "execute_home_position service is no longer available."
            )
            return None

        future = self._execute_home_position_client.call_async(ExecuteHome.Request())

        self._node.get_logger().info(
            "Waiting for execute_home_position response."
        )

        rclpy.spin_until_future_complete(self._node, future)

        if future.exception() is not None:
            self._node.get_logger().error(
                f"Execute-home service call raised an exception: {future.exception()}"
            )
            return None

        if not future.done():
            self._node.get_logger().error(
                "Execute-home service call did not complete."
            )
            return None

        response = future.result()

        if response is None:
            self._node.get_logger().error(
                "Execute-home service returned no response."
            )
            return None

        return response

    def move_home(self) -> bool:
        response = self.call_execute_home_service()

        if response is None:
            self._node.get_logger().error(
                "Home move failed because no response was received."
            )
            return False

        if not response.success:
            self._node.get_logger().error(f"Home move failed: {response.message}")
            return False

        self._node.get_logger().info(f"Home move succeeded: {response.message}")
        return True

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