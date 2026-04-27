import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from pick_place_calibration.calibration_pose_sweep import CalibrationPoseSweep
from pick_place_calibration.calibration_data_collection import CalibrationDataCollection


class CalibrationHandToEyeNode(Node):
    """
    Orchestrate the hand-to-eye calibration sequence by coordinating robot
    motion requests and camera snapshot acquisition.
    """

    def __init__(self) -> None:
        super().__init__("calibration_hand_to_eye_node")

        self._coordinator_status = "unknown"

        self._coordinator_status_subscription = self.create_subscription(
            String,
            "/panda_coordinator/status",
            self.coordinator_status_callback,
            10,
        )

        self._pose_sweep = CalibrationPoseSweep(self)
        self._data_collection = CalibrationDataCollection(self)

        self.get_logger().info("Calibration hand-to-eye orchestrator started.")

    def coordinator_status_callback(self, msg: String) -> None:
        """
        Cache the latest coordinator status message.

        Inputs:
            msg: Coordinator status message.

        Returns:
            None
        """
        self._coordinator_status = msg.data

    def wait_for_coordinator_ready(self) -> bool:
        """
        Wait until the coordinator reports that startup is complete.

        Inputs:
            None

        Returns:
            bool: True if the coordinator became ready, otherwise False.
        """
        self.get_logger().info("Waiting for coordinator to report ready status...")

        while rclpy.ok():
            if self._coordinator_status == "ready":
                self.get_logger().info("Coordinator is ready.")
                return True

            self.get_logger().info(
                f"Coordinator not ready yet. Current status: {self._coordinator_status}"
            )
            rclpy.spin_once(self, timeout_sec=1.0)

        return False

    def wait_for_system_ready(self) -> bool:
        """
        Wait for all calibration orchestration prerequisites.

        Inputs:
            None

        Returns:
            bool: True if all prerequisites became available, otherwise False.
        """
        if not self.wait_for_coordinator_ready():
            self.get_logger().error(
                "Coordinator did not become ready. Exiting."
            )
            return False

        if not self._pose_sweep.wait_for_execute_task_pose_service():
            self.get_logger().error(
                "execute_task_pose service was not available. Exiting."
            )
            return False

        if not self._data_collection.wait_for_capture_service():
            self.get_logger().error(
                "capture_snapshot service was not available. Exiting."
            )
            return False

        self.get_logger().info(
            "All hand-to-eye calibration prerequisites are satisfied."
        )
        return True

    def run_calibration_sequence(self) -> bool:
        """
        Run the ordered hand-to-eye calibration movement and capture sequence.

        Inputs:
            None

        Returns:
            bool: True if the sequence completed, otherwise False.
        """
        sequence = self._pose_sweep.get_calibration_sequence()

        if not sequence:
            self.get_logger().error(
                "Calibration sequence is empty. Nothing to execute."
            )
            return False

        self.get_logger().info(
            f"Starting hand-to-eye calibration sequence with {len(sequence)} steps."
        )

        successful_captures = 0
        attempted_image_poses = 0

        for index, step in enumerate(sequence, start=1):
            step_type = "image" if step.is_image_pose else "intermediate"

            self.get_logger().info(
                f"Executing calibration step {index}/{len(sequence)}: "
                f"{step.name} ({step_type})"
            )

            motion_succeeded = self._pose_sweep.move_to_sequence_step(
                step,
                speed_scale=0.25,
            )

            if not motion_succeeded:
                self.get_logger().error(
                    f"Calibration motion failed at step '{step.name}'."
                )
                return False

            if not step.is_image_pose:
                continue

            attempted_image_poses += 1

            result = self._data_collection.capture_at_pose(
                step.name,
                step.pose,
            )

            if result.image_capture_success:
                self._data_collection.save_capture_result(result)
                successful_captures += 1
            else:
                self.get_logger().warn(
                    f"Image capture failed at calibration image pose '{step.name}'."
                )

        self.get_logger().info(
            "Hand-to-eye calibration sequence complete: "
            f"{successful_captures}/{attempted_image_poses} image captures succeeded."
        )
        return True


def main(args=None) -> None:
    """
    Start the hand-to-eye calibration orchestrator, wait for all required
    services and readiness conditions, run the calibration sequence once,
    and shut down cleanly.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = CalibrationHandToEyeNode()

    try:
        if not node.wait_for_system_ready():
            return

        sequence_succeeded = node.run_calibration_sequence()

        if not sequence_succeeded:
            node.get_logger().error(
                "Hand-to-eye calibration sequence failed."
            )
            return

        node.get_logger().info(
            "Hand-to-eye calibration sequence completed successfully."
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()