import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Home pose values are kept as constants so they are easy to tune later.
HOME_JOINT_POSITIONS = [-1.5708, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
HOME_MOVE_DURATION_SEC = 5

# Keep the joint ordering in one place so every goal uses the same mapping.
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

class PandaArmControl:
    def __init__(self, node: Node):
        """
        Create a reusable Panda arm control helper attached to an existing ROS node.

        Inputs:
            node: The ROS 2 node that owns logging and spinning context.

        Returns:
            None
        """
        # The control class uses an existing node instead of creating its own.
        # That keeps it reusable inside the coordinator.
        self._node = node

        self._arm_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

    def wait_for_server(self) -> bool:
        """
        Wait for the Panda arm action server to become available.

        Inputs:
            None

        Returns:
            bool: True if the action server is available, otherwise False.
        """
        self._node.get_logger().info("Waiting for Panda arm action server...")

        available = self._arm_client.wait_for_server(timeout_sec=30.0)

        if not available:
            self._node.get_logger().error("Panda arm action server was not found.")
            return False

        self._node.get_logger().info("Panda arm action server is ready.")
        return True

    def create_goal(
        self,
        joint_positions: list[float],
        duration_sec: int,
    ) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for a Panda arm joint target.

        Inputs:
            joint_positions: A list of 7 joint target values in Panda joint order.
            duration_sec: Target motion duration in seconds.

        Returns:
            FollowJointTrajectory.Goal: A goal containing the requested arm motion.
        """
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(ARM_JOINT_NAMES)} joint positions, got {len(joint_positions)}."
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = duration_sec

        goal.trajectory.points.append(point)
        return goal

    def send_goal(self, goal: FollowJointTrajectory.Goal):
        """
        Send an arm trajectory goal to the Panda arm controller.

        Inputs:
            goal: The prepared FollowJointTrajectory goal.

        Returns:
            The accepted goal handle if successful, otherwise None.
        """
        self._node.get_logger().info("Sending arm goal to Panda arm controller...")

        send_goal_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self._node, send_goal_future)

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "No goal handle was returned by the arm action server."
            )
            return None

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "Arm goal was rejected by the action server."
            )
            return None

        self._node.get_logger().info("Arm goal was accepted by the action server.")
        return goal_handle

    def wait_for_result(self, goal_handle) -> bool:
        """
        Wait for an accepted Panda arm goal to finish executing.

        Inputs:
            goal_handle: The accepted ROS action goal handle returned by the server.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        self._node.get_logger().info("Waiting for Panda arm motion to finish...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self._node, result_future)

        result = result_future.result()

        if result is None:
            self._node.get_logger().error("No result was returned for the arm goal.")
            return False

        if result.status != 4:
            self._node.get_logger().error(
                f"Arm goal did not finish successfully. Status code: {result.status}"
            )
            return False

        self._node.get_logger().info("Panda arm motion completed successfully.")
        return True

    def move_to_joint_positions(
        self,
        joint_positions: list[float],
        duration_sec: int,
    ) -> bool:
        """
        Move the Panda arm to a specific joint target and wait for completion.

        Inputs:
            joint_positions: A list of 7 joint target values in Panda joint order.
            duration_sec: Target motion duration in seconds.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        goal = self.create_goal(joint_positions, duration_sec)
        goal_handle = self.send_goal(goal)

        if goal_handle is None:
            return False

        return self.wait_for_result(goal_handle)

    def move_home(self) -> bool:
        """
        Move the Panda arm to the predefined home pose.

        Inputs:
            None

        Returns:
            bool: True if the home motion completed successfully, otherwise False.
        """
        return self.move_to_joint_positions(
            HOME_JOINT_POSITIONS,
            HOME_MOVE_DURATION_SEC,
        )
    
def main(args=None):
    """
    Run a simple manual test of PandaArmControl by commanding the home pose.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # This main is intended as a manual integration test
    rclpy.init(args=args)

    node = Node("panda_arm_control_test")
    arm = PandaArmControl(node)

    try:
        if not arm.wait_for_server():
            node.get_logger().error("Arm server was not available for the test.")
            raise SystemExit(1)

        node.get_logger().info("Starting PandaArmControl home-motion test...")

        if not arm.move_home():
            node.get_logger().error("PandaArmControl home-motion test failed.")
            raise SystemExit(1)

        node.get_logger().info("PandaArmControl home-motion test succeeded.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()