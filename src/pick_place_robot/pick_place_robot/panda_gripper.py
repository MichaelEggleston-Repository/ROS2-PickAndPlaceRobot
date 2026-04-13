import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

# Open position for each Panda finger joint.
GRIPPER_OPEN_POSITION = 0.04
GRIPPER_CLOSED_POSITION = 0.0
GRIPPER_MOVE_DURATION_SEC = 2


class PandaGripperNode(Node):
    def __init__(self):
        """
        Create the Panda gripper node and connect to the gripper action server.

        Inputs:
            None

        Returns:
            None
        """
        # Give the node a stable, descriptive ROS name.
        super().__init__("panda_gripper_node")

        # Connect to the Panda gripper trajectory action server.
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_trajectory_controller/follow_joint_trajectory",
        )

    def wait_for_gripper_server(self) -> bool:
        """
        Wait for the Panda gripper action server to become available.

        Inputs:
            None

        Returns:
            bool: True if the action server is available, otherwise False.
        """
        # Waiting first avoids sending commands before the controller is ready.
        self.get_logger().info("Waiting for Panda gripper action server...")

        available = self._gripper_client.wait_for_server(timeout_sec=30.0)

        if not available:
            self.get_logger().error("Panda gripper action server was not found.")
            return False

        self.get_logger().info("Panda gripper action server is ready.")
        return True

    def create_open_goal(self) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for opening the Panda gripper.

        Inputs:
            None

        Returns:
            FollowJointTrajectory.Goal: A goal containing the open-gripper target.
        """
        goal = FollowJointTrajectory.Goal()

        # These are the two Panda finger joints controlled by the gripper controller.
        goal.trajectory.joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        point = JointTrajectoryPoint()

        # Open both fingers symmetrically.
        point.positions = [
            GRIPPER_OPEN_POSITION,
            GRIPPER_OPEN_POSITION,
        ]

        # Give the gripper time to move smoothly.
        point.time_from_start.sec = GRIPPER_MOVE_DURATION_SEC

        goal.trajectory.points.append(point)

        return goal

    def create_close_goal(self) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for closing the Panda gripper.

        Inputs:
            None

        Returns:
            FollowJointTrajectory.Goal: A goal containing the closed-gripper target.
        """
        goal = FollowJointTrajectory.Goal()

        # These are the two Panda finger joints controlled by the gripper controller.
        goal.trajectory.joint_names = [
            "panda_finger_joint1",
            "panda_finger_joint2",
        ]

        point = JointTrajectoryPoint()

        # Close both fingers symmetrically.
        point.positions = [
            GRIPPER_CLOSED_POSITION,
            GRIPPER_CLOSED_POSITION,
        ]

        # Give the gripper time to move smoothly.
        point.time_from_start.sec = GRIPPER_MOVE_DURATION_SEC

        goal.trajectory.points.append(point)

        return goal

    def send_gripper_goal(self, goal: FollowJointTrajectory.Goal):
        """
        Send the gripper goal to the Panda gripper controller.

        Inputs:
            goal: The prepared FollowJointTrajectory goal the gripper.

        Returns:
            The accepted goal handle if successful, otherwise None.
        """
        # Sending asynchronously matches the ROS action pattern.
        self.get_logger().info("Sending gripper goal to Panda gripper controller...")

        send_goal_future = self._gripper_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error("No goal handle was returned by the gripper action server.")
            return None

        if not goal_handle.accepted:
            self.get_logger().error("Gripper goal was rejected by the action server.")
            return None

        self.get_logger().info("Gripper goal was accepted by the action server.")
        return goal_handle

    def wait_for_gripper_result(self, goal_handle) -> bool:
        """
        Wait for the gripper goal to finish executing.

        Inputs:
            goal_handle: The accepted ROS action goal handle returned by the server.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        # Waiting for the result confirms the gripper really finished moving.
        self.get_logger().info("Waiting for Panda gripper goal...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error("No result was returned for the gripper goal.")
            return False

        if result.status != 4:
            self.get_logger().error(
                f"Gripper goal did not finish successfully. Status code: {result.status}"
            )
            return False

        self.get_logger().info("Panda gripper goal reached successfully.")
        return True

def main(args=None):
    """
    Start the Panda gripper node, check the server, and prepare the goal.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # Initialize ROS before creating any nodes.
    rclpy.init(args=args)

    node = PandaGripperNode()

    server_ready = node.wait_for_gripper_server()

    if server_ready:
        open_goal = node.create_open_goal()

        node.get_logger().info(
            f"Created open goal for joints: {open_goal.trajectory.joint_names}"
        )
        node.get_logger().info(
            f"Open positions: {open_goal.trajectory.points[0].positions}"
        )

        goal_handle = node.send_gripper_goal(open_goal)

        if goal_handle is not None:
            motion_succeeded = node.wait_for_gripper_result(goal_handle)

            if motion_succeeded:
                node.get_logger().info("Open-gripper motion step succeeded.")
            else:
                node.get_logger().error("Open-gripper motion step failed.")
        else:
            node.get_logger().error("Gripper goal sending step failed.")

        close_goal = node.create_close_goal()

        node.get_logger().info(
            f"Created close goal for joints: {close_goal.trajectory.joint_names}"
        )
        node.get_logger().info(
            f"Close positions: {close_goal.trajectory.points[0].positions}"
        )

        goal_handle = node.send_gripper_goal(close_goal)

        if goal_handle is not None:
            motion_succeeded = node.wait_for_gripper_result(goal_handle)

            if motion_succeeded:
                node.get_logger().info("Close-gripper motion step succeeded.")
            else:
                node.get_logger().error("Close-gripper motion step failed.")
        else:
            node.get_logger().error("Gripper goal sending step failed.")
    else:
        node.get_logger().error("Gripper node setup failed because the server is unavailable.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()