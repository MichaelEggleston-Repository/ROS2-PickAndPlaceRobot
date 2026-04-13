import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint

HOME_JOINT_POSITIONS = [0.0, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
HOME_MOVE_DURATION_SEC = 2

class PandaHomeNode(Node):
    def __init__(self):
        """
        Create the Panda home node and connect to the arm trajectory action server.

        Inputs:
            None

        Returns:
            None
        """
        # Give the node a stable, descriptive ROS name.
        super().__init__("panda_home_node")

        # Connect to the Panda arm trajectory action server.
        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

    def wait_for_arm_server(self) -> bool:
        """
        Wait for the Panda arm action server to become available.

        Inputs:
            None

        Returns:
            bool: True if the action server is available, otherwise False.
        """
        # Waiting first gives the simulator and controller time to finish starting.
        self.get_logger().info("Waiting for Panda arm action server...")

        available = self._arm_client.wait_for_server(timeout_sec=10.0)

        if not available:
            self.get_logger().error("Panda arm action server was not found.")
            return False

        self.get_logger().info("Panda arm action server is ready.")
        return True
    
    def create_home_goal(self) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for the Panda home position.

        Inputs:
            None

        Returns:
            FollowJointTrajectory.Goal: A goal containing the home joint target.
        """
        goal = FollowJointTrajectory.Goal()

        # These are the 7 arm joints controlled by the Panda trajectory controller.
        goal.trajectory.joint_names = [
            "panda_joint1",
            "panda_joint2",
            "panda_joint3",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
        ]

        point = JointTrajectoryPoint()

        # This is the home pose you already tested manually.
        point.positions = HOME_JOINT_POSITIONS

        # Give the controller time to move smoothly to the target.
        point.time_from_start.sec = HOME_MOVE_DURATION_SEC

        goal.trajectory.points.append(point)

        return goal
    
    def send_home_goal(self, goal: FollowJointTrajectory.Goal):
        """
        Send the home trajectory goal to the Panda arm controller.

        Inputs:
            goal: The prepared FollowJointTrajectory goal for the home pose.

        Returns:
            The accepted goal handle if successful, otherwise None.
        """
        # Sending asynchronously matches the ROS action pattern.
        self.get_logger().info("Sending home goal to Panda arm controller...")

        send_goal_future = self._arm_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self.get_logger().error("No goal handle was returned by the action server.")
            return None

        if not goal_handle.accepted:
            self.get_logger().error("Home goal was rejected by the action server.")
            return None

        self.get_logger().info("Home goal was accepted by the action server.")
        return goal_handle

    def wait_for_home_result(self, goal_handle) -> bool:
        """
        Wait for the Panda home goal to finish executing.

        Inputs:
            goal_handle: The accepted ROS action goal handle returned by the server.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        # Waiting for the result lets us confirm the motion actually finished.
        self.get_logger().info("Waiting for Panda to reach the home pose...")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()

        if result is None:
            self.get_logger().error("No result was returned for the home goal.")
            return False

        if result.status != 4:
            self.get_logger().error(
                f"Home goal did not finish successfully. Status code: {result.status}"
            )
            return False

        self.get_logger().info("Panda reached the home pose successfully.")
        return True


def main(args=None):
    """
    Start the node, check the Panda arm server, and prepare the home goal.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # Initialize ROS before creating any nodes.
    rclpy.init(args=args)

    node = PandaHomeNode()

    # Stop early if the controller action server is not available.
    server_ready = node.wait_for_arm_server()

    if server_ready:
        home_goal = node.create_home_goal()

        node.get_logger().info(
            f"Created home goal for joints: {home_goal.trajectory.joint_names}"
        )
        node.get_logger().info(
            f"Home joint positions: {home_goal.trajectory.points[0].positions}"
        )

        goal_handle = node.send_home_goal(home_goal)

        if goal_handle is not None:
            motion_succeeded = node.wait_for_home_result(goal_handle)

            if motion_succeeded:
                node.get_logger().info("Home motion step succeeded.")
            else:
                node.get_logger().error("Home motion step failed.")
        else:
            node.get_logger().error("Goal sending step failed.")
    else:
        node.get_logger().error("Node setup failed because the arm server is unavailable.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()