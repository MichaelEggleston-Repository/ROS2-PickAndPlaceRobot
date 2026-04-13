import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node

from control_msgs.action import FollowJointTrajectory


class PandaCoordinatorNode(Node):
    def __init__(self):
        """
        Create the Panda coordinator node and connect to the arm and gripper action servers.

        Inputs:
            None

        Returns:
            None
        """
        # Give the node a stable, descriptive ROS name.
        super().__init__("panda_coordinator_node")

        # Connect to the Panda arm trajectory action server.
        self._arm_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
        )

        # Connect to the Panda gripper trajectory action server.
        self._gripper_client = ActionClient(
            self,
            FollowJointTrajectory,
            "/gripper_trajectory_controller/follow_joint_trajectory",
        )

def main(args=None):
    """
    Start and shut down the Panda coordinator node cleanly.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # Initialize ROS before creating any nodes.
    rclpy.init(args=args)

    node = PandaCoordinatorNode()

    node.get_logger().info("Panda coordinator node started.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()