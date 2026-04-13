import rclpy
from rclpy.node import Node
from dataclasses import dataclass
import math

from moveit.planning import MoveItPy

from pick_place_robot.moveit_config_loader import build_moveit_config_dict

ARM_GROUP_NAME = "arm"

@dataclass
class PlanningResult:
    """
    Describe the result of asking the planner to create an arm motion.

    Inputs:
        success: True if planning succeeded, otherwise False.
        joint_positions: Final joint target chosen by the planner if available.
        message: Human-readable status message for logging and debugging.

    Returns:
        None
    """
    success: bool
    joint_positions: list[float] | None
    message: str

class PandaMoveItPlanner(Node):
    def __init__(self):
        """
        Create a Panda MoveIt planner node and initialize MoveItPy.

        Inputs:
            None

        Returns:
            None
        """
        super().__init__("panda_moveit_planner")

        self.get_logger().info("Loading MoveIt configuration.")
        moveit_config = build_moveit_config_dict()

        self.get_logger().info("About to initialize MoveItPy.")
        self._moveit = MoveItPy(
            node_name="moveit_py",
            config_dict=moveit_config,
        )
        self.get_logger().info("MoveItPy initialized successfully.")

        # The planner owns planning components, not config assembly details.
        self._arm = self._moveit.get_planning_component(ARM_GROUP_NAME)
        self.get_logger().info("Planning component for Panda arm is ready.")

    def log_requested_pose(self, pose) -> None:
        """
        Log the requested task-space pose before planning.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            None
        """
        self.get_logger().info(
            "Planning request received for task pose: "
            f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
            f"roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}"
        )

    def rpy_to_quaternion(
        self,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> tuple[float, float, float, float]:
        """
        Convert roll, pitch, yaw Euler angles into a quaternion.

        Inputs:
            roll: Rotation about the x-axis in radians.
            pitch: Rotation about the y-axis in radians.
            yaw: Rotation about the z-axis in radians.

        Returns:
            tuple[float, float, float, float]:
                Quaternion as (x, y, z, w).
        """
        half_roll = roll * 0.5
        half_pitch = pitch * 0.5
        half_yaw = yaw * 0.5

        cr = math.cos(half_roll)
        sr = math.sin(half_roll)
        cp = math.cos(half_pitch)
        sp = math.sin(half_pitch)
        cy = math.cos(half_yaw)
        sy = math.sin(half_yaw)

        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        qw = cr * cp * cy + sr * sp * sy

        return (qx, qy, qz, qw)

    def plan_to_task_pose(self, pose) -> PlanningResult:
        """
        Plan an arm motion to a task-space pose.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            PlanningResult: The outcome of the planning request.
        """
        self.log_requested_pose(pose)
        raise NotImplementedError("Task-space planning is not implemented yet.")
        
def main(args=None):
    """
    Start the Panda MoveIt planner node and keep it alive until shutdown.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = PandaMoveItPlanner()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()