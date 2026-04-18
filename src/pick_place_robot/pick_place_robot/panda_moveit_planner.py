import rclpy
from rclpy.node import Node
import math
from dataclasses import dataclass

from moveit.planning import MoveItPy
from moveit_msgs.msg import Constraints, OrientationConstraint
from geometry_msgs.msg import Pose, PoseStamped
from trajectory_msgs.msg import JointTrajectory

from pick_place_robot.moveit_config_loader import build_moveit_config_dict
from pick_place_robot.task_space_pose import TaskSpacePose

ARM_GROUP_NAME = "arm"

@dataclass
class PlanningResult:
    """
    Describe the result of asking the planner to create an arm motion.

    Inputs:
        success: True if planning succeeded, otherwise False.
        joint_trajectory: Planned joint trajectory if available.
        message: Human-readable status message for logging and debugging.

    Returns:
        None
    """
    success: bool
    joint_trajectory: JointTrajectory | None
    message: str

class PandaMoveItPlanner:
    def __init__(self, node: Node):
        """
        Create a Panda MoveIt planner helper and initialize MoveItPy.

        Inputs:
            node: ROS 2 node used for logging and lifecycle ownership.

        Returns:
            None
        """

        self._node = node

        self._node.get_logger().info("Loading MoveIt configuration.")
        moveit_config = build_moveit_config_dict()

        self._node.get_logger().info("About to initialize MoveItPy.")
        self._moveit = MoveItPy(
            node_name="moveit_py",
            config_dict=moveit_config,
        )
        self._node.get_logger().info("MoveItPy initialized successfully.")

        # The planner owns planning components, not config assembly details.
        self._arm = self._moveit.get_planning_component(ARM_GROUP_NAME)
        self._node.get_logger().info("Planning component for Panda arm is ready.")

    def log_requested_pose(self, pose) -> None:
        """
        Log the requested task-space pose before planning.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            None
        """
        self._node.get_logger().info(
            "Planning request received for task pose: "
            f"x={pose.x:.3f}, y={pose.y:.3f}, z={pose.z:.3f}, "
            f"roll={pose.roll:.3f}, pitch={pose.pitch:.3f}, yaw={pose.yaw:.3f}"
        )

    def log_current_tcp_pose(self) -> None:
        """
        Log the current TCP pose for debugging.

        Inputs:
            None

        Returns:
            None
        """
        tcp_pose = self.get_current_tcp_pose()

        if tcp_pose is None:
            self._node.get_logger().error("Current TCP pose is unavailable.")
            return

        self._node.get_logger().info(
            "Current TCP pose: "
            f"x={float(tcp_pose.position.x):.6f}, "
            f"y={float(tcp_pose.position.y):.6f}, "
            f"z={float(tcp_pose.position.z):.6f}, "
            f"qx={float(tcp_pose.orientation.x):.6f}, "
            f"qy={float(tcp_pose.orientation.y):.6f}, "
            f"qz={float(tcp_pose.orientation.z):.6f}, "
            f"qw={float(tcp_pose.orientation.w):.6f}"
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
    
    def create_pose_target(self, pose) -> PoseStamped:
        """
        Create a PoseStamped target from a task-space pose.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            PoseStamped: The target pose formatted for MoveIt planning.
        """
        qx, qy, qz, qw = self.rpy_to_quaternion(
            pose.roll,
            pose.pitch,
            pose.yaw,
        )

        target = PoseStamped()
        target.header.frame_id = "panda_link0"

        target.pose.position.x = pose.x
        target.pose.position.y = pose.y
        target.pose.position.z = pose.z

        target.pose.orientation.x = qx
        target.pose.orientation.y = qy
        target.pose.orientation.z = qz
        target.pose.orientation.w = qw

        return target
    
    def create_orientation_path_constraint(
        self,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        tolerance_rad: float,
    ) -> Constraints:
        """
        Create a path constraint that keeps the Panda TCP orientation close to
        the requested quaternion throughout the motion.

        Inputs:
            qx: Quaternion x component.
            qy: Quaternion y component.
            qz: Quaternion z component.
            qw: Quaternion w component.
            tolerance_rad: Absolute orientation tolerance in radians for each axis.

        Returns:
            Constraints: A MoveIt path-constraint message for the Panda TCP.
        """
        constraint = OrientationConstraint()
        constraint.header.frame_id = "panda_link0"
        constraint.link_name = "panda_hand_tcp"

        constraint.orientation.x = float(qx)
        constraint.orientation.y = float(qy)
        constraint.orientation.z = float(qz)
        constraint.orientation.w = float(qw)

        constraint.absolute_x_axis_tolerance = float(tolerance_rad)
        constraint.absolute_y_axis_tolerance = float(tolerance_rad)
        constraint.absolute_z_axis_tolerance = float(tolerance_rad)
        constraint.weight = float(1.0)

        path_constraints = Constraints()
        path_constraints.orientation_constraints.append(constraint)

        return path_constraints
    
    def get_current_tcp_pose(self) -> Pose | None:
        """
        Get the current pose of the Panda TCP from the available MoveItPy state interfaces.

        Inputs:
            None

        Returns:
            Pose | None:
                The current TCP pose if available, otherwise None.
        """
        tcp_link_name = "panda_hand_tcp"

        if hasattr(self._moveit, "get_planning_scene_monitor"):
            planning_scene_monitor = self._moveit.get_planning_scene_monitor()

            if planning_scene_monitor is not None:
                try:
                    with planning_scene_monitor.read_only() as scene:
                        current_state = scene.current_state
                        tcp_pose = current_state.get_pose(tcp_link_name)

                        if tcp_pose is not None:
                            pose = Pose()
                            pose.position.x = tcp_pose.position.x
                            pose.position.y = tcp_pose.position.y
                            pose.position.z = tcp_pose.position.z
                            pose.orientation.x = tcp_pose.orientation.x
                            pose.orientation.y = tcp_pose.orientation.y
                            pose.orientation.z = tcp_pose.orientation.z
                            pose.orientation.w = tcp_pose.orientation.w
                            return pose
                except Exception as exc:
                    self._node.get_logger().warn(
                        f"Failed to read TCP pose from planning scene monitor: {exc}"
                    )

        if hasattr(self._moveit, "get_planning_scene"):
            try:
                planning_scene = self._moveit.get_planning_scene()
                current_state = planning_scene.current_state
                tcp_pose = current_state.get_pose(tcp_link_name)

                if tcp_pose is not None:
                    pose = Pose()
                    pose.position.x = tcp_pose.position.x
                    pose.position.y = tcp_pose.position.y
                    pose.position.z = tcp_pose.position.z
                    pose.orientation.x = tcp_pose.orientation.x
                    pose.orientation.y = tcp_pose.orientation.y
                    pose.orientation.z = tcp_pose.orientation.z
                    pose.orientation.w = tcp_pose.orientation.w
                    return pose
            except Exception as exc:
                self._node.get_logger().warn(
                    f"Failed to read TCP pose from planning scene: {exc}"
                )

        self._node.get_logger().error(
            "Could not read the current TCP pose from MoveItPy. "
            f"Available MoveItPy methods: {dir(self._moveit)}"
        )
        return None
    
    def plan_to_task_pose(self, pose) -> PlanningResult:
        """
        Plan an arm motion to a task-space pose.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            PlanningResult: The outcome of the planning request.
        """
        self.log_requested_pose(pose)

        target_pose = self.create_pose_target(pose)

        self._node.get_logger().info(
            "Created pose target in frame "
            f"'{target_pose.header.frame_id}': "
            f"x={target_pose.pose.position.x:.3f}, "
            f"y={target_pose.pose.position.y:.3f}, "
            f"z={target_pose.pose.position.z:.3f}"
        )

        self._arm.set_start_state_to_current_state()

        self._arm.set_goal_state(
            pose_stamped_msg=target_pose,
            pose_link="panda_hand_tcp",
        )

        plan_result = self._arm.plan()

        if not plan_result:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt planning failed to produce a plan.",
            )

        if plan_result.trajectory is None:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt planning failed because no trajectory was returned.",
            )

        trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
        joint_trajectory = trajectory_msg.joint_trajectory

        if not joint_trajectory.points:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt planning returned an empty joint trajectory.",
            )

        return PlanningResult(
            success=True,
            joint_trajectory=joint_trajectory,
            message="MoveIt planning succeeded and a joint trajectory was extracted.",
        )
        
    def plan_to_task_pose_with_orientation_constraint(
        self,
        pose: TaskSpacePose,
        orientation_tolerance_rad: float = 1.0,
    ) -> PlanningResult:
        """
        Plan an arm motion to a task-space pose while constraining the TCP
        orientation throughout the path.

        Inputs:
            pose: The target end-effector pose in task space.
            orientation_tolerance_rad: Maximum allowed orientation error in radians
                about each axis.

        Returns:
            PlanningResult: The outcome of the constrained planning request.
        """
        self.log_requested_pose(pose)

        target_pose = self.create_pose_target(pose)

        self._node.get_logger().info(
            "Created constrained pose target in frame "
            f"'{target_pose.header.frame_id}': "
            f"x={target_pose.pose.position.x:.3f}, "
            f"y={target_pose.pose.position.y:.3f}, "
            f"z={target_pose.pose.position.z:.3f}"
        )

        qx, qy, qz, qw = self.rpy_to_quaternion(
            pose.roll,
            pose.pitch,
            pose.yaw,
        )

        path_constraints = self.create_orientation_path_constraint(
            qx,
            qy,
            qz,
            qw,
            orientation_tolerance_rad,
        )

        self._arm.set_start_state_to_current_state()

        try:
            self._arm.set_path_constraints(path_constraints)
        except AttributeError:
            self._arm.setPathConstraints(path_constraints)

        self._arm.set_goal_state(
            pose_stamped_msg=target_pose,
            pose_link="panda_hand_tcp",
        )

        plan_result = self._arm.plan()

        try:
            self._arm.set_path_constraints(Constraints())
        except AttributeError:
            self._arm.setPathConstraints(Constraints())

        if not plan_result:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt constrained planning failed to produce a plan.",
            )

        if plan_result.trajectory is None:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt constrained planning failed because no trajectory was returned.",
            )

        trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
        joint_trajectory = trajectory_msg.joint_trajectory

        if not joint_trajectory.points:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="MoveIt constrained planning returned an empty joint trajectory.",
            )

        return PlanningResult(
            success=True,
            joint_trajectory=joint_trajectory,
            message="MoveIt constrained planning succeeded and a joint trajectory was extracted.",
        )
    
    def run_planning_smoke_test(self) -> None:
        """
        Run a simple manual planning smoke test using one task-space target.

        Inputs:
            None

        Returns:
            None
        """
        test_pose = TaskSpacePose(
            x=0.45,
            y=0.00,
            z=0.55,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        self._node.get_logger().info("Starting planner smoke test...")
        result = self.plan_to_task_pose(test_pose)

        self._node.get_logger().info(f"Planning success: {result.success}")
        self._node.get_logger().info(f"Planning message: {result.message}")
        self._node.get_logger().info(
            "Joint positions returned: "
            f"{result.joint_trajectory is not None}"
        )
        self._node.get_logger().info(f"Returned joint positions: {result.joint_trajectory}")
        
def main(args=None):
    """
    Start the Panda MoveIt planner node and run a manual planning smoke test.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = Node("planner_test_node")
    planner = PandaMoveItPlanner(node)

    try:
        planner.run_planning_smoke_test()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()