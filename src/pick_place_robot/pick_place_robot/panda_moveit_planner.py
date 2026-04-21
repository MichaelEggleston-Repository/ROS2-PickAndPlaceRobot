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

PANDA_ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

PANDA_JOINT_LIMITS = {
    "panda_joint1": (-2.8973, 2.8973),
    "panda_joint2": (-1.7628, 1.7628),
    "panda_joint3": (-2.8973, 2.8973),
    "panda_joint4": (-3.0718, -0.0698),
    "panda_joint5": (-2.8973, 2.8973),
    "panda_joint6": (-0.0175, 3.7525),
    "panda_joint7": (-2.8973, 2.8973),
}

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
    
    def trajectory_respects_joint_margin(
        self,
        joint_trajectory: JointTrajectory,
        margin_rad: float = 0.10,
    ) -> tuple[bool, str]:
        """
        Check whether every waypoint in a planned trajectory stays a safe distance
        away from the Panda joint limits.

        Inputs:
            joint_trajectory: Planned arm trajectory returned by MoveIt.
            margin_rad: Minimum distance each waypoint must keep from the hard joint limits.

        Returns:
            tuple[bool, str]:
                True and a success message if the full trajectory is safe,
                otherwise False and the reason for rejection.
        """
        if not joint_trajectory.joint_names:
            return False, "Trajectory contains no joint names."

        if not joint_trajectory.points:
            return False, "Trajectory contains no points."

        for point_index, point in enumerate(joint_trajectory.points):
            for joint_name, joint_position in zip(
                joint_trajectory.joint_names,
                point.positions,
            ):
                limits = PANDA_JOINT_LIMITS.get(joint_name)

                if limits is None:
                    continue

                lower_limit, upper_limit = limits

                if joint_position <= lower_limit + margin_rad:
                    return (
                        False,
                        f"Waypoint {point_index} rejected because {joint_name}="
                        f"{joint_position:.4f} is within {margin_rad:.3f} rad "
                        f"of lower limit {lower_limit:.4f}.",
                    )

                if joint_position >= upper_limit - margin_rad:
                    return (
                        False,
                        f"Waypoint {point_index} rejected because {joint_name}="
                        f"{joint_position:.4f} is within {margin_rad:.3f} rad "
                        f"of upper limit {upper_limit:.4f}.",
                    )

        return True, "Trajectory stayed within joint safety margins."

    def final_waypoint_respects_branch_continuity(
        self,
        joint_trajectory: JointTrajectory,
        max_total_jump_rad: float = 4.0,
        max_single_joint_jump_rad: float = 1.8,
    ) -> tuple[bool, str]:
        """
        Check whether the final waypoint stays reasonably close to the current arm
        joint state, to reduce branch-flip solutions.

        Inputs:
            joint_trajectory: Planned arm trajectory returned by MoveIt.
            max_total_jump_rad: Maximum allowed sum of absolute joint deltas.
            max_single_joint_jump_rad: Maximum allowed absolute delta for any one joint.

        Returns:
            tuple[bool, str]:
                True and a success message if the final waypoint looks continuous,
                otherwise False and the reason for rejection.
        """
        current_joint_positions = self.get_current_arm_joint_positions()

        if current_joint_positions is None:
            return True, "Current joint state unavailable, skipping continuity check."

        if not joint_trajectory.points:
            return False, "Trajectory contains no points."

        final_point = joint_trajectory.points[-1]
        total_jump = 0.0

        for joint_name, final_position in zip(
            joint_trajectory.joint_names,
            final_point.positions,
        ):
            if joint_name not in current_joint_positions:
                continue

            delta = abs(float(final_position) - current_joint_positions[joint_name])
            total_jump += delta

            if delta > max_single_joint_jump_rad:
                return (
                    False,
                    f"Final waypoint rejected because {joint_name} jumps by "
                    f"{delta:.3f} rad, exceeding the limit of "
                    f"{max_single_joint_jump_rad:.3f} rad.",
                )

        if total_jump > max_total_jump_rad:
            return (
                False,
                f"Final waypoint rejected because total joint jump is "
                f"{total_jump:.3f} rad, exceeding the limit of "
                f"{max_total_jump_rad:.3f} rad.",
            )

        return True, "Final waypoint respected branch continuity limits."
    
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

    def get_current_arm_joint_positions(self) -> dict[str, float] | None:
        """
        Read the current Panda arm joint positions from the planning scene monitor.

        Inputs:
            None

        Returns:
            dict[str, float] | None:
                Mapping from Panda arm joint name to current joint position,
                or None if the state could not be read.
        """
        if not hasattr(self._moveit, "get_planning_scene_monitor"):
            self._node.get_logger().warn(
                "MoveItPy does not expose get_planning_scene_monitor()."
            )
            return None

        planning_scene_monitor = self._moveit.get_planning_scene_monitor()

        if planning_scene_monitor is None:
            self._node.get_logger().warn(
                "Planning scene monitor was unavailable while reading arm joints."
            )
            return None

        try:
            with planning_scene_monitor.read_only() as scene:
                current_state = scene.current_state
                group_positions = current_state.get_joint_group_positions(ARM_GROUP_NAME)

                if len(group_positions) != len(PANDA_ARM_JOINT_NAMES):
                    self._node.get_logger().warn(
                        "Unexpected Panda arm joint count returned from RobotState."
                    )
                    return None

                return {
                    joint_name: float(position)
                    for joint_name, position in zip(PANDA_ARM_JOINT_NAMES, group_positions)
                }
        except Exception as exc:
            self._node.get_logger().warn(
                f"Failed to read current Panda arm joint positions: {exc}"
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

        return self.plan_safe_trajectory_with_retries(
            max_attempts=30,
            margin_rad=0.10,
        )

    def final_goal_respects_joint_margin(
        self,
        joint_trajectory: JointTrajectory,
        margin_rad: float = 0.10,
    ) -> tuple[bool, str]:
        """
        Check whether the final waypoint in a planned trajectory stays a safe
        distance away from the Panda joint limits.

        Inputs:
            joint_trajectory: Planned arm trajectory returned by MoveIt.
            margin_rad: Minimum distance the final waypoint must keep from
                the hard joint limits.

        Returns:
            tuple[bool, str]:
                True and a success message if the final waypoint is safe,
                otherwise False and the reason for rejection.
        """
        if not joint_trajectory.joint_names:
            return False, "Trajectory contains no joint names."

        if not joint_trajectory.points:
            return False, "Trajectory contains no points."

        final_point = joint_trajectory.points[-1]

        for joint_name, joint_position in zip(
            joint_trajectory.joint_names,
            final_point.positions,
        ):
            limits = PANDA_JOINT_LIMITS.get(joint_name)

            if limits is None:
                continue

            lower_limit, upper_limit = limits

            if joint_position <= lower_limit + margin_rad:
                return (
                    False,
                    f"Final waypoint rejected because {joint_name}={joint_position:.4f} "
                    f"is within {margin_rad:.3f} rad of lower limit {lower_limit:.4f}.",
                )

            if joint_position >= upper_limit - margin_rad:
                return (
                    False,
                    f"Final waypoint rejected because {joint_name}={joint_position:.4f} "
                    f"is within {margin_rad:.3f} rad of upper limit {upper_limit:.4f}.",
                )

        return True, "Final waypoint stayed within joint safety margins."

    def plan_safe_trajectory_with_retries(
        self,
        max_attempts: int = 15,
        margin_rad: float = 0.10,
    ) -> PlanningResult:
        """
        Ask MoveIt for multiple plans and return the first trajectory that stays
        within the configured joint safety margin.

        Inputs:
            max_attempts: Maximum number of planning attempts to try.
            margin_rad: Minimum distance each waypoint must keep from hard joint limits.

        Returns:
            PlanningResult:
                Successful result containing the first safe joint trajectory found,
                otherwise a failure result describing the last rejection reason.
        """
        last_message = "MoveIt planning did not produce a safe trajectory."

        for attempt_index in range(1, max_attempts + 1):
            self._node.get_logger().info(
                f"Planning attempt {attempt_index}/{max_attempts}..."
            )

            plan_result = self._arm.plan()

            if not plan_result:
                last_message = (
                    f"Planning attempt {attempt_index} failed to produce a plan."
                )
                continue

            if plan_result.trajectory is None:
                last_message = (
                    f"Planning attempt {attempt_index} returned no trajectory."
                )
                continue

            trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
            joint_trajectory = trajectory_msg.joint_trajectory

            if not joint_trajectory.points:
                last_message = (
                    f"Planning attempt {attempt_index} returned an empty trajectory."
                )
                continue

            trajectory_is_safe, safety_message = self.final_goal_respects_joint_margin(
                joint_trajectory,
                margin_rad=margin_rad,
            )

            if not trajectory_is_safe:
                last_message = (
                    f"Planning attempt {attempt_index} produced an unsafe trajectory. "
                    f"{safety_message}"
                )
                self._node.get_logger().warn(last_message)
                continue

            return PlanningResult(
                success=True,
                joint_trajectory=joint_trajectory,
                message=(
                    f"MoveIt planning succeeded on attempt {attempt_index} "
                    "and the trajectory passed safety screening."
                ),
            )

        return PlanningResult(
            success=False,
            joint_trajectory=None,
            message=last_message,
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

        trajectory_is_safe, safety_message = self.trajectory_respects_joint_margin(
            joint_trajectory,
            margin_rad=0.10,
        )

        if not trajectory_is_safe:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message=(
                    "MoveIt constrained planning produced an unsafe joint trajectory. "
                    f"{safety_message}"
                ),
            )

        return PlanningResult(
            success=True,
            joint_trajectory=joint_trajectory,
            message=(
                "MoveIt constrained planning succeeded and the joint trajectory "
                "passed safety screening."
            ),
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