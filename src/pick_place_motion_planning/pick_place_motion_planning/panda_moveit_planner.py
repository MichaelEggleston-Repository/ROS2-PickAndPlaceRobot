import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy
from moveit.core.robot_state import RobotState
from moveit_msgs.msg import Constraints, OrientationConstraint
from pick_place_interfaces.msg import TaskSpacePose as TaskSpacePoseMsg
from pick_place_interfaces.srv import PlanToTaskPose, PlanToJointPositions
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory

from pick_place_motion_planning.moveit_config_loader import build_moveit_config_dict
from pick_place_motion_planning.panda_scene_planning import PandaPlanningScene

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

        self._node.declare_parameter("enable_calibration", False)
        enable_calibration = bool(
            self._node.get_parameter("enable_calibration").value
        )

        self._node.get_logger().info(
            f"Loading MoveIt configuration with enable_calibration={enable_calibration}."
        )
        moveit_config = build_moveit_config_dict(
            enable_calibration=enable_calibration,
        )

        self._node.get_logger().info("About to initialize MoveItPy.")
        self._moveit = MoveItPy(
            node_name="moveit_py",
            config_dict=moveit_config,
        )
        self._node.get_logger().info("MoveItPy initialized successfully.")

        self._arm = self._moveit.get_planning_component(ARM_GROUP_NAME)
        self._node.get_logger().info("Planning component for Panda arm is ready.")
        self._scene = PandaPlanningScene(self._node)

        self._plan_to_task_pose_service = self._node.create_service(
            PlanToTaskPose,
            "plan_to_task_pose",
            self.plan_to_task_pose_callback,
        )

        self._plan_to_joint_positions_service = self._node.create_service(
            PlanToJointPositions,
            "plan_to_joint_positions",
            self.plan_to_joint_positions_callback,
        )

    def plan_to_task_pose_callback(
        self,
        request: PlanToTaskPose.Request,
        response: PlanToTaskPose.Response,
    ) -> PlanToTaskPose.Response:
        """
        Handle a planning request for a task-space pose.

        Inputs:
            request: Planning request containing the target task-space pose and
                optional orientation-constraint settings.
            response: Service response to populate with the planning result.

        Returns:
            PlanToTaskPose.Response: The populated planning response.
        """
        self._node.get_logger().info(
            "Received plan_to_task_pose request."
        )

        orientation_constraint_enabled = False
        orientation_tolerance_rad = 1.0

        if hasattr(request, "constrain_orientation"):
            orientation_constraint_enabled = request.constrain_orientation

        if hasattr(request, "orientation_tolerance_rad"):
            orientation_tolerance_rad = request.orientation_tolerance_rad

        if orientation_constraint_enabled:
            result = self.plan_to_task_pose_with_orientation_constraint(
                request.pose,
                orientation_tolerance_rad=orientation_tolerance_rad,
            )
        else:
            result = self.plan_to_task_pose(request.pose)

        response.success = result.success
        response.message = result.message

        if result.joint_trajectory is not None:
            response.joint_trajectory = result.joint_trajectory

        return response
    
    def plan_to_joint_positions_callback(
        self,
        request: PlanToJointPositions.Request,
        response: PlanToJointPositions.Response,
    ) -> PlanToJointPositions.Response:
        """
        Handle a planning request for a joint-space target.

        Inputs:
            request: Planning request containing joint names and target positions.
            response: Service response to populate with the planning result.

        Returns:
            PlanToJointPositions.Response: The populated planning response.
        """
        self._node.get_logger().info("Received plan_to_joint_positions request.")

        result = self.plan_to_joint_positions(
            joint_names=list(request.joint_names),
            joint_positions=list(request.joint_positions),
        )

        response.success = result.success
        response.message = result.message

        if result.joint_trajectory is not None:
            response.joint_trajectory = result.joint_trajectory

        return response

    def log_requested_pose(self, pose: TaskSpacePoseMsg) -> None:
        """
        Log the incoming task-space pose request.

        Inputs:
            pose: Requested task-space pose.

        Returns:
            None
        """
        self._node.get_logger().info(
            "Requested task pose: "
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
        Convert roll, pitch, yaw to quaternion.

        Inputs:
            roll: Rotation about X axis in radians.
            pitch: Rotation about Y axis in radians.
            yaw: Rotation about Z axis in radians.

        Returns:
            tuple[float, float, float, float]: Quaternion (x, y, z, w).
        """
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return qx, qy, qz, qw

    def create_pose_target(self, pose: TaskSpacePoseMsg) -> PoseStamped:
        """
        Build a PoseStamped target for MoveIt.

        Inputs:
            pose: Requested task-space pose.

        Returns:
            PoseStamped: Target pose in panda_link0.
        """
        target_pose = PoseStamped()
        target_pose.header.frame_id = "panda_link0"

        target_pose.pose.position.x = pose.x
        target_pose.pose.position.y = pose.y
        target_pose.pose.position.z = pose.z

        qx, qy, qz, qw = self.rpy_to_quaternion(
            pose.roll,
            pose.pitch,
            pose.yaw,
        )
        target_pose.pose.orientation.x = qx
        target_pose.pose.orientation.y = qy
        target_pose.pose.orientation.z = qz
        target_pose.pose.orientation.w = qw

        return target_pose

    def create_orientation_path_constraint(
        self,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
        tolerance_rad: float,
    ) -> Constraints:
        """
        Create a MoveIt orientation path constraint for panda_hand_tcp.

        Inputs:
            qx: Quaternion x.
            qy: Quaternion y.
            qz: Quaternion z.
            qw: Quaternion w.
            tolerance_rad: Allowed absolute axis tolerance in radians.

        Returns:
            Constraints: Orientation path constraint container.
        """
        constraint = OrientationConstraint()
        constraint.header.frame_id = "panda_link0"
        constraint.link_name = "panda_hand_tcp"
        constraint.orientation.x = qx
        constraint.orientation.y = qy
        constraint.orientation.z = qz
        constraint.orientation.w = qw
        constraint.absolute_x_axis_tolerance = tolerance_rad
        constraint.absolute_y_axis_tolerance = tolerance_rad
        constraint.absolute_z_axis_tolerance = tolerance_rad
        constraint.weight = 1.0

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
            margin_rad: Minimum distance each waypoint must keep from hard joint limits.

        Returns:
            tuple[bool, str]:
                True and a success message if all waypoints are safe, otherwise
                False and the reason for rejection.
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
                        f"Waypoint {point_index} rejected because {joint_name}={joint_position:.4f} "
                        f"is within {margin_rad:.3f} rad of lower limit {lower_limit:.4f}.",
                    )

                if joint_position >= upper_limit - margin_rad:
                    return (
                        False,
                        f"Waypoint {point_index} rejected because {joint_name}={joint_position:.4f} "
                        f"is within {margin_rad:.3f} rad of upper limit {upper_limit:.4f}.",
                    )

        return True, "All waypoints stayed within joint safety margins."

    def get_current_tcp_pose(self) -> Pose | None:
        """
        Read the current TCP pose from MoveIt.

        Inputs:
            None

        Returns:
            Pose | None: Current panda_hand_tcp pose if available.
        """
        tcp_link_name = "panda_hand_tcp"

        if hasattr(self._moveit, "get_planning_scene_monitor"):
            try:
                planning_scene_monitor = self._moveit.get_planning_scene_monitor()
                if planning_scene_monitor is not None:
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
    
    def plan_to_joint_positions(
        self,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> PlanningResult:
        """
        Plan an arm motion to a requested joint-space target.

        Inputs:
            joint_names: Ordered list of joint names for the target.
            joint_positions: Ordered list of target joint positions in radians.

        Returns:
            PlanningResult: The outcome of the planning request.
        """
        if not joint_names:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="Joint-space planning request contained no joint names.",
            )
        
        if joint_names != PANDA_ARM_JOINT_NAMES:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message=(
                    "Joint-space planning request used an unexpected joint order. "
                    f"Expected {PANDA_ARM_JOINT_NAMES}, got {joint_names}."
                ),
            )

        if len(joint_names) != len(joint_positions):
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message=(
                    "Joint-space planning request had mismatched joint_names and "
                    "joint_positions lengths."
                ),
            )

        self._node.get_logger().info(
            "Requested joint target: "
            + ", ".join(
                f"{name}={position:.4f}"
                for name, position in zip(joint_names, joint_positions)
            )
        )

        self._arm.set_start_state_to_current_state()

        robot_model = self._moveit.get_robot_model()
        goal_state = RobotState(robot_model)

        goal_state.set_joint_group_positions(
            ARM_GROUP_NAME,
            [float(position) for position in joint_positions],
        )
        goal_state.update()

        self._arm.set_goal_state(robot_state=goal_state)

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
        pose: TaskSpacePoseMsg,
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
        test_pose = TaskSpacePoseMsg(
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
    Start the Panda MoveIt planner server node and spin until shutdown.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = Node("panda_moveit_planner_node")
    planner = PandaMoveItPlanner(node)

    try:
        node.get_logger().info("Panda MoveIt planner server is ready.")
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()