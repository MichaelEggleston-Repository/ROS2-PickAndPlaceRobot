import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState
from moveit_msgs.msg import Constraints, OrientationConstraint
from pick_place_interfaces.msg import TaskSpacePose as TaskSpacePoseMsg
from pick_place_interfaces.srv import PlanToTaskPose, PlanToJointPositions, ComputeApproachJoints
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

        # Publish static collision objects once the planning scene monitor has
        # had time to subscribe.  A one-shot 2-second timer avoids a race
        # between the publisher and the MoveIt planning scene monitor.
        self._scene_init_timer = self._node.create_timer(
            0.5, self._publish_static_scene_once
        )

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

        self._compute_approach_joints_service = self._node.create_service(
            ComputeApproachJoints,
            "compute_approach_joints",
            self.compute_approach_joints_callback,
        )

    def _publish_static_scene_once(self) -> None:
        """
        Publish the fixed environment collision objects to the MoveIt planning
        scene exactly once, then cancel the timer.

        Called by a one-shot timer 2 s after init so the planning scene monitor
        has had time to subscribe before the first publish.

        Inputs:
            None

        Returns:
            None
        """
        self._scene.add_static_environment()
        self._node.get_logger().info(
            "Static environment published to planning scene "
            "(conveyor_surface slab at z=0.40)."
        )
        self._scene_init_timer.cancel()

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
        speed_scale = 1.0

        if hasattr(request, "use_orientation_constraint"):
            orientation_constraint_enabled = request.use_orientation_constraint

        if hasattr(request, "orientation_tolerance_rad"):
            orientation_tolerance_rad = request.orientation_tolerance_rad

        if hasattr(request, "speed_scale") and request.speed_scale > 0.0:
            speed_scale = request.speed_scale

        if orientation_constraint_enabled:
            result = self.plan_to_task_pose_with_orientation_constraint(
                request.pose,
                orientation_tolerance_rad=orientation_tolerance_rad,
                speed_scale=speed_scale,
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

            ompl_params = PlanRequestParameters(self._moveit, "")
            ompl_params.planning_pipeline = "ompl"
            ompl_params.planner_id = "RRTConnectkConfigDefault"
            ompl_params.planning_time = 15.0
            plan_result = self._arm.plan(single_plan_parameters=ompl_params)

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
        speed_scale: float = 1.0,
        max_attempts: int = 50,
        margin_rad: float = 0.10,
    ) -> PlanningResult:
        """
        Plan a linear Cartesian motion to a task-space pose using PILZ LIN.

        PILZ LIN plans a guaranteed straight-line Cartesian path to the goal
        without using OMPL or orientation path constraints. This is the
        standard approach for grasp descent and lift motions in MoveIt2
        pick-and-place applications.

        If PILZ LIN fails (e.g. near a singularity), the method falls back to
        unconstrained OMPL planning so the coordinator retry loop can still
        recover.

        Inputs:
            pose: The target end-effector pose in task space.
            orientation_tolerance_rad: Unused — kept for API compatibility.
                PILZ enforces a straight Cartesian line by design.
            max_attempts: Unused for PILZ (it is deterministic). Retained for
                API compatibility and used only if the OMPL fallback is needed.
            margin_rad: Minimum distance the final waypoint must keep from hard
                joint limits.

        Returns:
            PlanningResult: The outcome of the planning request.
        """
        self.log_requested_pose(pose)

        target_pose = self.create_pose_target(pose)

        self._node.get_logger().info(
            "Created PILZ LIN target in frame "
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

        # PILZ validates joint velocity and acceleration limits during
        # trajectory generation and returns PLANNING_FAILED if any waypoint's
        # IK solution requires exceeding them.
        #
        # max_acceleration_scaling_factor controls the Cartesian TCP
        # acceleration in the trapezoidal profile.  PILZ errors report the raw
        # hardware joint limit (fixed, from URDF) and the acceleration the
        # generated trajectory requires.  Empirically, required joint accel
        # scales linearly with accel_scale, so the safe value is:
        #
        #   safe_accel = current_accel × (hw_limit / required) × safety_margin
        #
        # Worst observed cases:
        #   joint2 place descent: required=2.92 @ accel=0.25, hw_limit=1.875
        #       → safe_accel = 0.25 × (1.875/2.92) × 0.9 = 0.145  (binding)
        #   joint4 pick descent:  required=4.01 @ accel=0.50, hw_limit=3.125
        #       → safe_accel = 0.50 × (3.125/4.01) × 0.9 = 0.350
        #
        # A fixed accel_scale of 0.12 satisfies both with comfortable margin:
        #   joint2: 2.92 × (0.12/0.25) = 1.40 < 1.875  (25 % headroom)
        #   joint4: 4.01 × (0.12/0.50) = 0.96 < 3.125  (69 % headroom)
        #
        # vel_scale is kept at the caller's requested speed (up to 1.0).  For
        # the short ~10 cm descents and lifts the achievable TCP speed is
        # bounded by accel anyway (√(2·a·d)), so vel_scale has no effect there
        # but gives free speed on longer OMPL approach moves.
        _PILZ_FLOOR      = 0.1    # minimum velocity scaling
        _PILZ_ACCEL_FIXED = 0.10  # fixed accel derived from joint-limit analysis:
                                  # worst observed case joint2: 1.919 rad/s² actual at 0.12
                                  # → scaled to 0.10: 1.919×(0.10/0.12)=1.599 < 1.875 limit ✓

        pilz_vel_scale   = max(min(float(speed_scale), 1.0), _PILZ_FLOOR)
        pilz_accel_scale = _PILZ_ACCEL_FIXED

        self._node.get_logger().info(
            f"Attempting PILZ LIN planning "
            f"(vel_scale={pilz_vel_scale:.2f}, accel_scale={pilz_accel_scale:.2f})..."
        )

        try:
            # PlanRequestParameters(moveit, namespace) loads values from a YAML
            # namespace at construction time. Those keys are not present in the
            # node's parameter server, so every field falls back to an empty
            # default — including planning_pipeline, which becomes '' and causes
            # "No planning pipeline available for name ''".
            # Fix: construct with the default namespace, then set both
            # planning_pipeline and planner_id explicitly.
            pilz_params = PlanRequestParameters(self._moveit, "")
            pilz_params.planning_pipeline = "pilz_industrial_motion_planner"
            pilz_params.planner_id = "LIN"
            pilz_params.max_velocity_scaling_factor     = pilz_vel_scale
            pilz_params.max_acceleration_scaling_factor = pilz_accel_scale

            plan_result = self._arm.plan(single_plan_parameters=pilz_params)

        except Exception as exc:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message=f"PILZ LIN raised an exception during planning: {exc}",
            )

        if not plan_result:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="PILZ LIN failed to produce a plan.",
            )

        if plan_result.trajectory is None:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="PILZ LIN returned no trajectory.",
            )

        trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
        joint_trajectory = trajectory_msg.joint_trajectory

        if not joint_trajectory.points:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message="PILZ LIN returned an empty trajectory.",
            )

        trajectory_is_safe, safety_message = self.final_goal_respects_joint_margin(
            joint_trajectory,
            margin_rad=margin_rad,
        )

        if not trajectory_is_safe:
            return PlanningResult(
                success=False,
                joint_trajectory=None,
                message=f"PILZ LIN trajectory failed joint safety check: {safety_message}",
            )

        self._node.get_logger().info("PILZ LIN planning succeeded.")
        return PlanningResult(
            success=True,
            joint_trajectory=joint_trajectory,
            message=(
                "PILZ LIN planning produced a straight Cartesian trajectory "
                "that passed safety screening."
            ),
        )

    def compute_approach_joint_state(
        self,
        grasp_pose: TaskSpacePoseMsg,
        approach_pose: TaskSpacePoseMsg,
        max_attempts: int = 10,
        speed_scale: float = 1.0,
    ) -> tuple[list[str], list[float]] | tuple[None, None]:
        """
        Derive the approach joint configuration by planning OMPL to the approach pose.

        With the fixed _PILZ_ACCEL_FIXED = 0.12 in plan_to_task_pose_with_constraint,
        all approach configurations are safe for the subsequent PILZ LIN descent,
        so the first successful OMPL sample is accepted without a dry-run check.

        Inputs:
            grasp_pose: Target grasp pose in task space (reserved for future use).
            approach_pose: Approach pose directly above the grasp pose.
            max_attempts: Maximum OMPL retry cycles.
            speed_scale: Unused — kept for API compatibility with ComputeApproachJoints.srv.

        Returns:
            tuple[list[str], list[float]] | tuple[None, None]:
                Joint names and approach joint positions if successful,
                otherwise (None, None).
        """
        approach_target = self.create_pose_target(approach_pose)

        for attempt in range(1, max_attempts + 1):
            self._node.get_logger().info(
                f"compute_approach_joint_state attempt {attempt}/{max_attempts}: "
                "planning OMPL path to approach pose."
            )

            self._arm.set_start_state_to_current_state()
            self._arm.set_goal_state(
                pose_stamped_msg=approach_target,
                pose_link="panda_hand_tcp",
            )

            ompl_params = PlanRequestParameters(self._moveit, "")
            ompl_params.planning_pipeline = "ompl"
            ompl_params.planner_id = "RRTConnectkConfigDefault"
            ompl_params.planning_time = 15.0
            ompl_result = self._arm.plan(single_plan_parameters=ompl_params)

            if not ompl_result or ompl_result.trajectory is None:
                self._node.get_logger().warn(
                    f"compute_approach_joint_state attempt {attempt}: "
                    "OMPL plan to approach pose failed."
                )
                continue

            ompl_traj = ompl_result.trajectory.get_robot_trajectory_msg().joint_trajectory

            if not ompl_traj.points:
                self._node.get_logger().warn(
                    f"compute_approach_joint_state attempt {attempt}: "
                    "OMPL trajectory to approach pose was empty."
                )
                continue

            joint_names = list(ompl_traj.joint_names)
            approach_joint_positions = list(ompl_traj.points[-1].positions)

            self._node.get_logger().info(
                f"compute_approach_joint_state attempt {attempt}: "
                "approach joint state found via OMPL."
            )
            return joint_names, approach_joint_positions

        self._node.get_logger().error(
            "compute_approach_joint_state: all attempts exhausted without a valid solution."
        )
        return None, None

    def compute_approach_joints_callback(
        self,
        request: ComputeApproachJoints.Request,
        response: ComputeApproachJoints.Response,
    ) -> ComputeApproachJoints.Response:
        """
        Handle a request to derive a compatible approach joint configuration.

        Inputs:
            request: Service request containing grasp_pose, approach_pose,
                and max_attempts.
            response: Service response to populate.

        Returns:
            ComputeApproachJoints.Response: Populated response.
        """
        self._node.get_logger().info("Received compute_approach_joints request.")

        max_attempts = request.max_attempts if request.max_attempts > 0 else 10
        speed_scale  = float(request.speed_scale) if request.speed_scale > 0.0 else 1.0

        joint_names, joint_positions = self.compute_approach_joint_state(
            grasp_pose=request.grasp_pose,
            approach_pose=request.approach_pose,
            max_attempts=max_attempts,
            speed_scale=speed_scale,  # forwarded for API completeness; not used internally
        )

        if joint_names is None:
            response.success = False
            response.message = (
                "Failed to derive a compatible approach joint state after "
                f"{max_attempts} attempts."
            )
            return response

        response.success = True
        response.message = "Approach joint state derived successfully via reversed PILZ LIN."
        response.joint_names = joint_names
        response.joint_positions = joint_positions

        return response

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