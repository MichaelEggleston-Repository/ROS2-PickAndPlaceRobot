import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformListener, TransformException

import math
import time
import threading

from std_msgs.msg import String
from sensor_msgs.msg import JointState

from pick_place_robot.panda_arm_control import (
    ARM_JOINT_NAMES,
    HOME_JOINT_POSITIONS,
    PandaArmControl,
)

from pick_place_robot.panda_gripper_control import (
    GRIPPER_MOVE_DURATION_SEC,
    GRIPPER_OPEN_POSITION,
    PandaGripperControl,
)

from pick_place_interfaces.srv import (
    ComputeApproachJoints,
    ExecuteTaskPose,
    MoveGripper,
    PlanToJointPositions,
    PlanToTaskPose,
    ExecuteHome,
)
from pick_place_interfaces.msg import TaskSpacePose as TaskSpacePoseMsg

class PandaCoordinatorNode(Node):
    def __init__(self):
        """
        Create the Panda coordinator node and attach the reusable arm and gripper control objects.

        Inputs:
            None

        Returns:
            None
        """
        # Give the node a stable, descriptive ROS name.
        super().__init__("panda_coordinator_node")

        # Attach reusable control objects to this coordinator node.
        self._status_publisher = self.create_publisher(
            String,
            "panda_coordinator/status",
            10,
        )
        self._status_timer = self.create_timer(
            1.0,
            self.publish_status,
        )
        self._status_text = "starting"

        self._service_callback_group = ReentrantCallbackGroup()
        self._planner_client_callback_group = ReentrantCallbackGroup()
        self._arm_action_callback_group = ReentrantCallbackGroup()
        self._gripper_action_callback_group = ReentrantCallbackGroup()

        self._plan_to_task_pose_client = self.create_client(
            PlanToTaskPose,
            "plan_to_task_pose",
            callback_group=self._planner_client_callback_group,
        )

        self._plan_to_joint_positions_client = self.create_client(
            PlanToJointPositions,
            "plan_to_joint_positions",
            callback_group=self._planner_client_callback_group,
        )

        self._compute_approach_joints_client = self.create_client(
            ComputeApproachJoints,
            "compute_approach_joints",
            callback_group=self._planner_client_callback_group,
        )
        
        self._execute_task_pose_service = self.create_service(
            ExecuteTaskPose,
            "execute_task_pose",
            self.execute_task_pose_callback,
            callback_group=self._service_callback_group,
        )

        self._execute_home_position_service = self.create_service(
            ExecuteHome,
            "execute_home_position",
            self.execute_home_position_callback,
            callback_group=self._service_callback_group,
        )

        self._move_gripper_service = self.create_service(
            MoveGripper,
            "move_gripper",
            self.move_gripper_callback,
            callback_group=self._service_callback_group,
        )

        self._arm = PandaArmControl(
            self,
            callback_group=self._arm_action_callback_group,
        )

        self._gripper = PandaGripperControl(
            self,
            callback_group=self._gripper_action_callback_group,
        )

        # Subscribe to joint states so we can log actual finger positions
        # immediately before each gripper command.  This lets us confirm
        # whether the fingers are already at the target position (which would
        # explain "goal reached but no movement") or genuinely open when a
        # close is commanded.
        self._finger_positions: dict[str, float] = {}
        self._joint_state_subscription = self.create_subscription(
            JointState,
            "/joint_states",
            self._joint_state_callback,
            10,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._robot_base_frame = "panda_link0"
        self._robot_tool_frame = "panda_hand_tcp"

        self._task_pose_position_tolerance_m = 0.010
        self._task_pose_orientation_tolerance_rad = math.radians(8.0)
        self._task_pose_max_attempts = 3
        self._post_motion_settle_time_sec = 0.25
        self._large_position_error_abort_m = 0.050
        self._large_orientation_error_abort_rad = math.radians(20.0)


        self._is_ready = False

    def _joint_state_callback(self, msg: JointState) -> None:
        """
        Cache the latest position for each finger joint from /joint_states.

        Inputs:
            msg: Latest joint state message.

        Returns:
            None
        """
        for name, position in zip(msg.name, msg.position):
            if "finger" in name:
                self._finger_positions[name] = position

    def publish_status(self) -> None:
        """
        Publish the current coordinator status for other nodes.

        Inputs:
            None

        Returns:
            None
        """
        msg = String()
        msg.data = self._status_text
        self._status_publisher.publish(msg)

    def set_status(self, status_text: str) -> None:
        """
        Update and immediately publish the coordinator status.

        Inputs:
            status_text: Human-readable coordinator status.

        Returns:
            None
        """
        self._status_text = status_text
        self.publish_status()
        
    def wait_for_control_servers(self) -> bool:
        """
        Wait for both the Panda arm and gripper action servers to become available.

        Inputs:
            None

        Returns:
            bool: True if both servers are available, otherwise False.
        """
        arm_ready = self._arm.wait_for_server()
        gripper_ready = self._gripper.wait_for_server()
        if arm_ready and gripper_ready:
            # Allow the ros2_control hardware interfaces time to fully claim
            # their joints after the action servers come up.  Gazebo runs at
            # ~10 % real-time so the controller manager can still be settling
            # joint ownership when the action server first reports available.
            # Without this dwell, the first gripper trajectory is rejected with
            # "Joints on incoming trajectory don't match the controller joints."
            self.get_logger().info(
                "Control servers ready — waiting 3 s for controller hardware "
                "claim to settle before accepting requests."
            )
            time.sleep(3.0)
        return arm_ready and gripper_ready
    
    def is_ready(self) -> bool:
        """
        Report whether the coordinator is ready to accept robot requests.

        Inputs:
            None

        Returns:
            bool: True if startup completed successfully, otherwise False.
        """
        return self._is_ready
    
    def require_ready(self) -> bool:
        """
        Check whether the coordinator is ready to process a request.

        Inputs:
            None

        Returns:
            bool: True if ready, otherwise False.
        """
        if self._is_ready:
            return True

        self.get_logger().warn(
            "Coordinator request rejected because startup sequence is not complete."
        )
        return False
    
    def wait_for_planner_service(self) -> bool:
        """
        Wait for the planner services to become available.

        Inputs:
            None

        Returns:
            bool: True if the services became available, otherwise False.
        """
        self.get_logger().info("Waiting for planner services...")

        while rclpy.ok():
            task_pose_ready = self._plan_to_task_pose_client.wait_for_service(timeout_sec=1.0)
            joint_goal_ready = self._plan_to_joint_positions_client.wait_for_service(timeout_sec=1.0)
            approach_joints_ready = self._compute_approach_joints_client.wait_for_service(timeout_sec=1.0)

            if task_pose_ready and joint_goal_ready and approach_joints_ready:
                self.get_logger().info("Planner services are available.")
                return True

            self.get_logger().info(
                "Planner services not available yet, waiting again..."
            )

        return False
    
    def lookup_current_tool_transform(
        self,
        timeout_sec: float = 1.0,
    ) -> TransformStamped | None:
        """
        Look up the latest robot base to tool transform.

        Inputs:
            timeout_sec: Maximum TF wait time in seconds.

        Returns:
            TransformStamped | None: Latest transform if available.
        """
        deadline = time.time() + timeout_sec

        while rclpy.ok() and time.time() < deadline:
            try:
                return self._tf_buffer.lookup_transform(
                    self._robot_base_frame,
                    self._robot_tool_frame,
                    rclpy.time.Time(),
                )
            except TransformException:
                time.sleep(0.02)

        self.get_logger().warn(
            f"Timed out looking up TF {self._robot_base_frame} -> {self._robot_tool_frame}."
        )
        return None
    
    def pose_to_orientation_xyzw(
        self,
        pose: TaskSpacePoseMsg,
    ) -> tuple[float, float, float, float]:
        """
        Convert a task-space pose orientation into a quaternion tuple.

        Inputs:
            pose: The task-space pose containing roll, pitch, and yaw.

        Returns:
            tuple[float, float, float, float]:
                Quaternion as (x, y, z, w).
        """
        half_roll = pose.roll * 0.5
        half_pitch = pose.pitch * 0.5
        half_yaw = pose.yaw * 0.5

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

    def normalize_quaternion(
        self,
        qx: float,
        qy: float,
        qz: float,
        qw: float,
    ) -> tuple[float, float, float, float]:
        """
        Normalize a quaternion.

        Inputs:
            qx: Quaternion x.
            qy: Quaternion y.
            qz: Quaternion z.
            qw: Quaternion w.

        Returns:
            tuple[float, float, float, float]: Normalized quaternion.
        """
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)

        if norm <= 1e-12:
            return 0.0, 0.0, 0.0, 1.0

        return qx / norm, qy / norm, qz / norm, qw / norm

    def compute_position_error_m(
        self,
        pose: TaskSpacePoseMsg,
        transform: TransformStamped,
    ) -> float:
        """
        Compute Euclidean position error between a requested task pose and a TF transform.

        Inputs:
            pose: Requested task-space pose.
            transform: Measured base-to-tool TF transform.

        Returns:
            float: Position error in meters.
        """
        dx = transform.transform.translation.x - pose.x
        dy = transform.transform.translation.y - pose.y
        dz = transform.transform.translation.z - pose.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def compute_orientation_error_rad(
        self,
        pose: TaskSpacePoseMsg,
        transform: TransformStamped,
    ) -> float:
        """
        Compute the shortest angular distance between requested and measured orientation.

        Inputs:
            pose: Requested task-space pose.
            transform: Measured base-to-tool TF transform.

        Returns:
            float: Orientation error in radians.
        """
        requested_qx, requested_qy, requested_qz, requested_qw = self.pose_to_orientation_xyzw(
            pose
        )
        actual_qx = transform.transform.rotation.x
        actual_qy = transform.transform.rotation.y
        actual_qz = transform.transform.rotation.z
        actual_qw = transform.transform.rotation.w

        requested_qx, requested_qy, requested_qz, requested_qw = self.normalize_quaternion(
            requested_qx,
            requested_qy,
            requested_qz,
            requested_qw,
        )
        actual_qx, actual_qy, actual_qz, actual_qw = self.normalize_quaternion(
            actual_qx,
            actual_qy,
            actual_qz,
            actual_qw,
        )

        dot = (
            requested_qx * actual_qx
            + requested_qy * actual_qy
            + requested_qz * actual_qz
            + requested_qw * actual_qw
        )
        dot = max(-1.0, min(1.0, abs(dot)))
        return 2.0 * math.acos(dot)
    
    def verify_task_pose_reached(
        self,
        pose: TaskSpacePoseMsg,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
    ) -> tuple[bool, str, float | None, float | None]:
        """
        Verify the current TCP pose against the requested task-space pose.

        Inputs:
            pose: Requested task-space pose.
            position_tolerance_m: Allowed position error in meters.
            orientation_tolerance_rad: Allowed orientation error in radians.

        Returns:
            tuple[bool, str, float | None, float | None]:
                Verification success, status message, measured position error,
                and measured orientation error.
        """
        transform = self.lookup_current_tool_transform(timeout_sec=1.0)

        if transform is None:
            return False, "Could not verify final TCP pose because TF lookup failed.", None, None

        position_error_m = self.compute_position_error_m(pose, transform)
        orientation_error_rad = self.compute_orientation_error_rad(pose, transform)

        measured_x = transform.transform.translation.x
        measured_y = transform.transform.translation.y
        measured_z = transform.transform.translation.z

        message = (
            "Final TCP verification: "
            f"requested=({pose.x:.4f}, {pose.y:.4f}, {pose.z:.4f}), "
            f"actual=({measured_x:.4f}, {measured_y:.4f}, {measured_z:.4f}), "
            f"position_error_m={position_error_m:.4f}, "
            f"orientation_error_deg={math.degrees(orientation_error_rad):.2f}"
        )

        within_tolerance = (
            position_error_m <= position_tolerance_m
            and orientation_error_rad <= orientation_tolerance_rad
        )

        return within_tolerance, message, position_error_m, orientation_error_rad

    def run_startup_sequence(self) -> bool:
        """
        Move the arm to the home joint configuration at node startup.

        Uses an UNPLANNED direct joint command (move_home_unplanned) because
        the arm is in a known-safe state at startup and MoveIt is not yet
        fully initialised.  This is the ONLY place move_home_unplanned should
        ever be called.  All post-startup homing must go through
        plan_and_move_home() so that collision-aware path planning is active.

        Inputs:
            None

        Returns:
            bool: True if the home motion succeeded, otherwise False.
        """
        self.get_logger().info("Starting coordinator startup sequence: moving to home...")

        if not self._arm.move_home_unplanned():
            self.get_logger().error("Coordinator failed during startup home motion.")
            return False

        self.get_logger().info("Coordinator startup sequence completed successfully.")
        return True
    
    def create_home_pose(self) -> TaskSpacePoseMsg:
        """
        Create the nominal home task-space pose for planned recovery.

        Inputs:
            None

        Returns:
            TaskSpacePoseMsg: Home pose for planner-based recovery.
        """
        return TaskSpacePoseMsg(
            x=0.45,
            y=0.00,
            z=0.55,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )
    
    def create_grasp_pose(self, object_pose: TaskSpacePoseMsg) -> TaskSpacePoseMsg:
        """
        Create the grasp pose at the object.

        Inputs:
            object_pose: The detected object pose in task space.

        Returns:
            TaskSpacePoseMsg: The grasp pose at object height.
        """
        return TaskSpacePoseMsg(
            x=object_pose.x,
            y=object_pose.y,
            z=object_pose.z,
            roll=object_pose.roll,
            pitch=object_pose.pitch,
            yaw=object_pose.yaw,
        )

    def create_place_pose(
        self,
        x: float,
        y: float,
        z: float,
        roll: float = 3.14159,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> TaskSpacePoseMsg:
        """
        Create the final place pose in task space.

        Inputs:
            x: Target x position in meters.
            y: Target y position in meters.
            z: Target z position in meters.
            roll: Rotation about the x-axis in radians.
            pitch: Rotation about the y-axis in radians.
            yaw: Rotation about the z-axis in radians.

        Returns:
            TaskSpacePoseMsg: The place pose for dropping the object.
        """
        return TaskSpacePoseMsg(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
    
    def offset_pose_z(self, base_pose: TaskSpacePoseMsg, z_offset: float) -> TaskSpacePoseMsg:
        """
        Create a new task-space pose by offsetting an existing pose in z.

        Inputs:
            base_pose: The starting pose.
            z_offset: The z offset to apply in meters.

        Returns:
            TaskSpacePoseMsg: A new pose with the adjusted z value.
        """
        return TaskSpacePoseMsg(
            x=base_pose.x,
            y=base_pose.y,
            z=base_pose.z + z_offset,
            roll=base_pose.roll,
            pitch=base_pose.pitch,
            yaw=base_pose.yaw,
        )
    
    def create_pre_grasp_pose(
        self,
        grasp_pose: TaskSpacePoseMsg,
        z_offset: float = 0.10,
    ) -> TaskSpacePoseMsg:
        """
        Create a safe approach pose above the grasp pose.

        Inputs:
            grasp_pose: The task-space pose where the gripper should perform the grasp.
            z_offset: Vertical offset above the grasp pose in meters.

        Returns:
            TaskSpacePoseMsg: A pose above the grasp point for safe approach.
        """
        # Approach from above to reduce the chance of colliding with the object or conveyor.
        return self.offset_pose_z(grasp_pose, z_offset)

    def create_lift_pose(
        self,
        grasp_pose: TaskSpacePoseMsg,
        z_offset: float = 0.12,
    ) -> TaskSpacePoseMsg:
        """
        Create a lifted retreat pose above the grasp pose after the object is picked.

        Inputs:
            grasp_pose: The task-space pose where the object is grasped.
            z_offset: Vertical retreat offset above the grasp pose in meters.

        Returns:
            TaskSpacePoseMsg: A pose above the grasp point for safe retreat.
        """
        # Lift vertically before traveling so the object clears the surface safely.
        return self.offset_pose_z(grasp_pose, z_offset)

    def create_pre_place_pose(
        self,
        place_pose: TaskSpacePoseMsg,
        z_offset: float = 0.10,
    ) -> TaskSpacePoseMsg:
        """
        Create a safe approach pose above the place pose.

        Inputs:
            place_pose: The final task-space pose where the object should be placed.
            z_offset: Vertical offset above the place pose in meters.

        Returns:
            TaskSpacePoseMsg: A pose above the place point for safe approach.
        """
        # Approach the placement point from above before descending to release.
        return self.offset_pose_z(place_pose, z_offset)

    def create_place_depart_pose(
        self,
        place_pose: TaskSpacePoseMsg,
        z_offset: float = 0.10,
    ) -> TaskSpacePoseMsg:
        """
        Create a safe retreat pose above the place pose after releasing the object.

        Inputs:
            place_pose: The final task-space pose where the object was placed.
            z_offset: Vertical retreat offset above the place pose in meters.

        Returns:
            TaskSpacePoseMsg: A pose above the place point for safe departure.
        """
        # Retreat upward after release to avoid brushing the placed object.
        return self.offset_pose_z(place_pose, z_offset)
    
    def request_plan_to_task_pose(
        self,
        pose: TaskSpacePoseMsg,
        use_orientation_constraint: bool = False,
        orientation_tolerance_rad: float = 1.0,
        speed_scale: float = 1.0,
    ) -> PlanToTaskPose.Response | None:
        """
        Request a joint-trajectory plan for a task-space pose from the planner service.

        Inputs:
            pose: Target task-space pose message.
            use_orientation_constraint: True to request PILZ LIN planning.
            orientation_tolerance_rad: Allowed orientation error in radians about
                each axis when constrained planning is requested.
            speed_scale: Velocity and acceleration scaling factor forwarded to the
                planner. For PILZ LIN, scaling is applied at planning time because
                PILZ validates joint velocity/acceleration limits during trajectory
                generation. Passing a value < 1.0 prevents limit violations.

        Returns:
            PlanToTaskPose.Response | None: Service response if successful,
            otherwise None.
        """
        request = PlanToTaskPose.Request()
        request.pose = pose
        request.speed_scale = speed_scale
        request.use_orientation_constraint = use_orientation_constraint
        request.orientation_tolerance_rad = orientation_tolerance_rad

        future = self._plan_to_task_pose_client.call_async(request)

        while rclpy.ok() and not future.done():
            time.sleep(0.01)

        if not future.done():
            self.get_logger().warn("Plan-to-task-pose service call did not complete.")
            return None

        if future.exception() is not None:
            self.get_logger().error(
                f"Plan-to-task-pose service call raised an exception: {future.exception()}"
            )
            return None

        response = future.result()

        if response is None:
            self.get_logger().warn("Plan-to-task-pose service returned no response.")
            return None

        return response
    
    def request_plan_to_joint_positions(
        self,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> PlanToJointPositions.Response | None:
        """
        Request a joint-trajectory plan for a joint-space target from the planner service.

        Inputs:
            joint_names: Ordered list of joint names.
            joint_positions: Ordered list of target joint positions in radians.

        Returns:
            PlanToJointPositions.Response | None:
                Service response if successful, otherwise None.
        """
        request = PlanToJointPositions.Request()
        request.joint_names = joint_names
        request.joint_positions = joint_positions

        future = self._plan_to_joint_positions_client.call_async(request)

        while rclpy.ok() and not future.done():
            time.sleep(0.01)

        if not future.done():
            self.get_logger().warn(
                "Plan-to-joint-positions service call did not complete."
            )
            return None

        if future.exception() is not None:
            self.get_logger().error(
                f"Plan-to-joint-positions service call raised an exception: {future.exception()}"
            )
            return None

        response = future.result()

        if response is None:
            self.get_logger().warn(
                "Plan-to-joint-positions service returned no response."
            )
            return None

        return response
    
    def request_compute_approach_joints(
        self,
        grasp_pose: TaskSpacePoseMsg,
        approach_pose: TaskSpacePoseMsg,
        max_attempts: int = 10,
        speed_scale: float = 1.0,
    ) -> "ComputeApproachJoints.Response | None":
        """
        Ask the planner to derive an approach joint configuration that is
        guaranteed to allow a PILZ LIN descent at the requested speed.

        Inputs:
            grasp_pose: Target grasp pose in task space.
            approach_pose: Approach pose directly above the grasp pose.
            max_attempts: Maximum OMPL + PILZ-dry-run retry cycles in the planner.
            speed_scale: The speed at which the subsequent PILZ LIN descent will
                be executed.  The planner performs a dry-run at this speed so
                only configurations that pass the joint-limit check are returned.

        Returns:
            ComputeApproachJoints.Response | None:
                Service response if successful, otherwise None.
        """
        request = ComputeApproachJoints.Request()
        request.grasp_pose = grasp_pose
        request.approach_pose = approach_pose
        request.max_attempts = max_attempts
        request.speed_scale = speed_scale

        future = self._compute_approach_joints_client.call_async(request)

        while rclpy.ok() and not future.done():
            time.sleep(0.01)

        if not future.done():
            self.get_logger().warn(
                "compute_approach_joints service call did not complete."
            )
            return None

        if future.exception() is not None:
            self.get_logger().error(
                f"compute_approach_joints service call raised an exception: "
                f"{future.exception()}"
            )
            return None

        response = future.result()

        if response is None:
            self.get_logger().warn(
                "compute_approach_joints service returned no response."
            )
            return None

        return response

    def plan_and_move_to_joint_positions(
        self,
        joint_names: list[str],
        joint_positions: list[float],
        speed_scale: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Plan and execute a motion to a joint-space target.

        Plans up to _JOINT_LIMIT_REPLAN_ATTEMPTS times if OMPL returns a
        trajectory whose final configuration lands any joint within
        _JOINT_LIMIT_MARGIN_RAD of its limit.  The margin check prevents
        subsequent constrained moves (PILZ LIN) from becoming infeasible
        because the arm is cornered near a limit.

        Inputs:
            joint_names: Ordered list of joint names.
            joint_positions: Ordered list of target joint positions in radians.
            speed_scale: Motion speed scale factor for trajectory execution.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        for attempt_index in range(1, self._JOINT_LIMIT_REPLAN_ATTEMPTS + 1):
            self.get_logger().info(
                f"Joint-space planning attempt "
                f"{attempt_index}/{self._JOINT_LIMIT_REPLAN_ATTEMPTS}."
            )

            response = self.request_plan_to_joint_positions(
                joint_names=joint_names,
                joint_positions=joint_positions,
            )

            if response is None:
                return False, "No joint-space planning response was received."

            self.get_logger().info(f"Joint-space planning success: {response.success}")
            self.get_logger().info(f"Joint-space planning message: {response.message}")

            if not response.success:
                return False, f"Joint-space planning failed: {response.message}"

            if not response.joint_trajectory.points:
                return False, "Joint-space planning returned no joint trajectory."

            # Inspect the final trajectory waypoint for joint-limit proximity.
            traj_joint_names = list(response.joint_trajectory.joint_names)
            final_positions = list(response.joint_trajectory.points[-1].positions)

            near_limit, violating = self._check_joint_limit_clearances(
                traj_joint_names,
                final_positions,
            )

            if near_limit:
                self.get_logger().warn(
                    f"Joint-space plan rejected: joint(s) near limit: {violating}.  "
                    f"Requesting a fresh OMPL sample "
                    f"(attempt {attempt_index}/{self._JOINT_LIMIT_REPLAN_ATTEMPTS})."
                )
                if attempt_index < self._JOINT_LIMIT_REPLAN_ATTEMPTS:
                    continue
                return False, (
                    f"Joint-space planning produced a near-limit configuration after "
                    f"{self._JOINT_LIMIT_REPLAN_ATTEMPTS} attempts.  "
                    f"Violating joints: {violating}."
                )

            # Configuration is acceptable — execute.
            self.get_logger().info(
                f"Joint-space plan accepted.  Executing trajectory with "
                f"{len(response.joint_trajectory.points)} points."
            )

            motion_succeeded = self._arm.move_to_joint_trajectory(
                response.joint_trajectory,
                speed_scale=speed_scale,
            )

            if not motion_succeeded:
                return False, "Arm motion to joint positions failed after successful planning."

            time.sleep(self._post_motion_settle_time_sec)

            return True, "Motion to joint positions succeeded."

        return False, (
            f"Joint-space planning exhausted all "
            f"{self._JOINT_LIMIT_REPLAN_ATTEMPTS} replan attempts."
        )

    def execute_pose_once(
        self,
        pose: TaskSpacePoseMsg,
        speed_scale: float = 1.0,
        use_orientation_constraint: bool = False,
        orientation_tolerance_rad: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Plan and execute a single task-space motion attempt without internal retries.

        Inputs:
            pose: The target end-effector pose in task space.
            speed_scale: Motion speed scale factor for trajectory execution.
            use_orientation_constraint: True to request constrained planning.
            orientation_tolerance_rad: Allowed orientation error in radians for
                constrained planning.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        response = self.request_plan_to_task_pose(
            pose,
            use_orientation_constraint=use_orientation_constraint,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

        if response is None:
            return False, "No planning response was received."

        self.get_logger().info(f"Planning success: {response.success}")
        self.get_logger().info(f"Planning message: {response.message}")

        if not response.success:
            return False, "Planning failed."

        if not response.joint_trajectory.points:
            return False, "No joint trajectory was returned."

        self.get_logger().info(
            f"Executing planned joint trajectory with "
            f"{len(response.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            response.joint_trajectory,
            speed_scale=speed_scale,
        )

        if not motion_succeeded:
            return False, "Arm motion failed after successful planning."

        time.sleep(self._post_motion_settle_time_sec)

        return True, "Single execution attempt completed."
    
    def plan_and_move_home(
        self,
        speed_scale: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Plan and execute motion to the nominal home joint configuration.

        Inputs:
            speed_scale: Motion speed scale factor for trajectory execution.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        self.get_logger().info("Attempting planned recovery move to home joint configuration.")

        response = self.request_plan_to_joint_positions(
            joint_names=ARM_JOINT_NAMES,
            joint_positions=HOME_JOINT_POSITIONS,
        )

        if response is None:
            return False, "No joint-space planning response was received for home recovery."

        self.get_logger().info(f"Planning success: {response.success}")
        self.get_logger().info(f"Planning message: {response.message}")

        if not response.success:
            return False, "Joint-space planning to home failed."

        if not response.joint_trajectory.points:
            return False, "Joint-space planning to home returned no joint trajectory."

        self.get_logger().info(
            f"Executing planned home joint trajectory with "
            f"{len(response.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            response.joint_trajectory,
            speed_scale=speed_scale,
        )

        if not motion_succeeded:
            return False, "Planned home joint motion failed after successful planning."

        time.sleep(self._post_motion_settle_time_sec)

        return True, "Planned recovery move to home joint configuration succeeded."
    
    def plan_and_move_to_pose(
        self,
        pose: TaskSpacePoseMsg,
        speed_scale: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Request a plan to a task-space pose, execute it, verify final TCP pose,
        and replan a bounded number of times if needed.

        Inputs:
            pose: The target end-effector pose in task space.
            speed_scale: Motion speed scale factor for trajectory execution.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        if not self.require_ready():
            return False, "Coordinator is not ready. Startup sequence is not complete."

        last_message = "Task-space motion did not complete successfully."
        last_failure_was_tolerance_only = False
        last_verify_message = ""

        for attempt_index in range(1, self._task_pose_max_attempts + 1):
            self.get_logger().info(
                f"Task-space execution attempt {attempt_index}/{self._task_pose_max_attempts}."
            )

            response = self.request_plan_to_task_pose(
                pose,
                use_orientation_constraint=False,
            )

            if response is None:
                last_message = "Cannot execute motion because no planning response was received."
                last_failure_was_tolerance_only = False
                self.get_logger().error(last_message)
                continue

            self.get_logger().info(f"Planning success: {response.success}")
            self.get_logger().info(f"Planning message: {response.message}")

            if not response.success:
                last_message = "Cannot execute motion because planning failed."
                last_failure_was_tolerance_only = False
                self.get_logger().error(last_message)
                continue

            if not response.joint_trajectory.points:
                last_message = "Cannot execute motion because no joint trajectory was returned."
                last_failure_was_tolerance_only = False
                self.get_logger().error(last_message)
                continue

            # Check the final trajectory configuration for joint-limit proximity
            # before committing to execution.
            traj_joint_names = list(response.joint_trajectory.joint_names)
            final_positions = list(response.joint_trajectory.points[-1].positions)

            near_limit, violating = self._check_joint_limit_clearances(
                traj_joint_names,
                final_positions,
            )

            if near_limit:
                last_message = (
                    f"OMPL plan rejected: joint(s) near limit: {violating}.  "
                    f"Requesting a fresh sample."
                )
                last_failure_was_tolerance_only = False
                self.get_logger().warn(last_message)
                continue

            self.get_logger().info(
                f"Joint-limit check passed.  Executing planned trajectory with "
                f"{len(response.joint_trajectory.points)} points."
            )

            motion_succeeded = self._arm.move_to_joint_trajectory(
                response.joint_trajectory,
                speed_scale=speed_scale,
            )

            if not motion_succeeded:
                last_message = "Arm motion failed after successful planning."
                last_failure_was_tolerance_only = False
                self.get_logger().error(last_message)
                continue

            time.sleep(self._post_motion_settle_time_sec)

            reached_pose, verify_message, position_error_m, orientation_error_rad = (
                self.verify_task_pose_reached(
                    pose,
                    position_tolerance_m=self._task_pose_position_tolerance_m,
                    orientation_tolerance_rad=self._task_pose_orientation_tolerance_rad,
                )
            )

            if reached_pose:
                success_message = (
                    "Planned arm motion completed successfully and final TCP pose "
                    "was within tolerance. "
                    + verify_message
                )
                self.get_logger().info(success_message)
                return True, success_message

            last_message = (
                "Arm motion executed but final TCP pose was outside tolerance. "
                + verify_message
            )
            last_failure_was_tolerance_only = True
            last_verify_message = verify_message
            self.get_logger().error(last_message)

            large_error_detected = (
                position_error_m is not None
                and orientation_error_rad is not None
                and (
                    position_error_m > self._large_position_error_abort_m
                    or orientation_error_rad > self._large_orientation_error_abort_rad
                )
            )

            if large_error_detected:
                self.get_logger().error(
                    "Final TCP error was large. Attempting planned recovery move to home "
                    "before retrying the requested task pose."
                )

                recovered_home, recovery_message = self.plan_and_move_home(
                    speed_scale=speed_scale,
                )

                if not recovered_home:
                    abort_message = (
                        "Aborting retries because final TCP error was too large and "
                        "planned recovery to home failed. "
                        f"Recovery message: {recovery_message}. "
                        + verify_message
                    )
                    self.get_logger().error(abort_message)
                    return False, abort_message

                self.get_logger().info(
                    "Planned recovery move to home succeeded. Will retry the requested task pose."
                )
                continue

        if last_failure_was_tolerance_only:
            self.get_logger().error(
                "Task-space motion exhausted normal retries while remaining outside "
                "tolerance. Attempting final planned recovery move to home before one "
                "last retry."
            )

            recovered_home, recovery_message = self.plan_and_move_home(
                speed_scale=speed_scale,
            )

            if not recovered_home:
                failure_message = (
                    f"Task-space motion failed after {self._task_pose_max_attempts} attempts, "
                    f"and planned recovery to home also failed. "
                    f"Recovery message: {recovery_message}. "
                    + last_message
                )
                self.get_logger().error(failure_message)
                return False, failure_message

            self.get_logger().info(
                "Final planned recovery move to home succeeded. Attempting one last retry "
                "for the requested task pose."
            )

            final_retry_succeeded, final_retry_message = self.execute_pose_once(
                pose,
                speed_scale=speed_scale,
                use_orientation_constraint=False,
            )

            if not final_retry_succeeded:
                failure_message = (
                    f"Task-space motion failed after {self._task_pose_max_attempts} attempts, "
                    f"and the final retry after planned home recovery also failed. "
                    f"Final retry message: {final_retry_message}"
                )
                self.get_logger().error(failure_message)
                return False, failure_message

            reached_pose, verify_message, _, _ = self.verify_task_pose_reached(
                pose,
                position_tolerance_m=self._task_pose_position_tolerance_m,
                orientation_tolerance_rad=self._task_pose_orientation_tolerance_rad,
            )

            if reached_pose:
                success_message = (
                    "Task-space motion succeeded after planned home recovery and final retry. "
                    + verify_message
                )
                self.get_logger().info(success_message)
                return True, success_message

            failure_message = (
                "Final retry after planned home recovery executed, but TCP remained outside "
                "tolerance. "
                + verify_message
            )
            self.get_logger().error(failure_message)
            return False, failure_message

        failure_message = (
            f"Task-space motion failed after {self._task_pose_max_attempts} attempts. "
            + last_message
        )
        self.get_logger().error(failure_message)
        return False, failure_message
    
    def plan_and_move_to_pose_with_orientation_constraint(
        self,
        pose: TaskSpacePoseMsg,
        orientation_tolerance_rad: float = 1.0,
        speed_scale: float = 1.0,
        approach_height_m: float = 0.0,
        approach_only: bool = False,
    ) -> tuple[bool, str]:
        """
        Execute a straight-line Cartesian motion to a task-space pose using PILZ LIN.

        When approach_height_m > 0 (used for grasp descents), a compatible approach
        joint configuration is first derived by working backwards from the target pose:
          1. Solve IK for the target (grasp) pose → grasp_joints.
          2. Plan a reversed PILZ LIN from grasp_joints to the approach pose above it
             → approach_joints (guaranteed on the same IK branch as the grasp).
          3. Use OMPL to move to approach_joints (joint-space goal, no IK branch issue).
          4. Execute a forward PILZ LIN to the target (grasp) pose.

        When approach_height_m == 0 (used for lift motions), a PILZ LIN is executed
        directly from the current robot state.

        Inputs:
            pose: The target end-effector pose in task space.
            orientation_tolerance_rad: Kept for API compatibility; PILZ LIN enforces
                a straight Cartesian line by design.
            speed_scale: Motion speed scale factor for trajectory execution.
            approach_height_m: Height in metres above the target pose at which the
                approach joint configuration should be computed. Pass 0.0 to skip
                approach planning (e.g. for lift motions).
            approach_only: When True and approach_height_m > 0, execute only the
                backward-IK + OMPL move to the approach pose and return immediately
                without descending with PILZ LIN. The caller issues a follow-up call
                with approach_height_m=0 to execute the PILZ LIN as a separate motion.
                This splits the two executions so that ros2_control fully settles
                between the OMPL stop and the PILZ start.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        if not self.require_ready():
            return False, "Coordinator is not ready. Startup sequence is not complete."

        # --- Backward-IK approach phase (grasp descents only) ---
        if approach_height_m > 0.0:
            approach_pose = self.offset_pose_z(pose, approach_height_m)

            self.get_logger().info(
                f"Computing compatible approach joint state "
                f"(approach_height_m={approach_height_m:.3f}m)."
            )

            approach_response = self.request_compute_approach_joints(
                grasp_pose=pose,
                approach_pose=approach_pose,
                max_attempts=10,
                speed_scale=speed_scale,
            )

            if approach_response is None or not approach_response.success:
                failure_message = (
                    "Failed to compute a compatible approach joint state. "
                    + (approach_response.message if approach_response else "No response.")
                )
                self.get_logger().error(failure_message)
                return False, failure_message

            self.get_logger().info(
                "Approach joint state derived. Moving to approach configuration via OMPL."
            )

            approach_succeeded, approach_message = self.plan_and_move_to_joint_positions(
                joint_names=list(approach_response.joint_names),
                joint_positions=list(approach_response.joint_positions),
                speed_scale=speed_scale,
            )

            if not approach_succeeded:
                failure_message = (
                    f"Failed to reach approach joint configuration: {approach_message}"
                )
                self.get_logger().error(failure_message)
                return False, failure_message

            if approach_only:
                # Caller will issue a separate PILZ LIN call once the gripper
                # and any other pre-grasp steps are ready.  Returning here
                # keeps the OMPL stop and the PILZ start as two independent
                # ros2_control execution cycles.
                success_message = (
                    "Reached PILZ-compatible approach configuration via OMPL. "
                    "PILZ LIN descent deferred to a follow-up call."
                )
                self.get_logger().info(success_message)
                return True, success_message

            self.get_logger().info(
                "Reached approach joint configuration. "
                "Executing PILZ LIN descent to target pose."
            )

        # --- PILZ LIN phase (descent from approach, or direct lift) ---
        last_message = "PILZ LIN motion did not complete successfully."

        for attempt_index in range(1, self._task_pose_max_attempts + 1):
            self.get_logger().info(
                f"PILZ LIN execution attempt {attempt_index}/{self._task_pose_max_attempts}."
            )

            response = self.request_plan_to_task_pose(
                pose,
                use_orientation_constraint=True,
                orientation_tolerance_rad=orientation_tolerance_rad,
                speed_scale=speed_scale,
            )

            if response is None:
                last_message = "Cannot execute PILZ LIN motion: no planning response received."
                self.get_logger().error(last_message)
                continue

            self.get_logger().info(f"Planning success: {response.success}")
            self.get_logger().info(f"Planning message: {response.message}")

            if not response.success:
                last_message = f"PILZ LIN planning failed: {response.message}"
                self.get_logger().error(last_message)
                continue

            if not response.joint_trajectory.points:
                last_message = "PILZ LIN planning returned no joint trajectory."
                self.get_logger().error(last_message)
                continue

            self.get_logger().info(
                f"Executing PILZ LIN trajectory with "
                f"{len(response.joint_trajectory.points)} points."
            )

            motion_succeeded = self._arm.move_to_joint_trajectory(
                response.joint_trajectory,
                speed_scale=speed_scale,
            )

            if not motion_succeeded:
                last_message = "Arm motion failed after successful PILZ LIN planning."
                self.get_logger().error(last_message)
                continue

            time.sleep(self._post_motion_settle_time_sec)

            reached_pose, verify_message, position_error_m, orientation_error_rad = (
                self.verify_task_pose_reached(
                    pose,
                    position_tolerance_m=self._task_pose_position_tolerance_m,
                    orientation_tolerance_rad=self._task_pose_orientation_tolerance_rad,
                )
            )

            if reached_pose:
                success_message = (
                    "PILZ LIN motion completed successfully and final TCP pose was "
                    "within tolerance. " + verify_message
                )
                self.get_logger().info(success_message)
                return True, success_message

            last_message = (
                "PILZ LIN motion executed but final TCP pose was outside tolerance. "
                + verify_message
            )
            self.get_logger().warn(last_message)

            large_error_detected = (
                position_error_m is not None
                and orientation_error_rad is not None
                and (
                    position_error_m > self._large_position_error_abort_m
                    or orientation_error_rad > self._large_orientation_error_abort_rad
                )
            )

            if large_error_detected:
                abort_message = (
                    "Aborting PILZ LIN retries due to large TCP error. " + verify_message
                )
                self.get_logger().error(abort_message)
                return False, abort_message

        failure_message = (
            f"PILZ LIN motion failed after {self._task_pose_max_attempts} attempts. "
            + last_message
        )
        self.get_logger().error(failure_message)
        return False, failure_message

    def execute_home_position_callback(
        self,
        request: ExecuteHome.Request,
        response: ExecuteHome.Response,
    ) -> ExecuteHome.Response:
        """
        Execute a planned move to the home joint configuration.

        Inputs:
            request: Execute-home service request.
            response: Service response to populate.

        Returns:
            ExecuteHome.Response: Populated execution result.
        """
        if not self.require_ready():
            response.success = False
            response.message = (
                "Coordinator is not ready. Startup sequence is not complete."
            )
            return response

        self.get_logger().info(
            "Received execute_home_position request."
        )

        success, message = self.plan_and_move_home(
            speed_scale=1.0,
        )

        response.success = success
        response.message = message

        self.get_logger().info(
            f"execute_home_position returning: success={response.success}, "
            f"message='{response.message}'"
        )

        return response
    
    def execute_task_pose_callback(
        self,
        request: ExecuteTaskPose.Request,
        response: ExecuteTaskPose.Response,
    ) -> ExecuteTaskPose.Response:
        """
        Plan and execute motion to a requested task-space pose.

        Inputs:
            request: Execute-task-pose service request.
            response: Execute-task-pose service response to populate.

        Returns:
            ExecuteTaskPose.Response: Populated execution result.
        """
        if not self.require_ready():
            response.success = False
            response.message = (
                "Coordinator is not ready. Startup sequence is not complete."
            )
            return response
        
        self.get_logger().info(
            f"execute_task_pose request received: "
            f"x={request.pose.x:.3f}, y={request.pose.y:.3f}, z={request.pose.z:.3f}, "
            f"roll={request.pose.roll:.3f}, pitch={request.pose.pitch:.3f}, yaw={request.pose.yaw:.3f}, "
            f"speed_scale={request.speed_scale:.3f}"
        )

        if request.use_orientation_constraint:
            self.get_logger().info(
                f"Using PILZ LIN with orientation constraint, tolerance="
                f"{request.orientation_tolerance_rad:.3f} rad, "
                f"approach_height_m={request.approach_height_m:.3f}m, "
                f"approach_only={request.approach_only}."
            )
            success, message = self.plan_and_move_to_pose_with_orientation_constraint(
                request.pose,
                orientation_tolerance_rad=request.orientation_tolerance_rad,
                speed_scale=request.speed_scale,
                approach_height_m=request.approach_height_m,
                approach_only=request.approach_only,
            )
        else:
            success, message = self.plan_and_move_to_pose(
                request.pose,
                speed_scale=request.speed_scale,
            )

        response.success = success
        response.message = message

        self.get_logger().info(
            f"execute_task_pose returning: success={response.success}, message='{response.message}'"
        )

        return response
    
    # Tolerance used when verifying that each finger reached its target after
    # a gripper command.  Any finger further than this from the target triggers
    # an automatic retry.
    # 10 mm accommodates the case where joint2 is pinned at the 0.040 m hard
    # limit after an asymmetric place (off-centre grasp pushes it to the upper
    # stop) but the open target is 0.034 m — a 6 mm gap that falls inside this
    # tolerance so the place-open is not falsely retried to exhaustion.
    # Grasp closes always use the separate grasp_tolerance_m path so the wider
    # default does not affect pick reliability.
    _GRIPPER_REACH_TOLERANCE_M = 0.010

    # Maximum number of combined-trajectory retry cycles for the gripper.
    # The JTC reports success when the trajectory duration elapses, not when
    # the joints physically reach the target.  In Gazebo, gz_ros2_control can
    # silently drop a position command for one joint during a given physics
    # step.  Retrying the same combined two-joint trajectory with a fresh
    # start_positions waypoint usually succeeds within a few attempts.
    _GRIPPER_MAX_RETRIES = 5

    # ── Joint-limit proximity guard ───────────────────────────────────────────
    # Position limits for each Panda arm joint, taken directly from
    # panda.urdf.xacro <limit> tags (lower, upper) in radians.
    _ARM_JOINT_LIMITS: dict[str, tuple[float, float]] = {
        "panda_joint1": (-2.897246558310587,    2.897246558310587),
        "panda_joint2": (-1.7627825445142729,   1.7627825445142729),
        "panda_joint3": (-2.897246558310587,    2.897246558310587),
        "panda_joint4": (-3.07177948351002,    -0.06981317007977318),
        "panda_joint5": (-2.897246558310587,    2.897246558310587),
        "panda_joint6": (-0.017453292519943295,  3.752457891787808),
        "panda_joint7": (-2.897246558310587,    2.897246558310587),
    }

    # Minimum clearance (radians) from any joint limit that a planned
    # trajectory's final configuration must have before execution is permitted.
    # 10 ° gives PILZ LIN and subsequent arm moves enough headroom to stay
    # feasible without meaningfully restricting the reachable workspace.
    _JOINT_LIMIT_MARGIN_RAD: float = math.radians(10.0)

    # Number of times to ask OMPL for a new plan when the previous one lands
    # a joint too close to its limit.  OMPL is non-deterministic, so a fresh
    # sample usually finds a better configuration within a couple of retries.
    _JOINT_LIMIT_REPLAN_ATTEMPTS: int = 3

    def _fingers_at_target(
        self,
        target_position: float,
        tolerance_m: float | None = None,
    ) -> bool:
        """
        Return True if every tracked finger joint is within tolerance of
        target_position.

        Inputs:
            target_position: Expected per-finger position in metres.
            tolerance_m: Position tolerance in metres.  Defaults to
                _GRIPPER_REACH_TOLERANCE_M (5 mm) when None.  Pass a larger
                value when closing on an object so the fingers stopping at the
                object surface (not at 0.0 m) does not trigger a false retry.

        Returns:
            bool: True if all fingers are close enough to the target.
        """
        if not self._finger_positions:
            # No data yet — assume OK to avoid blocking startup.
            return True

        effective_tolerance = (
            tolerance_m if tolerance_m is not None else self._GRIPPER_REACH_TOLERANCE_M
        )

        for name, position in self._finger_positions.items():
            if abs(position - target_position) > effective_tolerance:
                self.get_logger().warn(
                    f"Finger '{name}' is at {position:.4f}m but target is "
                    f"{target_position:.4f}m "
                    f"(error={abs(position - target_position):.4f}m > "
                    f"tolerance={effective_tolerance:.4f}m)."
                )
                return False

        return True

    def _check_joint_limit_clearances(
        self,
        joint_names: list[str],
        joint_positions: list[float],
    ) -> tuple[bool, list[str]]:
        """
        Check whether any arm joint in the given configuration is within
        _JOINT_LIMIT_MARGIN_RAD of its position limit.

        Inputs:
            joint_names: Names of the joints in the trajectory.
            joint_positions: Corresponding positions in radians.

        Returns:
            tuple[bool, list[str]]:
                (any_near_limit, list_of_violating_joint_names)
        """
        violating: list[str] = []
        for name, position in zip(joint_names, joint_positions):
            if name not in self._ARM_JOINT_LIMITS:
                continue
            lower, upper = self._ARM_JOINT_LIMITS[name]
            if min(position - lower, upper - position) < self._JOINT_LIMIT_MARGIN_RAD:
                violating.append(name)
        return len(violating) > 0, violating

    def move_gripper_callback(
        self,
        request: MoveGripper.Request,
        response: MoveGripper.Response,
    ) -> MoveGripper.Response:
        """
        Move the gripper to a requested finger separation width, with automatic
        retry when Gazebo silently ignores one finger joint's command.

        The requested width_m is the total gap between fingers. It is
        divided by two to get the per-finger position that the gripper
        controller expects.

        Inputs:
            request: MoveGripper service request containing width_m.
            response: Service response to populate.

        Returns:
            MoveGripper.Response: Populated result.
        """
        if not self.require_ready():
            response.success = False
            response.message = "Coordinator is not ready. Startup sequence is not complete."
            return response

        max_width_m = GRIPPER_OPEN_POSITION * 2.0

        if request.width_m < 0.0 or request.width_m > max_width_m:
            response.success = False
            response.message = (
                f"Requested gripper width {request.width_m:.4f}m is outside the "
                f"valid range [0.0, {max_width_m:.4f}]m."
            )
            self.get_logger().warn(response.message)
            return response

        finger_position = request.width_m / 2.0

        # When the caller supplies an expected object width, the fingers will
        # stop at the object surface rather than 0.0 m.  Use a tolerance that
        # accepts any position between 0.0 m and (half-width + margin) so both
        # the JTC goal check and our own post-motion verify pass cleanly.
        # When no object is expected (open, empty close, diagnostics) keep the
        # tight 5 mm default so real positioning errors are still caught.
        _GRASP_TOLERANCE_MARGIN_M = 0.005  # 5 mm margin above object half-width
        if request.expected_object_width_m > 0.0:
            grasp_tolerance_m = (
                request.expected_object_width_m / 2.0 + _GRASP_TOLERANCE_MARGIN_M
            )
            self.get_logger().info(
                f"Grasping object: expected_width={request.expected_object_width_m:.4f}m "
                f"→ per-finger JTC tolerance={grasp_tolerance_m:.4f}m"
            )
        else:
            grasp_tolerance_m = None  # use tight controller default

        for attempt in range(1, self._GRIPPER_MAX_RETRIES + 1):
            # Log actual finger positions before every attempt so we can see
            # whether the fingers are already at the target or need to move.
            if self._finger_positions:
                positions_str = ", ".join(
                    f"{n}={v:.4f}m" for n, v in sorted(self._finger_positions.items())
                )
                self.get_logger().info(
                    f"Gripper attempt {attempt}/{self._GRIPPER_MAX_RETRIES} — "
                    f"finger positions: [{positions_str}] "
                    f"→ target={finger_position:.4f}m"
                )
            else:
                self.get_logger().warn(
                    f"Gripper attempt {attempt}/{self._GRIPPER_MAX_RETRIES} — "
                    "no finger joint state data yet."
                )

            # Prepend an explicit t=0 waypoint using the actual joint positions
            # so the JTC does not need to sample hardware state itself, which
            # eliminates the service-callback timing race condition.
            start_positions: list[float] | None = None
            j1 = self._finger_positions.get("panda_finger_joint1")
            j2 = self._finger_positions.get("panda_finger_joint2")
            if j1 is not None and j2 is not None:
                start_positions = [j1, j2]

            action_success = self._gripper.move_to_position(
                finger_position,
                GRIPPER_MOVE_DURATION_SEC,
                start_positions=start_positions,
                goal_tolerance_m=grasp_tolerance_m,
            )

            if not action_success:
                response.success = False
                response.message = (
                    f"Gripper action failed on attempt {attempt} "
                    f"for target width={request.width_m:.4f}m."
                )
                self.get_logger().error(response.message)
                return response

            # Verify that every finger actually reached the target.  The JTC
            # reports success based on trajectory time expiry (goal_tolerance is
            # not configured for non-grasp moves), so "success" does not
            # guarantee physical movement.  A Gazebo/gz_ros2_control race
            # condition can cause one joint to silently ignore the commanded
            # trajectory for that execution cycle.
            # For grasp moves the same tolerance is used so fingers stopping
            # against an object surface pass without triggering a false retry.
            if self._fingers_at_target(finger_position, tolerance_m=grasp_tolerance_m):
                response.success = True
                response.message = (
                    f"Gripper reached width={request.width_m:.4f}m "
                    f"on attempt {attempt}."
                )
                self.get_logger().info(response.message)
                return response

            self.get_logger().warn(
                f"Gripper action reported success on attempt {attempt} but "
                f"one or more fingers did not reach target {finger_position:.4f}m. "
                f"Retrying."
            )

        response.success = False
        response.message = (
            f"Gripper failed to reach width={request.width_m:.4f}m after "
            f"{self._GRIPPER_MAX_RETRIES} attempts."
        )
        self.get_logger().error(response.message)
        return response

def main(args=None):
    rclpy.init(args=args)
    node = PandaCoordinatorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    spin_thread = None

    try:
        executor.add_node(node)

        spin_thread = threading.Thread(
            target=executor.spin,
            name="panda_coordinator_executor",
            daemon=True,
        )
        spin_thread.start()

        node.set_status("starting")

        control_servers_ready = node.wait_for_control_servers()
        if not control_servers_ready:
            node.get_logger().error(
                "Coordinator setup failed because one or more control servers are unavailable."
            )
            node.set_status("error:control_servers_unavailable")
            return

        planner_service_ready = node.wait_for_planner_service()
        if not planner_service_ready:
            node.get_logger().error(
                "Coordinator setup failed because the planner service is unavailable."
            )
            node.set_status("error:planner_service_unavailable")
            return

        sequence_succeeded = node.run_startup_sequence()
        if not sequence_succeeded:
            node.get_logger().error("Coordinator startup sequence failed.")
            node.set_status("error:startup_sequence_failed")
            return

        node._is_ready = True
        node.set_status("ready")
        node.get_logger().info(
            "Coordinator is ready. Startup sequence completed successfully."
        )

        while rclpy.ok():
            time.sleep(0.5)
    finally:
        executor.shutdown()

        if spin_thread is not None:
            spin_thread.join(timeout=2.0)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()