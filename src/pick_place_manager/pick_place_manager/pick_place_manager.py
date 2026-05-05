import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from pathlib import Path
import math
import time
import threading

from pick_place_manager.calibration_loader import load_eye_to_hand_calibration
from pick_place_manager.transforms import (
    transform_from_translation_quaternion,
    transform_from_translation_rpy,
    translation_rpy_from_transform,
    task_space_pose_from_translation_rpy,
)

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import String
from pick_place_interfaces.srv import DetectedObjects
from pick_place_interfaces.srv import ExecuteHome
from pick_place_interfaces.srv import ExecuteTaskPose
from pick_place_interfaces.srv import MoveGripper
from pick_place_interfaces.srv import GeneratePlan
from pick_place_interfaces.msg import TaskSpacePose

from pick_place_pddl.world_state_tracker import WorldStateTracker

# Pick sequence tuning constants.
_APPROACH_HEIGHT_M = 0.05    # metres above cube top surface for pre-grasp approach
_LIFT_HEIGHT_M = _APPROACH_HEIGHT_M  # lift returns to same height as approach (symmetric)
_GRIPPER_OPEN_WIDTH_M  = 0.068 # total finger gap when open; 0.034m per finger,
                                # giving 6mm margin below the 0.04m hard joint limit.
                                # 2mm was insufficient — Gazebo PID overshoot was
                                # reaching the limit and jamming the finger.
_GRIPPER_CLOSE_WIDTH_M = 0.0  # fully closed — Gazebo rigid-body physics stops
                               # the fingers at the cube surface naturally
_PICK_SPEED_SCALE = 1.0      # speed scale for all pick/place motions.
                              # compute_approach_joints performs a PILZ LIN
                              # dry-run at this speed and rejects any approach
                              # configuration whose descent would violate a joint
                              # acceleration limit, so the actual descent is
                              # guaranteed to succeed at this speed.
_PLACE_Y_OFFSET_M = 0.050   # place target offset used for simple test sequence only
# Dwell after the gripper action returns before starting the lift.
# The JTC reports success when the trajectory sim-time elapses; Gazebo may
# be running well below real-time (~10 % RT observed) so by the time the
# action returns the grippers have had ample sim-time to reach contact.
# This short wall-clock dwell lets the cube contact forces stabilise.
_POST_GRASP_SETTLE_SEC = 0.5

# Wall-clock dwell after the PILZ LIN descent before closing the gripper.
# Gazebo runs at ~10 % RT, so 2.0 s wall-clock ≈ 0.2 sim-seconds, giving
# the arm dynamics enough time to settle before the gripper PID engages.
_POST_ARM_SETTLE_SEC = 2.0

# How close a cube centroid must be to a slot centre (metres) to be
# considered occupying that slot at the start of a plan.
_SLOT_ASSIGNMENT_RADIUS_M = 0.05

# Maximum full replan attempts before aborting the PDDL task.
_MAX_REPLAN_ATTEMPTS = 3


class PickPlaceManager(Node):
    def __init__(self) -> None:
        super().__init__("pick_place_manager")

        self.declare_parameter("calibration_file", "")
        calibration_file = self.get_parameter("calibration_file").get_parameter_value().string_value
        self._calibration_file = Path(calibration_file)
        self._base_to_camera_calibration = load_eye_to_hand_calibration(self._calibration_file)

        self._base_T_camera = transform_from_translation_quaternion(
            self._base_to_camera_calibration["translation_xyz"],
            self._base_to_camera_calibration["quaternion_xyzw"],
        )

        # Which cube ID to target for the simple test sequence.
        self.declare_parameter("target_cube_id", "green_cube")
        self._target_cube_id = (
            self.get_parameter("target_cube_id").get_parameter_value().string_value
        )
        self.get_logger().info(f"Target cube ID: '{self._target_cube_id}'")

        # ---- PDDL task parameters ----------------------------------------
        # goal_type: "stack" or "arrange"
        # goal_sequence: comma-separated cube IDs in goal order.
        #   stack   → bottom-to-top, e.g. "red_cube,green_cube,blue_cube"
        #   arrange → slot_1 to slot_3, e.g. "blue_cube,green_cube,red_cube"
        self.declare_parameter("goal_type", "stack")
        self.declare_parameter("goal_sequence", "red_cube,green_cube,blue_cube")
        self._goal_type = (
            self.get_parameter("goal_type").get_parameter_value().string_value
        )
        self._goal_sequence = [
            s.strip()
            for s in self.get_parameter("goal_sequence")
                         .get_parameter_value().string_value.split(",")
            if s.strip()
        ]
        self.get_logger().info(
            f"PDDL goal: type='{self._goal_type}' sequence={self._goal_sequence}"
        )

        # ---- Slot position parameters ------------------------------------
        # Each slot has three parameters: <slot>_x, <slot>_y, <slot>_z_surface.
        # z_surface is the height of the table surface at that slot (metres).
        # Slots define only x/y positions in the robot base frame.
        # The surface z is derived at runtime from detected cube positions
        # so there is no hardcoded table height.
        _SLOT_NAMES = ["slot_1", "slot_2", "slot_3"]
        _SLOT_DEFAULTS = {
            "slot_1": (0.50, -0.20),
            "slot_2": (0.50,  0.00),
            "slot_3": (0.50,  0.20),
        }
        self._slot_data: dict[str, tuple[float, float]] = {}
        for slot_name in _SLOT_NAMES:
            dx, dy = _SLOT_DEFAULTS[slot_name]
            self.declare_parameter(f"{slot_name}_x", dx)
            self.declare_parameter(f"{slot_name}_y", dy)
            sx = self.get_parameter(f"{slot_name}_x").get_parameter_value().double_value
            sy = self.get_parameter(f"{slot_name}_y").get_parameter_value().double_value
            self._slot_data[slot_name] = (sx, sy)
            self.get_logger().info(
                f"Slot '{slot_name}': x={sx:.3f}, y={sy:.3f}"
            )

        # ---- Task execution mode -----------------------------------------
        # "pddl"  — run the full PDDL-planned task (default)
        # "test"  — run the simple single-cube pick-and-place test sequence
        self.declare_parameter("execution_mode", "pddl")
        self._execution_mode = (
            self.get_parameter("execution_mode").get_parameter_value().string_value
        )
        self.get_logger().info(f"Execution mode: '{self._execution_mode}'")

        # ---- Gripper state cache ----------------------------------------
        # Tracks whether the gripper is currently open so that redundant
        # open commands (e.g. pick immediately following a place that already
        # opened the gripper) are skipped.  Driving the gripper to its open
        # limit when it is already open can force the finger into the hard
        # joint-position limit and prevent a clean close on the next grasp.
        # Assumed closed at startup; updated after every open/close action.
        self._gripper_is_open: bool = False

        # ---- ROS infrastructure ------------------------------------------
        self._coordinator_status = "unknown"

        self._coordinator_status_subscription = self.create_subscription(
            String,
            "/panda_coordinator/status",
            self.coordinator_status_callback,
            10,
        )

        self._detect_objects_client = self.create_client(
            DetectedObjects,
            "detect_objects",
        )

        self._execute_task_pose_client = self.create_client(
            ExecuteTaskPose,
            "execute_task_pose",
        )

        self._move_gripper_client = self.create_client(
            MoveGripper,
            "move_gripper",
        )

        self._generate_plan_client = self.create_client(
            GeneratePlan,
            "/generate_plan",
        )

        self._execute_home_client = self.create_client(
            ExecuteHome,
            "execute_home_position",
        )

        # ---- Planning scene publisher ------------------------------------
        # Use transient_local (latched) durability so the MoveIt planning
        # scene monitor receives messages even if it subscribes after us.
        _ps_qos = QoSProfile(
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._planning_scene_pub = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            _ps_qos,
        )

    # ------------------------------------------------------------------
    # Status / ready helpers
    # ------------------------------------------------------------------

    def coordinator_status_callback(self, msg: String) -> None:
        """
        Cache the latest coordinator status message.

        Inputs:
            msg: Coordinator status message.

        Returns:
            None
        """
        self._coordinator_status = msg.data

    def _wait_for_coordinator_ready(self) -> bool:
        self.get_logger().info("Waiting for coordinator to report ready status...")

        while rclpy.ok():
            if self._coordinator_status == "ready":
                self.get_logger().info("Coordinator is ready.")
                return True

            self.get_logger().info(
                f"Coordinator not ready yet. Current status: '{self._coordinator_status}'"
            )
            time.sleep(1.0)

        return False

    def _wait_for_detect_objects_service(self) -> bool:
        self.get_logger().info("Waiting for detect_objects service...")

        while rclpy.ok():
            if self._detect_objects_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("detect_objects service is available.")
                return True

            self.get_logger().info("detect_objects service not available yet, waiting...")

        return False

    def _wait_for_execute_task_pose_service(self) -> bool:
        self.get_logger().info("Waiting for execute_task_pose service...")

        while rclpy.ok():
            if self._execute_task_pose_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("execute_task_pose service is available.")
                return True

            self.get_logger().info("execute_task_pose service not available yet, waiting...")

        return False

    def _wait_for_move_gripper_service(self) -> bool:
        self.get_logger().info("Waiting for move_gripper service...")

        while rclpy.ok():
            if self._move_gripper_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("move_gripper service is available.")
                return True

            self.get_logger().info("move_gripper service not available yet, waiting...")

        return False

    def _wait_for_generate_plan_service(self) -> bool:
        self.get_logger().info("Waiting for /generate_plan service...")

        while rclpy.ok():
            if self._generate_plan_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("/generate_plan service is available.")
                return True

            self.get_logger().info("/generate_plan service not available yet, waiting...")

        return False

    def _wait_for_system_ready(self) -> bool:
        if not self._wait_for_coordinator_ready():
            self.get_logger().error("Coordinator did not become ready. Exiting.")
            return False

        if not self._wait_for_detect_objects_service():
            self.get_logger().error("detect_objects service was not available. Exiting.")
            return False

        if not self._wait_for_execute_task_pose_service():
            self.get_logger().error("execute_task_pose service was not available. Exiting.")
            return False

        if not self._wait_for_move_gripper_service():
            self.get_logger().error("move_gripper service was not available. Exiting.")
            return False

        if self._execution_mode == "pddl":
            if not self._wait_for_generate_plan_service():
                self.get_logger().error("/generate_plan service was not available. Exiting.")
                return False

        self.get_logger().info("All pick-place manager prerequisites are satisfied.")
        return True

    # ------------------------------------------------------------------
    # Synchronous service call helpers
    # ------------------------------------------------------------------

    def _call_execute_task_pose_sync(
        self,
        pose: TaskSpacePose,
        speed_scale: float,
        use_orientation_constraint: bool = False,
        orientation_tolerance_rad: float = 0.1,
        approach_height_m: float = 0.0,
        approach_only: bool = False,
        timeout_sec: float = 120.0,
    ) -> bool:
        """
        Synchronously call execute_task_pose and return whether it succeeded.

        Inputs:
            pose: Target task-space pose.
            speed_scale: Motion speed scale factor.
            use_orientation_constraint: If True, the coordinator executes a
                PILZ LIN motion to keep the path Cartesian-linear.
            orientation_tolerance_rad: Allowed orientation deviation in
                radians when constraint is active. Default 0.1 (~6 degrees).
            approach_height_m: When > 0 and use_orientation_constraint is True,
                the coordinator derives a compatible approach joint configuration
                by working backwards from the target pose before descending with
                PILZ LIN. Pass 0.0 for lift motions (no approach needed).
            timeout_sec: Maximum time to wait for a response.

        Returns:
            bool: True if the motion succeeded, otherwise False.
        """
        request = ExecuteTaskPose.Request()
        request.pose = pose
        request.speed_scale = speed_scale
        request.use_orientation_constraint = use_orientation_constraint
        request.orientation_tolerance_rad = orientation_tolerance_rad
        request.approach_height_m = approach_height_m
        request.approach_only = approach_only

        future = self._execute_task_pose_client.call_async(request)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("execute_task_pose call timed out.")
                return False
            time.sleep(0.01)

        if future.exception() is not None:
            self.get_logger().error(
                f"execute_task_pose call raised an exception: {future.exception()}"
            )
            return False

        response = future.result()
        self.get_logger().info(
            f"execute_task_pose response: success={response.success}, "
            f"message='{response.message}'"
        )
        return response.success

    def _call_move_gripper_sync(
        self,
        width_m: float,
        expected_object_width_m: float = 0.0,
        timeout_sec: float = 120.0,
    ) -> bool:
        """
        Synchronously call move_gripper and return whether it succeeded.

        Inputs:
            width_m: Target total gap between fingers in metres.
            expected_object_width_m: Width of the object being grasped in
                metres.  Pass 0.0 (default) for open/empty-close/diagnostic
                moves — a tight 5 mm tolerance is used.  Pass the detected
                object width for actual grasp closes so the fingers stopping
                at the object surface is accepted as success.
            timeout_sec: Maximum wall-clock time to wait for a response.
                Set high (120 s default) because Gazebo can run well below
                real-time (~10 % RT) after heavy arm motions, making a 1
                sim-second gripper trajectory take ~10 wall-seconds.

        Returns:
            bool: True if the gripper motion succeeded, otherwise False.
        """
        request = MoveGripper.Request()
        request.width_m = width_m
        request.expected_object_width_m = expected_object_width_m

        future = self._move_gripper_client.call_async(request)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("move_gripper call timed out.")
                return False
            time.sleep(0.01)

        if future.exception() is not None:
            self.get_logger().error(
                f"move_gripper call raised an exception: {future.exception()}"
            )
            return False

        response = future.result()
        self.get_logger().info(
            f"move_gripper response: success={response.success}, "
            f"message='{response.message}'"
        )
        return response.success

    def _call_execute_home_sync(self, timeout_sec: float = 120.0) -> bool:
        """
        Synchronously call execute_home_position and return whether it succeeded.

        Moves the arm to the home joint configuration so it is clear of the
        camera field of view before object detection.

        Inputs:
            timeout_sec: Maximum wall-clock time to wait for a response.

        Returns:
            bool: True if homing succeeded, otherwise False.
        """
        if not self._execute_home_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("execute_home_position service not available.")
            return False

        request = ExecuteHome.Request()
        future = self._execute_home_client.call_async(request)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("execute_home_position call timed out.")
                return False
            time.sleep(0.01)

        if future.exception() is not None:
            self.get_logger().error(
                f"execute_home_position call raised an exception: {future.exception()}"
            )
            return False

        response = future.result()
        self.get_logger().info(
            f"execute_home_position response: success={response.success}, "
            f"message='{response.message}'"
        )
        # Note: home only moves the arm — it does NOT open or close the gripper.
        # Do not update _gripper_is_open here; the gripper stays in whatever
        # state it was in before the home call.
        return response.success

    # ------------------------------------------------------------------
    # MoveIt planning scene helpers
    # ------------------------------------------------------------------

    def _scene_add_cube(
        self,
        cube_id: str,
        x: float,
        y: float,
        z_centroid: float,
        half_height: float,
        half_width: float | None = None,
    ) -> None:
        """
        Add or update a cube collision object in the MoveIt planning scene.

        Call this after successfully placing a cube so that subsequent
        OMPL plans (including home trajectories) avoid it.

        Inputs:
            cube_id:    Unique object name in the planning scene.
            x, y:       Centroid x/y position in the world frame (metres).
            z_centroid: Centroid z position in the world frame (metres).
            half_height: Half the cube height (metres).
            half_width:  Half the cube width (metres); defaults to half_height
                         for uniform cubes.
        """
        hw = half_width if half_width is not None else half_height
        side = hw * 2.0
        height = half_height * 2.0

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [side, side, height]

        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z_centroid
        pose.orientation.w = 1.0

        obj = CollisionObject()
        obj.id = cube_id
        obj.header.frame_id = "world"
        obj.operation = CollisionObject.ADD
        obj.primitives.append(primitive)
        obj.primitive_poses.append(pose)

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(obj)
        self._planning_scene_pub.publish(ps)

        self.get_logger().info(
            f"Planning scene: added '{cube_id}' at "
            f"({x:.3f}, {y:.3f}, {z_centroid:.3f}), size={side:.3f}×{height:.3f}m."
        )

    def _scene_remove_cube(self, cube_id: str) -> None:
        """
        Remove a cube collision object from the MoveIt planning scene.

        Call this before picking a cube so that the approach trajectory
        is not blocked by the cube's own collision geometry.

        Inputs:
            cube_id: Unique object name to remove.
        """
        obj = CollisionObject()
        obj.id = cube_id
        obj.header.frame_id = "world"
        obj.operation = CollisionObject.REMOVE

        ps = PlanningScene()
        ps.is_diff = True
        ps.world.collision_objects.append(obj)
        self._planning_scene_pub.publish(ps)

        self.get_logger().info(
            f"Planning scene: removed '{cube_id}'."
        )

    def _call_detect_all_objects_sync(
        self,
        timeout_sec: float = 10.0,
    ) -> list:
        """
        Synchronously call detect_objects and return all detected objects.

        Inputs:
            timeout_sec: Maximum time to wait for a response.

        Returns:
            list[DetectedObject]: All detections, or an empty list on failure.
        """
        request = DetectedObjects.Request()
        future = self._detect_objects_client.call_async(request)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("detect_objects call timed out.")
                return []
            time.sleep(0.01)

        if future.exception() is not None:
            self.get_logger().error(
                f"detect_objects call raised an exception: {future.exception()}"
            )
            return []

        response = future.result()

        if not response.success:
            self.get_logger().error(
                f"detect_objects returned failure: '{response.message}'"
            )
            return []

        for det in response.detections:
            self.get_logger().info(
                f"  Detection: id='{det.id}', "
                f"x={det.pose_camera.x:.3f}, "
                f"y={det.pose_camera.y:.3f}, "
                f"z={det.pose_camera.z:.3f}, "
                f"half_height={det.estimated_half_height_m:.4f}m, "
                f"half_width={det.estimated_half_width_m:.4f}m"
            )

        return list(response.detections)

    def _call_detect_objects_sync(
        self,
        timeout_sec: float = 10.0,
    ):
        """
        Synchronously call detect_objects and return the DetectedObject
        matching the target cube ID, or None if not found.

        Logs all received detections before selecting so the full observed
        scene is always visible in the node output.

        Inputs:
            timeout_sec: Maximum time to wait for a response.

        Returns:
            DetectedObject | None: The selected detection, or None if the
            target was not found or the call failed.
        """
        detections = self._call_detect_all_objects_sync(timeout_sec)

        if not detections:
            self.get_logger().warn("detect_objects returned no detections.")
            return None

        for det in detections:
            if det.id == self._target_cube_id:
                self.get_logger().info(f"Selected target: '{det.id}'")
                return det

        self.get_logger().warn(
            f"Target cube '{self._target_cube_id}' was not found in detections. "
            f"Available IDs: {[d.id for d in detections]}"
        )
        return None

    def _call_generate_plan_sync(
        self,
        detections: list,
        occupied_slots: list[str],
        timeout_sec: float = 30.0,
    ) -> list | None:
        """
        Synchronously call /generate_plan and return the PddlAction list.

        Inputs:
            detections: List of DetectedObject messages for all visible cubes.
            occupied_slots: Parallel list of slot names for each detection
                (empty string if the cube is not in a named slot).
            timeout_sec: Maximum time to wait for a response.

        Returns:
            list[PddlAction] | None: The plan, or None on failure.
        """
        request = GeneratePlan.Request()
        request.detected_objects = detections
        request.occupied_slots   = occupied_slots
        request.goal_type        = self._goal_type
        request.goal_sequence    = self._goal_sequence

        future = self._generate_plan_client.call_async(request)

        start = time.time()
        while not future.done():
            if time.time() - start > timeout_sec:
                self.get_logger().error("/generate_plan call timed out.")
                return None
            time.sleep(0.01)

        if future.exception() is not None:
            self.get_logger().error(
                f"/generate_plan call raised an exception: {future.exception()}"
            )
            return None

        response = future.result()

        if not response.success:
            self.get_logger().error(
                f"/generate_plan returned failure: '{response.message}'"
            )
            return None

        self.get_logger().info(
            f"/generate_plan succeeded: {response.message}"
        )
        return list(response.plan)

    # ------------------------------------------------------------------
    # Pose computation helpers
    # ------------------------------------------------------------------

    def _compute_base_frame_pose_from_camera_pose(
        self,
        pose_camera: TaskSpacePose,
    ) -> tuple[list[float], list[float]]:
        camera_T_object = transform_from_translation_rpy(
            [pose_camera.x, pose_camera.y, pose_camera.z],
            [pose_camera.roll, pose_camera.pitch, pose_camera.yaw],
        )

        base_T_object = self._base_T_camera @ camera_T_object

        return translation_rpy_from_transform(base_T_object)

    def _build_approach_task_pose(
        self,
        object_translation: list[float],
        object_rpy: list[float],
        z_offset: float = 0.10,
    ) -> TaskSpacePose:
        approach_translation = object_translation.copy()
        approach_translation[2] += z_offset

        return task_space_pose_from_translation_rpy(
            approach_translation,
            object_rpy,
        )

    # ------------------------------------------------------------------
    # Slot assignment: match detected cubes to known slot positions
    # ------------------------------------------------------------------

    def _assign_slots_to_detections(
        self,
        detections: list,
    ) -> list[str]:
        """
        For each detection, find the nearest slot within
        _SLOT_ASSIGNMENT_RADIUS_M.  Returns a parallel list of slot names
        (empty string if no slot is close enough).

        Inputs:
            detections: List of DetectedObject messages.

        Returns:
            list[str]: Slot name for each detection (same order), or "" if
            the cube is not in any slot.
        """
        result = []
        for det in detections:
            base_t, _ = self._compute_base_frame_pose_from_camera_pose(
                det.pose_camera
            )
            cube_x, cube_y = base_t[0], base_t[1]

            best_slot = ""
            best_dist = _SLOT_ASSIGNMENT_RADIUS_M

            for slot_name, (sx, sy) in self._slot_data.items():
                dist = math.hypot(cube_x - sx, cube_y - sy)
                if dist < best_dist:
                    best_dist = dist
                    best_slot = slot_name

            result.append(best_slot)
            if best_slot:
                self.get_logger().info(
                    f"  Assigned '{det.id}' → '{best_slot}' "
                    f"(dist={best_dist:.3f}m)"
                )
            else:
                self.get_logger().info(
                    f"  '{det.id}' not in any named slot."
                )

        return result

    def _build_cube_data_from_detections(
        self,
        detections: list,
    ) -> dict[str, tuple[float, float, float, float, float]]:
        """
        Convert DetectedObject list to the cube_data dict expected by
        WorldStateTracker: {cube_id: (x, y, z_centroid, half_height, half_width)}.
        """
        cube_data: dict[str, tuple] = {}
        for det in detections:
            base_t, _ = self._compute_base_frame_pose_from_camera_pose(
                det.pose_camera
            )
            # pose_camera gives the top-surface position; centroid is half-height below.
            centroid_x = base_t[0]
            centroid_y = base_t[1]
            centroid_z = base_t[2] - det.estimated_half_height_m
            cube_data[det.id] = (
                centroid_x,
                centroid_y,
                centroid_z,
                det.estimated_half_height_m,
                det.estimated_half_width_m,
            )
        return cube_data

    def _compute_surface_z_from_cube_data(
        self,
        cube_data: dict[str, tuple[float, float, float, float, float]],
    ) -> float:
        """
        Estimate the conveyor surface height from detected cube positions.

        Each detected cube centroid is half_height above the surface, so
        surface_z = centroid_z - half_height for every cube.  The mean
        across all cubes is returned so individual detection noise averages
        out.  All cubes are expected to be resting on the same flat surface
        at the time of detection.

        Inputs:
            cube_data: {cube_id: (x, y, z_centroid, half_height, half_width)}
                       as returned by _build_cube_data_from_detections.

        Returns:
            float: Estimated surface z in the robot base frame (metres).
        """
        surface_zs = [z - hh for (_, _, z, hh, _) in cube_data.values()]
        surface_z = sum(surface_zs) / len(surface_zs)
        self.get_logger().info(
            f"Detected surface z = {surface_z:.4f} m "
            f"(mean of {len(surface_zs)} cube base measurements)"
        )
        return surface_z

    # ------------------------------------------------------------------
    # Physical pick / place execution  (pose-based, used by PDDL loop)
    # ------------------------------------------------------------------

    def _execute_pick_at_pose(
        self,
        cube_id: str,
        x: float,
        y: float,
        z_centroid: float,
        half_height: float,
        half_width: float,
    ) -> bool:
        """
        Execute a full pick sequence at the given base-frame centroid pose.

        Sequence: open gripper → OMPL to approach → PILZ LIN descent →
                  settle → close gripper → PILZ LIN lift.

        Inputs:
            cube_id:     Cube identifier (for log messages only).
            x, y:        Horizontal centroid position in base frame (metres).
            z_centroid:  Vertical centroid position in base frame (metres).
            half_height: Cube half-height in metres (used to compute top-surface).
            half_width:  Cube half-width in metres (used for grasp tolerance).

        Returns:
            bool: True if the full sequence succeeded, otherwise False.
        """
        self.get_logger().info(f"Pick at pose: id='{cube_id}' "
                               f"centroid=({x:.3f}, {y:.3f}, {z_centroid:.3f})")

        grasp_z  = z_centroid
        top_z    = z_centroid + half_height
        lift_z   = top_z + _LIFT_HEIGHT_M

        grasp_orientation_rpy = [math.pi, 0.0, 0.0]

        grasp_pose = task_space_pose_from_translation_rpy(
            [x, y, grasp_z], grasp_orientation_rpy,
        )
        lift_pose = task_space_pose_from_translation_rpy(
            [x, y, lift_z], grasp_orientation_rpy,
        )

        if self._gripper_is_open:
            self.get_logger().info(
                "Pick step 1/5: Gripper already open — skipping redundant open command."
            )
        else:
            self.get_logger().info("Pick step 1/5: Opening gripper.")
            if not self._call_move_gripper_sync(_GRIPPER_OPEN_WIDTH_M):
                self.get_logger().error("Pick failed: could not open gripper.")
                return False
            self._gripper_is_open = True

        # Remove the cube from the MoveIt planning scene before approaching so
        # the gripper is not blocked by the cube's own collision geometry.
        self._scene_remove_cube(cube_id)

        self.get_logger().info("Pick step 2/5: Moving to approach pose via OMPL.")
        if not self._call_execute_task_pose_sync(
            grasp_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=_APPROACH_HEIGHT_M,
            approach_only=True,
        ):
            self.get_logger().error("Pick failed: could not reach approach pose.")
            return False

        self.get_logger().info("Pick step 3/5: Descending to grasp pose via PILZ LIN.")
        if not self._call_execute_task_pose_sync(
            grasp_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=0.0,
        ):
            self.get_logger().error("Pick failed: could not reach grasp pose.")
            return False

        time.sleep(_POST_ARM_SETTLE_SEC)

        self.get_logger().info("Pick step 4/5: Closing gripper on object.")
        object_width_m = half_width * 2.0
        if not self._call_move_gripper_sync(
            _GRIPPER_CLOSE_WIDTH_M,
            expected_object_width_m=object_width_m,
        ):
            self.get_logger().error("Pick failed: could not close gripper.")
            return False
        self._gripper_is_open = False

        time.sleep(_POST_GRASP_SETTLE_SEC)

        self.get_logger().info("Pick step 5/5: Lifting via PILZ LIN.")
        if not self._call_execute_task_pose_sync(
            lift_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=0.0,
        ):
            self.get_logger().error("Pick failed: could not reach lift pose.")
            return False

        self.get_logger().info(f"Pick of '{cube_id}' completed successfully.")
        return True

    def _execute_place_at_pose(
        self,
        cube_id: str,
        x: float,
        y: float,
        z_centroid: float,
        half_height: float,
    ) -> bool:
        """
        Execute a full place sequence at the given base-frame centroid pose.

        Sequence: OMPL to approach → PILZ LIN descent → settle →
                  open gripper → PILZ LIN lift clear.

        Inputs:
            cube_id:    Cube identifier (for log messages only).
            x, y:       Horizontal centroid position in base frame (metres).
            z_centroid: Vertical centroid position to place at (metres).
            half_height: Cube half-height (used to compute lift-clear height).

        Returns:
            bool: True if the full sequence succeeded, otherwise False.
        """
        self.get_logger().info(f"Place at pose: id='{cube_id}' "
                               f"centroid=({x:.3f}, {y:.3f}, {z_centroid:.3f})")

        top_z  = z_centroid + half_height
        lift_z = top_z + _LIFT_HEIGHT_M

        place_orientation_rpy = [math.pi, 0.0, 0.0]

        place_pose = task_space_pose_from_translation_rpy(
            [x, y, z_centroid], place_orientation_rpy,
        )
        lift_pose = task_space_pose_from_translation_rpy(
            [x, y, lift_z], place_orientation_rpy,
        )

        self.get_logger().info("Place step 1/4: Moving to approach pose via OMPL.")
        if not self._call_execute_task_pose_sync(
            place_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=_APPROACH_HEIGHT_M,
            approach_only=True,
        ):
            self.get_logger().error("Place failed: could not reach approach pose.")
            return False

        self.get_logger().info("Place step 2/4: Descending to place pose via PILZ LIN.")
        if not self._call_execute_task_pose_sync(
            place_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=0.0,
        ):
            self.get_logger().error("Place failed: could not reach place pose.")
            return False

        time.sleep(_POST_ARM_SETTLE_SEC)

        self.get_logger().info("Place step 3/4: Opening gripper to release object.")
        if not self._call_move_gripper_sync(_GRIPPER_OPEN_WIDTH_M):
            self.get_logger().error("Place failed: could not open gripper.")
            return False
        self._gripper_is_open = True

        # Add the placed cube back to the MoveIt planning scene at its new
        # position.  This ensures that subsequent OMPL plans (including the
        # home trajectory) route around it rather than colliding with it.
        self._scene_add_cube(
            cube_id=cube_id,
            x=x,
            y=y,
            z_centroid=z_centroid,
            half_height=half_height,
        )

        self.get_logger().info("Place step 4/4: Lifting clear via PILZ LIN.")
        if not self._call_execute_task_pose_sync(
            lift_pose, _PICK_SPEED_SCALE,
            use_orientation_constraint=True,
            approach_height_m=0.0,
        ):
            self.get_logger().error("Place failed: could not lift clear after release.")
            return False

        self.get_logger().info(f"Place of '{cube_id}' completed successfully.")
        return True

    # ------------------------------------------------------------------
    # Simple test-sequence helpers (single-cube pick + offset place)
    # ------------------------------------------------------------------

    def _execute_pick(self, detection) -> bool:
        """
        Execute the full pick sequence for a detected cube.

        Wrapper around _execute_pick_at_pose that converts a DetectedObject
        (camera frame) to a base-frame centroid pose.

        Inputs:
            detection: DetectedObject for the target cube.

        Returns:
            bool: True if the full sequence succeeded, otherwise False.
        """
        self.get_logger().info(f"Starting pick sequence for '{detection.id}'.")

        base_translation, _ = self._compute_base_frame_pose_from_camera_pose(
            detection.pose_camera
        )

        self.get_logger().info(
            f"Base frame top surface: "
            f"x={base_translation[0]:.3f}, "
            f"y={base_translation[1]:.3f}, "
            f"z={base_translation[2]:.3f}"
        )

        # pose_camera gives the top surface; centroid is half-height below.
        centroid_z = base_translation[2] - detection.estimated_half_height_m

        return self._execute_pick_at_pose(
            cube_id=detection.id,
            x=base_translation[0],
            y=base_translation[1],
            z_centroid=centroid_z,
            half_height=detection.estimated_half_height_m,
            half_width=detection.estimated_half_width_m,
        )

    def _execute_place(self, detection) -> bool:
        """
        Execute the place sequence for a previously picked cube.

        The place target is _PLACE_Y_OFFSET_M to the left (+Y) of the
        original pick position.

        Inputs:
            detection: DetectedObject used for the preceding pick.

        Returns:
            bool: True if the full sequence succeeded, otherwise False.
        """
        self.get_logger().info(f"Starting place sequence for '{detection.id}'.")

        base_translation, _ = self._compute_base_frame_pose_from_camera_pose(
            detection.pose_camera
        )

        place_x = base_translation[0]
        place_y = base_translation[1] + _PLACE_Y_OFFSET_M
        # Place centroid at same height as pick centroid.
        place_z = base_translation[2] - detection.estimated_half_height_m

        self.get_logger().info(
            f"Place target: ({place_x:.3f}, {place_y:.3f}, {place_z:.3f})"
        )

        return self._execute_place_at_pose(
            cube_id=detection.id,
            x=place_x,
            y=place_y,
            z_centroid=place_z,
            half_height=detection.estimated_half_height_m,
        )

    # ------------------------------------------------------------------
    # PDDL action executor
    # ------------------------------------------------------------------

    def _execute_pddl_action(
        self,
        action,
        tracker: WorldStateTracker,
    ) -> bool:
        """
        Execute a single PddlAction using the WorldStateTracker for pose
        computation.

        Inputs:
            action:  PddlAction message (action_type, cube_id, location).
            tracker: WorldStateTracker holding current physical cube states.

        Returns:
            bool: True if the physical action succeeded, otherwise False.
        """
        action_type = action.action_type
        cube_id     = action.cube_id
        location    = action.location

        self.get_logger().info(
            f"Executing PDDL action: {action_type}({cube_id}, {location})"
        )

        cube_state = tracker.cube_state(cube_id)

        if action_type == "pick_from_surface":
            x, y, z, hw = tracker.pick_pose(cube_id)
            return self._execute_pick_at_pose(
                cube_id=cube_id,
                x=x, y=y, z_centroid=z,
                half_height=cube_state.half_height,
                half_width=cube_state.half_width,
            )

        elif action_type == "pick_from_stack":
            x, y, z, hw = tracker.pick_pose(cube_id)
            return self._execute_pick_at_pose(
                cube_id=cube_id,
                x=x, y=y, z_centroid=z,
                half_height=cube_state.half_height,
                half_width=cube_state.half_width,
            )

        elif action_type == "place_on_surface":
            x, y, z = tracker.place_pose_on_surface(cube_id, location)
            return self._execute_place_at_pose(
                cube_id=cube_id,
                x=x, y=y, z_centroid=z,
                half_height=cube_state.half_height,
            )

        elif action_type == "place_on_cube":
            x, y, z = tracker.place_pose_on_cube(cube_id, location)
            return self._execute_place_at_pose(
                cube_id=cube_id,
                x=x, y=y, z_centroid=z,
                half_height=cube_state.half_height,
            )

        else:
            self.get_logger().error(
                f"Unknown PDDL action type: '{action_type}'"
            )
            return False

    # ------------------------------------------------------------------
    # Sequence runners
    # ------------------------------------------------------------------

    def _run_pddl_sequence(self) -> None:
        """
        Execute the PDDL task planning + execution loop.

        Flow:
          1. Detect all cubes.
          2. Assign cubes to slots.
          3. Build WorldStateTracker.
          4. Call /generate_plan.
          5. Execute each action step.
          6. On step failure, replan from the current state (up to
             _MAX_REPLAN_ATTEMPTS times).
          7. Abort and report if no plan can be found.
        """
        self.get_logger().info("PDDL task sequence started.")

        for attempt in range(1, _MAX_REPLAN_ATTEMPTS + 1):
            self.get_logger().info(
                f"Planning attempt {attempt}/{_MAX_REPLAN_ATTEMPTS}..."
            )

            # ---- Home robot before detection (clears camera FOV) ----
            # On a replan the arm may be left at an intermediate pose after a
            # failed step, which blocks the overhead camera.  Always home first
            # so the depth sensor has a clear view of all cubes.
            self.get_logger().info(
                "Homing robot before object detection to clear camera field of view..."
            )
            if not self._call_execute_home_sync():
                self.get_logger().error(
                    "PDDL sequence aborted: failed to home robot before detection."
                )
                return

            # ---- Detect ----
            detections = self._call_detect_all_objects_sync()
            if not detections:
                self.get_logger().error(
                    "PDDL sequence aborted: no cubes detected."
                )
                return

            # ---- Slot assignment ----
            occupied_slots = self._assign_slots_to_detections(detections)

            # ---- Build tracker ----
            cube_data = self._build_cube_data_from_detections(detections)

            # Seed the MoveIt planning scene with the current cube positions.
            # Removes any stale entries first (e.g. cubes that moved between
            # a failed attempt and this replanning pass) then adds fresh ones.
            for cid, (cx, cy, cz, hh, hw) in cube_data.items():
                self._scene_remove_cube(cid)
            time.sleep(0.1)  # brief pause so removes flush before adds
            for cid, (cx, cy, cz, hh, hw) in cube_data.items():
                self._scene_add_cube(cid, cx, cy, cz, hh, hw)

            # Derive the conveyor surface height from detected cube positions
            # so no hardcoded table height is needed anywhere.
            surface_z = self._compute_surface_z_from_cube_data(cube_data)
            slot_data_with_z = {
                name: (x, y, surface_z)
                for name, (x, y) in self._slot_data.items()
            }

            occupied_pairs = [
                (slot_name, det.id)
                for det, slot_name in zip(detections, occupied_slots)
                if slot_name
            ]

            tracker = WorldStateTracker(
                cube_data=cube_data,
                slot_data=slot_data_with_z,
                occupied_slots=occupied_pairs,
            )

            # ---- Plan ----
            plan = self._call_generate_plan_sync(detections, occupied_slots)
            if plan is None:
                self.get_logger().error(
                    f"Planning attempt {attempt} found no solution. "
                    + ("Retrying..." if attempt < _MAX_REPLAN_ATTEMPTS else "Aborting.")
                )
                if attempt < _MAX_REPLAN_ATTEMPTS:
                    time.sleep(1.0)
                    continue
                return

            self.get_logger().info(
                f"Plan has {len(plan)} steps: "
                + ", ".join(
                    f"{a.action_type}({a.cube_id},{a.location})" for a in plan
                )
            )

            # ---- Execute ----
            action_failed = False
            for step_idx, action in enumerate(plan):
                self.get_logger().info(
                    f"Executing step {step_idx + 1}/{len(plan)}: "
                    f"{action.action_type}({action.cube_id}, {action.location})"
                )

                if not self._execute_pddl_action(action, tracker):
                    self.get_logger().error(
                        f"Step {step_idx + 1} failed — triggering replan."
                    )
                    action_failed = True
                    break

                # Update symbolic + physical state for subsequent steps.
                tracker.apply_action(
                    action.action_type, action.cube_id, action.location
                )

            if not action_failed:
                self.get_logger().info(
                    "PDDL task sequence completed successfully."
                )
                # Move the arm to home so it is clear of the finished stack.
                # This prevents the arm from hanging at the final lift position
                # above the placed cubes, and avoids any OMPL paths through the
                # stack if the node is restarted or a new sequence is triggered.
                self.get_logger().info(
                    "Homing robot to safe position after successful task completion."
                )
                self._call_execute_home_sync()
                return

            # Short pause before re-detecting.
            time.sleep(1.0)

        self.get_logger().error(
            f"PDDL sequence aborted after {_MAX_REPLAN_ATTEMPTS} failed planning attempts."
        )

    def _run_startup_test_sequence(self) -> None:
        """
        Launch the configured sequence in a dedicated thread.

        Running in a separate thread allows the sequence to make
        synchronous service calls via polling while the executor
        continues spinning in the background to deliver responses.
        """
        def sequence():
            if self._execution_mode == "pddl":
                self._run_pddl_sequence()
                return

            # ---- Simple single-cube test sequence ----
            self.get_logger().info("Simple pick-and-place test sequence started.")

            detection = self._call_detect_objects_sync()

            if detection is None:
                self.get_logger().error(
                    "Sequence aborted: target cube was not detected."
                )
                return

            if not self._execute_pick(detection):
                self.get_logger().error("Sequence aborted: pick failed.")
                return

            if not self._execute_place(detection):
                self.get_logger().error("Sequence finished: place failed.")
                return

            self.get_logger().info("Pick-and-place test sequence completed successfully.")

        threading.Thread(target=sequence, daemon=True).start()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = PickPlaceManager()
    executor = MultiThreadedExecutor(num_threads=4)
    spin_thread = None

    try:
        executor.add_node(node)

        # Start the executor spinning in the background before waiting
        # for system ready, so subscription callbacks (e.g. coordinator
        # status) can fire while the main thread is in the wait loop.
        spin_thread = threading.Thread(
            target=executor.spin,
            name="pick_place_manager_executor",
            daemon=True,
        )
        spin_thread.start()

        if not node._wait_for_system_ready():
            node.get_logger().error(
                "Pick place manager prerequisites were not satisfied."
            )
            return

        node._run_startup_test_sequence()

        # Keep the main thread alive while the sequence thread
        # and executor run in the background.
        while rclpy.ok():
            time.sleep(0.5)

    finally:
        executor.shutdown()

        if spin_thread is not None:
            spin_thread.join(timeout=2.0)

        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
