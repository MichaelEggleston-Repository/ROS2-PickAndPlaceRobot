import rclpy
from rclpy.node import Node
from dataclasses import dataclass

from pick_place_robot.panda_arm_control import PandaArmControl
from pick_place_robot.panda_gripper_control import PandaGripperControl
from pick_place_robot.panda_moveit_planner import PandaMoveItPlanner
from pick_place_robot.panda_scene_planning import PandaPlanningScene
from pick_place_robot.task_space_pose import TaskSpacePose

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
        self._arm = PandaArmControl(self)
        self._gripper = PandaGripperControl(self)
        self._scene = PandaPlanningScene(self)
        self._planner = PandaMoveItPlanner(self)
        
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
        return arm_ready and gripper_ready
    
    def run_startup_sequence(self) -> bool:
        """
        Run a simple first coordinated behavior:
        move the arm home, open the gripper, then close the gripper.

        Inputs:
            None

        Returns:
            bool: True if the full sequence succeeded, otherwise False.
        """
        self.get_logger().info("Starting coordinator motion sequence...")

        if not self._arm.move_home():
            self.get_logger().error("Coordinator failed during arm home motion.")
            return False

        if not self._gripper.open_gripper():
            self.get_logger().error("Coordinator failed during gripper open motion.")
            return False

        self.get_logger().info("Coordinator motion sequence completed successfully.")
        return True
    
    def pose_to_orientation_xyzw(
        self,
        pose: TaskSpacePose,
    ) -> tuple[float, float, float, float]:
        """
        Convert a task-space pose orientation into a quaternion tuple.

        Inputs:
            pose: The task-space pose containing roll, pitch, and yaw.

        Returns:
            tuple[float, float, float, float]:
                Quaternion as (x, y, z, w).
        """
        return self._planner.rpy_to_quaternion(
            pose.roll,
            pose.pitch,
            pose.yaw,
        )
    
    def create_grasp_pose(self, object_pose: TaskSpacePose) -> TaskSpacePose:
        """
        Create the grasp pose at the object.

        Inputs:
            object_pose: The detected object pose in task space.

        Returns:
            TaskSpacePose: The grasp pose at object height.
        """
        return TaskSpacePose(
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
    ) -> TaskSpacePose:
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
            TaskSpacePose: The place pose for dropping the object.
        """
        return TaskSpacePose(
            x=x,
            y=y,
            z=z,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
        )
    
    def offset_pose_z(self, base_pose: TaskSpacePose, z_offset: float) -> TaskSpacePose:
        """
        Create a new task-space pose by offsetting an existing pose in z.

        Inputs:
            base_pose: The starting pose.
            z_offset: The z offset to apply in meters.

        Returns:
            TaskSpacePose: A new pose with the adjusted z value.
        """
        return TaskSpacePose(
            x=base_pose.x,
            y=base_pose.y,
            z=base_pose.z + z_offset,
            roll=base_pose.roll,
            pitch=base_pose.pitch,
            yaw=base_pose.yaw,
        )
    
    def create_pre_grasp_pose(
        self,
        grasp_pose: TaskSpacePose,
        z_offset: float = 0.10,
    ) -> TaskSpacePose:
        """
        Create a safe approach pose above the grasp pose.

        Inputs:
            grasp_pose: The task-space pose where the gripper should perform the grasp.
            z_offset: Vertical offset above the grasp pose in meters.

        Returns:
            TaskSpacePose: A pose above the grasp point for safe approach.
        """
        # Approach from above to reduce the chance of colliding with the object or conveyor.
        return self.offset_pose_z(grasp_pose, z_offset)

    def create_lift_pose(
        self,
        grasp_pose: TaskSpacePose,
        z_offset: float = 0.12,
    ) -> TaskSpacePose:
        """
        Create a lifted retreat pose above the grasp pose after the object is picked.

        Inputs:
            grasp_pose: The task-space pose where the object is grasped.
            z_offset: Vertical retreat offset above the grasp pose in meters.

        Returns:
            TaskSpacePose: A pose above the grasp point for safe retreat.
        """
        # Lift vertically before traveling so the object clears the surface safely.
        return self.offset_pose_z(grasp_pose, z_offset)

    def create_pre_place_pose(
        self,
        place_pose: TaskSpacePose,
        z_offset: float = 0.10,
    ) -> TaskSpacePose:
        """
        Create a safe approach pose above the place pose.

        Inputs:
            place_pose: The final task-space pose where the object should be placed.
            z_offset: Vertical offset above the place pose in meters.

        Returns:
            TaskSpacePose: A pose above the place point for safe approach.
        """
        # Approach the placement point from above before descending to release.
        return self.offset_pose_z(place_pose, z_offset)

    def create_place_depart_pose(
        self,
        place_pose: TaskSpacePose,
        z_offset: float = 0.10,
    ) -> TaskSpacePose:
        """
        Create a safe retreat pose above the place pose after releasing the object.

        Inputs:
            place_pose: The final task-space pose where the object was placed.
            z_offset: Vertical retreat offset above the place pose in meters.

        Returns:
            TaskSpacePose: A pose above the place point for safe departure.
        """
        # Retreat upward after release to avoid brushing the placed object.
        return self.offset_pose_z(place_pose, z_offset)
    
    def plan_to_pose(self, pose: TaskSpacePose) -> bool:
        """
        Ask the planner for a motion to a task-space pose and log the result.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            bool: True if planning succeeded, otherwise False.
        """
        self.get_logger().info("Requesting a plan from the Panda planner...")

        result = self._planner.plan_to_task_pose(pose)

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")

        if result.joint_trajectory is not None:
            self.get_logger().info(
                f"Planned joint trajectory points: {len(result.joint_trajectory.points)}"
            )

        return result.success
    
    def plan_and_move_to_pose(
        self,
        pose: TaskSpacePose,
    ) -> bool:
        """
        Plan to a task-space pose and execute the returned joint trajectory.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            bool: True if planning and execution both succeeded, otherwise False.
        """
        self.get_logger().info(
            "Requesting planning and arm execution for a task-space pose..."
        )

        result = self._planner.plan_to_task_pose(pose)

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")

        if not result.success:
            self.get_logger().error("Cannot execute motion because planning failed.")
            return False

        if result.joint_trajectory is None:
            self.get_logger().error(
                "Cannot execute motion because no joint trajectory was returned."
            )
            return False

        self.get_logger().info(
            f"Executing planned joint trajectory with "
            f"{len(result.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            result.joint_trajectory,
            speed_scale=1.0
        )

        if not motion_succeeded:
            self.get_logger().error("Arm motion failed after successful planning.")
            return False

        self.get_logger().info("Planned arm motion completed successfully.")
        return True
    
    def plan_and_move_to_pose_with_orientation_constraint(
        self,
        pose: TaskSpacePose,
        orientation_tolerance_rad: float = 1.0,
    ) -> bool:
        """
        Plan to a task-space pose with an orientation path constraint and
        execute the returned joint trajectory.

        Inputs:
            pose: The target end-effector pose in task space.
            orientation_tolerance_rad: Maximum allowed orientation error in radians
                about each axis.

        Returns:
            bool: True if planning and execution both succeeded, otherwise False.
        """
        self.get_logger().info(
            "Requesting constrained planning and arm execution for a task-space pose..."
        )

        result = self._planner.plan_to_task_pose_with_orientation_constraint(
            pose,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")

        if not result.success:
            self.get_logger().error(
                "Cannot execute motion because constrained planning failed."
            )
            return False

        if result.joint_trajectory is None:
            self.get_logger().error(
                "Cannot execute motion because no joint trajectory was returned."
            )
            return False

        self.get_logger().info(
            f"Executing planned joint trajectory with "
            f"{len(result.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            result.joint_trajectory,
            speed_scale=1.0
        )

        if not motion_succeeded:
            self.get_logger().error(
                "Arm motion failed after successful constrained planning."
            )
            return False

        self.get_logger().info("Constrained planned arm motion completed successfully.")
        return True
    
    def execute_pick_and_place_for_pose(
        self,
        object_id: str,
        object_pose: TaskSpacePose,
        place_pose: TaskSpacePose,
    ) -> bool:
        """
        Execute a pick-and-place sequence for a provided object pose and place pose.

        Inputs:
            object_id: Planning-scene object name.
            object_pose: Task-space pose of the object to pick.
            place_pose: Task-space pose where the object should be placed.

        Returns:
            bool: True if the full pick-and-place sequence succeeded, otherwise False.
        """
        self.get_logger().info(
            f"Starting pick-and-place sequence for object '{object_id}'."
        )

        grasp_pose = self.create_grasp_pose(object_pose)
        pre_grasp_pose = self.create_pre_grasp_pose(grasp_pose, 0.05)
        lift_pose = self.create_lift_pose(grasp_pose, 0.10)
        pre_place_pose = self.create_pre_place_pose(place_pose, 0.05)
        place_depart_pose = self.create_place_depart_pose(place_pose, 0.10)

        placed_object_orientation = self.pose_to_orientation_xyzw(place_pose)

        self.get_logger().info(f"Moving to {object_id} pre-grasp pose...")
        if not self.plan_and_move_to_pose(pre_grasp_pose):
            self.get_logger().error(f"{object_id} pre-grasp motion failed.")
            return False

        rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(
            f"Moving to {object_id} grasp pose with orientation constraint..."
        )
        if not self.plan_and_move_to_pose_with_orientation_constraint(
            grasp_pose,
            orientation_tolerance_rad=1.0,
        ):
            self.get_logger().error(f"{object_id} constrained grasp motion failed.")
            return False

        rclpy.spin_once(self, timeout_sec=0.2)

        if not self._gripper.close_gripper():
            self.get_logger().error(f"{object_id} gripper close motion failed.")
            return False

        self._scene.remove_collision_object(object_id)
        rclpy.spin_once(self, timeout_sec=0.2)

        self._scene.attach_box_to_link(
            object_id=object_id,
            link_name="panda_hand",
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(0.0, 0.0, 0.10),
            orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
            touch_links=["panda_hand", "panda_leftfinger", "panda_rightfinger"],
        )
        rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(f"Moving to {object_id} lift pose with orientation constraint...")
        if not self.plan_and_move_to_pose_with_orientation_constraint(
            lift_pose,
            orientation_tolerance_rad=1.0,
        ):
            self.get_logger().error(f"{object_id} constrained lift motion failed.")
            return False

        rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(f"Moving to {object_id} pre-place pose...")
        if not self.plan_and_move_to_pose(pre_place_pose):
            self.get_logger().error(f"{object_id} pre-place motion failed.")
            return False

        rclpy.spin_once(self, timeout_sec=0.2)

        self.get_logger().info(f"Moving to {object_id} place pose with orientation constraint...")
        if not self.plan_and_move_to_pose_with_orientation_constraint(
            place_pose,
            orientation_tolerance_rad=1.0,
        ):
            self.get_logger().error(f"{object_id} constrained place motion failed.")
            return False

        rclpy.spin_once(self, timeout_sec=0.2)

        self._scene.detach_object(object_id, "panda_hand")
        rclpy.spin_once(self, timeout_sec=0.2)

        self._scene.add_box_collision_object(
            object_id=object_id,
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(place_pose.x, place_pose.y, place_pose.z),
            orientation_xyzw=placed_object_orientation,
            frame_id="world",
        )

        rclpy.spin_once(self, timeout_sec=0.2)

        if not self._gripper.open_gripper():
            self.get_logger().error(f"{object_id} gripper open motion failed.")
            return False

        self.get_logger().info(f"Moving to {object_id} place-depart pose...")
        if not self.plan_and_move_to_pose(place_depart_pose):
            self.get_logger().error(f"{object_id} place-depart motion failed.")
            return False

        self.get_logger().info(
            f"Pick-and-place sequence for object '{object_id}' completed successfully."
        )
        return True
    
    def run_cube_2_pick_approach_test(self) -> bool:
        """
        Run a smoke test for cube_2 using a hardcoded object pose and place pose.

        Inputs:
            None

        Returns:
            bool: True if the smoke test sequence succeeded, otherwise False.
        """
        cube_2_object_pose = TaskSpacePose(
            x=0.500,
            y=0.000,
            z=0.425,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        place_pose = self.create_place_pose(
            x=0.35,
            y=0.00,
            z=0.425,
            roll=3.14159,
            pitch=0.0,
            yaw=0.7854,
        )

        self.get_logger().info("Starting cube_2 pick-and-place smoke test...")

        self._scene.republish_static_environment()
        self._scene.add_cube_1_collision_object()
        self._scene.add_cube_2_collision_object()
        self._scene.add_cube_3_collision_object()

        return self.execute_pick_and_place_for_pose(
            object_id="cube_2",
            object_pose=cube_2_object_pose,
            place_pose=place_pose,
        )

def main(args=None):
    """
    Start the Panda coordinator node, check both control servers, and shut down cleanly.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # Initialize ROS before creating any nodes.
    rclpy.init(args=args)

    node = PandaCoordinatorNode()

    servers_ready = node.wait_for_control_servers()

    if servers_ready:
        node.get_logger().info("Coordinator setup succeeded. Arm and gripper are ready.")

        sequence_succeeded = node.run_startup_sequence()

        if sequence_succeeded:
            node.run_cube_2_pick_approach_test()
        else:
            node.get_logger().error("Coordinator startup sequence failed.")
    else:
        node.get_logger().error("Coordinator setup failed because one or more servers are unavailable.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()