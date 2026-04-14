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

    def create_place_pose(self) -> TaskSpacePose:
        """
        Create the final place pose in task space.

        Inputs:
            None

        Returns:
            TaskSpacePose: The place pose for dropping the object.
        """
        return TaskSpacePose(
            x=0.30,
            y=-0.30,
            z=0.18,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
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
    
    def create_pre_grasp_pose(self, grasp_pose: TaskSpacePose) -> TaskSpacePose:
        """
        Create a safe approach pose above the grasp pose.

        Inputs:
            grasp_pose: The task-space pose where the gripper should perform the grasp.

        Returns:
            TaskSpacePose: A pose above the grasp point for safe approach.
        """
        # Approach from above to reduce the chance of colliding with the object or conveyor.
        return self.offset_pose_z(grasp_pose, 0.10)

    def create_lift_pose(self, grasp_pose: TaskSpacePose) -> TaskSpacePose:
        """
        Create a lifted retreat pose above the grasp pose after the object is picked.

        Inputs:
            grasp_pose: The task-space pose where the object is grasped.

        Returns:
            TaskSpacePose: A pose above the grasp point for safe retreat.
        """
        # Lift vertically before traveling so the object clears the surface safely.
        return self.offset_pose_z(grasp_pose, 0.12)

    def create_pre_place_pose(self, place_pose: TaskSpacePose) -> TaskSpacePose:
        """
        Create a safe approach pose above the place pose.

        Inputs:
            place_pose: The final task-space pose where the object should be placed.

        Returns:
            TaskSpacePose: A pose above the place point for safe approach.
        """
        # Approach the placement point from above before descending to release.
        return self.offset_pose_z(place_pose, 0.10)

    def create_place_depart_pose(self, place_pose: TaskSpacePose) -> TaskSpacePose:
        """
        Create a safe retreat pose above the place pose after releasing the object.

        Inputs:
            place_pose: The final task-space pose where the object was placed.

        Returns:
            TaskSpacePose: A pose above the place point for safe departure.
        """
        # Retreat upward after release to avoid brushing the placed object.
        return self.offset_pose_z(place_pose, 0.10)
    
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

        if result.joint_positions is not None:
            self.get_logger().info(
                f"Planned joint positions: {result.joint_positions}"
            )

        return result.success
    
    def plan_and_move_to_pose(
        self,
        pose: TaskSpacePose,
        duration_sec: int = 5,
    ) -> bool:
        """
        Plan to a task-space pose and execute the returned joint target.

        Inputs:
            pose: The target end-effector pose in task space.
            duration_sec: The arm motion duration in seconds.

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

        if result.joint_positions is None:
            self.get_logger().error(
                "Cannot execute motion because no joint positions were returned."
            )
            return False

        self.get_logger().info(
            f"Executing planned joint positions: {result.joint_positions}"
        )

        motion_succeeded = self._arm.move_to_joint_positions(
            result.joint_positions,
            duration_sec,
        )

        if not motion_succeeded:
            self.get_logger().error("Arm motion failed after successful planning.")
            return False

        self.get_logger().info("Planned arm motion completed successfully.")
        return True
    
    def run_cube_top_pose_test(self) -> bool:
        """
        Run a simple planning and motion test to hover above each cube.

        Inputs:
            None

        Returns:
            bool: True if all cube-top motions succeeded, otherwise False.
        """
        cube_1_top_pose = TaskSpacePose(
            x=0.600,
            y=-0.15,
            z=0.5,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        cube_2_top_pose = TaskSpacePose(
            x=0.500,
            y=0.000,
            z=0.5,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        cube_3_top_pose = TaskSpacePose(
            x=0.550,
            y=0.250,
            z=0.5,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        self.get_logger().info("Starting cube top pose test...")

        self._scene.republish_static_environment()
        self._scene.add_cube_1_collision_object()
        self._scene.add_cube_2_collision_object()
        self._scene.add_cube_3_collision_object()

        self.get_logger().info("Testing cube_1 top pose...")
        if not self.plan_and_move_to_pose(cube_1_top_pose):
            self.get_logger().error("Cube_1 top pose test failed.")
            return False

        self.get_logger().info("Cube_1 top pose test succeeded.")

        self.get_logger().info("Testing cube_2 top pose...")
        if not self.plan_and_move_to_pose(cube_2_top_pose):
            self.get_logger().error("Cube_2 top pose test failed.")
            return False

        self.get_logger().info("Cube_2 top pose test succeeded.")

        self.get_logger().info("Testing cube_3 top pose...")
        if not self.plan_and_move_to_pose(cube_3_top_pose):
            self.get_logger().error("Cube_3 top pose test failed.")
            return False

        self.get_logger().info("Cube_3 top pose test succeeded.")
        self.get_logger().info("Cube top pose test completed successfully.")

        sequence_succeeded = self.run_startup_sequence()

        return True

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
            node.run_cube_top_pose_test()
        else:
            node.get_logger().error("Coordinator startup sequence failed.")
    else:
        node.get_logger().error("Coordinator setup failed because one or more servers are unavailable.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()