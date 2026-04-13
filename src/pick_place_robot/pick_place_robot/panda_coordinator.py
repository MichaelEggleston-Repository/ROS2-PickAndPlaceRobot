import rclpy
from rclpy.node import Node
from dataclasses import dataclass

from pick_place_robot.panda_arm_control import PandaArmControl
from pick_place_robot.panda_gripper_control import PandaGripperControl

@dataclass
class TaskSpacePose:
    """
    Describe a target end-effector pose in task space.

    Inputs:
        x: Target x position in meters.
        y: Target y position in meters.
        z: Target z position in meters.
        roll: Target roll angle in radians.
        pitch: Target pitch angle in radians.
        yaw: Target yaw angle in radians.

    Returns:
        None
    """
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

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

        if not self._gripper.close_gripper():
            self.get_logger().error("Coordinator failed during gripper close motion.")
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
    
    def task_pose_to_joint_positions(self, pose: TaskSpacePose) -> list[float]:
        """
        Convert a task-space pose into Panda joint positions.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            list[float]: A 7-joint target for the Panda arm.

        Raises:
            NotImplementedError: This conversion has not been implemented yet.
        """
        # This method is the future boundary between task planning and motion execution.
        # Later, this can use IK, MoveIt, or another solver.
        raise NotImplementedError(
            "Task-space to joint-space conversion is not implemented yet."
        )
    
    def log_example_task_space_sequence(self) -> None:
        """
        Build and log an example task-space pick-and-place sequence.

        Inputs:
            None

        Returns:
            None
        """
        # This is a temporary sample object pose for testing the task-space logic.
        # Later, this should come from perception or another upstream source.
        object_pose = TaskSpacePose(
            x=0.45,
            y=0.00,
            z=0.05,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

        grasp_pose = self.create_grasp_pose(object_pose)
        pre_grasp_pose = self.create_pre_grasp_pose(grasp_pose)
        lift_pose = self.create_lift_pose(grasp_pose)

        place_pose = self.create_place_pose()
        pre_place_pose = self.create_pre_place_pose(place_pose)
        place_depart_pose = self.create_place_depart_pose(place_pose)

        self.get_logger().info("Example task-space sequence:")
        self.get_logger().info(f"  Object pose:       {object_pose}")
        self.get_logger().info(f"  Pre-grasp pose:    {pre_grasp_pose}")
        self.get_logger().info(f"  Grasp pose:        {grasp_pose}")
        self.get_logger().info(f"  Lift pose:         {lift_pose}")
        self.get_logger().info(f"  Pre-place pose:    {pre_place_pose}")
        self.get_logger().info(f"  Place pose:        {place_pose}")
        self.get_logger().info(f"  Place-depart pose: {place_depart_pose}")

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
            node.get_logger().info("Coordinator startup sequence succeeded.")
            node.log_example_task_space_sequence()
        else:
            node.get_logger().error("Coordinator startup sequence failed.")
    else:
        node.get_logger().error("Coordinator setup failed because one or more servers are unavailable.")

    # Clean shutdown keeps the node lifecycle tidy.
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()