import rclpy
from rclpy.node import Node
from dataclasses import dataclass
import math

from moveit.planning import MoveItPy

from geometry_msgs.msg import Pose, PoseStamped
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive

from pick_place_robot.moveit_config_loader import build_moveit_config_dict
from pick_place_robot.task_space_pose import TaskSpacePose

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

        self._planning_scene_publisher = self.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )
        self.get_logger().info("Planning scene publisher is ready.")

        self.add_static_environment()

    def add_box_collision_object(
        self,
        object_id: str,
        size_xyz: tuple[float, float, float],
        position_xyz: tuple[float, float, float],
        frame_id: str = "panda_link0",
    ) -> None:
        """
        Add a box-shaped collision object to the MoveIt planning scene.

        Inputs:
            object_id: Unique name for the collision object.
            size_xyz: Box dimensions in meters as (x, y, z).
            position_xyz: Box center position in meters as (x, y, z).
            frame_id: Frame in which the box pose is defined.

        Returns:
            None
        """
        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = list(size_xyz)

        pose = Pose()
        pose.position.x = position_xyz[0]
        pose.position.y = position_xyz[1]
        pose.position.z = position_xyz[2]
        pose.orientation.w = 1.0

        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = object_id
        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(pose)
        collision_object.operation = CollisionObject.ADD

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)

        self._planning_scene_publisher.publish(planning_scene)

        self.get_logger().info(
            f"Added collision object '{object_id}' in frame '{frame_id}' "
            f"with size {size_xyz} at position {position_xyz}."
        )

    def add_static_environment(self) -> None:
        """
        Add the fixed environment collision objects to the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.add_box_collision_object(
            object_id="conveyor_base",
            size_xyz=(2.0, 0.6, 0.4),
            position_xyz=(0.0, 0.0, 0.2),
            frame_id="world",
        )

    def republish_static_environment(self) -> None:
        """
        Republish the fixed environment collision objects to the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.get_logger().info("Republishing static environment collision objects.")
        self.add_static_environment()

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

        self.get_logger().info(
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
                joint_positions=None,
                message="MoveIt planning failed to produce a plan.",
            )

        if plan_result.trajectory is None:
            return PlanningResult(
                success=False,
                joint_positions=None,
                message="MoveIt planning failed because no trajectory was returned.",
            )
        
        trajectory_msg = plan_result.trajectory.get_robot_trajectory_msg()
        joint_trajectory = trajectory_msg.joint_trajectory

        final_point = joint_trajectory.points[-1]
        joint_positions = list(final_point.positions)

        return PlanningResult(
            success=True,
            joint_positions=joint_positions,
            message="MoveIt planning succeeded and joint positions were extracted.",
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

        self.get_logger().info("Starting planner smoke test...")
        self.republish_static_environment()
        result = self.plan_to_task_pose(test_pose)

        self.get_logger().info(f"Planning success: {result.success}")
        self.get_logger().info(f"Planning message: {result.message}")
        self.get_logger().info(
            "Joint positions returned: "
            f"{result.joint_positions is not None}"
        )
        self.get_logger().info(f"Returned joint positions: {result.joint_positions}")
        
def main(args=None):
    """
    Start the Panda MoveIt planner node and run a manual planning smoke test.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = PandaMoveItPlanner()

    try:
        node.run_planning_smoke_test()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()