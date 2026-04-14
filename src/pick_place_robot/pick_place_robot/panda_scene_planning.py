import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import CollisionObject, PlanningScene
from shape_msgs.msg import SolidPrimitive


class PandaPlanningScene:
    def __init__(self, node: Node):
        """
        Create a planning-scene helper that publishes collision objects.

        Inputs:
            node: ROS 2 node used for publishers, logging, and spinning.

        Returns:
            None
        """
        self._node = node
        self._planning_scene_publisher = self._node.create_publisher(
            PlanningScene,
            "/planning_scene",
            10,
        )

        self._node.get_logger().info("Planning scene helper is ready.")

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

        self._node.get_logger().info(
            f"Added collision object '{object_id}' in frame '{frame_id}' "
            f"with size {size_xyz} at position {position_xyz}."
        )

    def add_cube_1_collision_object(self) -> None:
        """
        Add cube_1 as a collision object in the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.add_box_collision_object(
            object_id="cube_1",
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(-0.15, 0.0, 0.425),
            frame_id="world",
        )
    
    def add_cube_2_collision_object(self) -> None:
        """
        Add cube_2 as a collision object in the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.add_box_collision_object(
            object_id="cube_2",
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(0.0, 0.1, 0.425),
            frame_id="world",
        )

    def add_cube_3_collision_object(self) -> None:
        """
        Add cube_3 as a collision object in the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.add_box_collision_object(
            object_id="cube_3",
            size_xyz=(0.05, 0.05, 0.05),
            position_xyz=(0.25, 0.0, 0.425),
            frame_id="world",
        )

    def add_test_cubes(self) -> None:
        """
        Add all three test cubes to the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self.add_cube_1_collision_object()
        self.add_cube_2_collision_object()
        self.add_cube_3_collision_object()

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

    def remove_collision_object(
        self,
        object_id: str,
        frame_id: str = "world",
    ) -> None:
        """
        Remove a collision object from the MoveIt planning scene.

        Inputs:
            object_id: Unique name for the collision object to remove.
            frame_id: Frame in which the collision object is defined.

        Returns:
            None
        """
        collision_object = CollisionObject()
        collision_object.header.frame_id = frame_id
        collision_object.id = object_id
        collision_object.operation = CollisionObject.REMOVE

        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.world.collision_objects.append(collision_object)

        self._planning_scene_publisher.publish(planning_scene)
        rclpy.spin_once(self._node, timeout_sec=0.1)

        self._node.get_logger().info(
            f"Removed collision object '{object_id}' from frame '{frame_id}'."
        )

    def republish_static_environment(self) -> None:
        """
        Republish the fixed environment collision objects to the MoveIt planning scene.

        Inputs:
            None

        Returns:
            None
        """
        self._node.get_logger().info("Republishing static environment collision objects.")
        self.add_static_environment()