import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
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
        orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        frame_id: str = "world",      
    ) -> None:
        """
        Add a box collision object to the planning scene.

        Inputs:
            object_id: Unique planning-scene object name.
            size_xyz: Box dimensions in meters.
            position_xyz: Box center position in the given frame.
            frame_id: Reference frame for the box pose.
            orientation_xyzw: Box pose orientation as a quaternion.

        Returns:
            None
        """
        planning_scene = PlanningScene()
        planning_scene.is_diff = True

        collision_object = CollisionObject()
        collision_object.id = object_id
        collision_object.header.frame_id = frame_id
        collision_object.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size_xyz[0], size_xyz[1], size_xyz[2]]

        primitive_pose = Pose()
        primitive_pose.position.x = position_xyz[0]
        primitive_pose.position.y = position_xyz[1]
        primitive_pose.position.z = position_xyz[2]
        primitive_pose.orientation.x = orientation_xyzw[0]
        primitive_pose.orientation.y = orientation_xyzw[1]
        primitive_pose.orientation.z = orientation_xyzw[2]
        primitive_pose.orientation.w = orientation_xyzw[3]

        collision_object.primitives.append(primitive)
        collision_object.primitive_poses.append(primitive_pose)

        planning_scene.world.collision_objects.append(collision_object)

        self._planning_scene_publisher.publish(planning_scene)

        self._node.get_logger().info(
            f"Added collision object '{object_id}' in frame '{frame_id}'."
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
            orientation_xyzw = (0.0, 0.0, 0.0, 1.0),
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
            orientation_xyzw = (0.0, 0.0, 0.0, 1.0),
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
            orientation_xyzw = (0.0, 0.0, 0.0, 1.0),
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
            orientation_xyzw = (0.0, 0.0, 0.0, 1.0),
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

    def attach_box_to_link(
        self,
        object_id: str,
        link_name: str,
        size_xyz: tuple[float, float, float],
        position_xyz: tuple[float, float, float],
        orientation_xyzw: tuple[float, float, float, float],
        touch_links: list[str],
    ) -> None:
        """
        Attach a box collision object to a robot link in the planning scene.

        Inputs:
            object_id: Unique planning-scene object name.
            link_name: Robot link that the object should be attached to.
            size_xyz: Box dimensions in meters.
            position_xyz: Box pose position relative to the attach link.
            orientation_xyzw: Box pose orientation relative to the attach link.
            touch_links: Robot links that are allowed to touch the attached object.

        Returns:
            None
        """
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.robot_state.is_diff = True

        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.touch_links = touch_links

        attached_object.object.id = object_id
        attached_object.object.header.frame_id = link_name
        attached_object.object.operation = CollisionObject.ADD

        primitive = SolidPrimitive()
        primitive.type = SolidPrimitive.BOX
        primitive.dimensions = [size_xyz[0], size_xyz[1], size_xyz[2]]

        primitive_pose = Pose()
        primitive_pose.position.x = position_xyz[0]
        primitive_pose.position.y = position_xyz[1]
        primitive_pose.position.z = position_xyz[2]
        primitive_pose.orientation.x = orientation_xyzw[0]
        primitive_pose.orientation.y = orientation_xyzw[1]
        primitive_pose.orientation.z = orientation_xyzw[2]
        primitive_pose.orientation.w = orientation_xyzw[3]

        attached_object.object.primitives.append(primitive)
        attached_object.object.primitive_poses.append(primitive_pose)

        planning_scene.robot_state.attached_collision_objects.append(attached_object)

        self._planning_scene_publisher.publish(planning_scene)

        self._node.get_logger().info(
            f"Attached collision object '{object_id}' to link '{link_name}'."
        )
    
    def detach_object(self, object_id: str, link_name: str) -> None:
        """
        Detach a collision object from a robot link in the planning scene.

        Inputs:
            object_id: Unique planning-scene object name.
            link_name: Robot link the object is currently attached to.

        Returns:
            None
        """
        planning_scene = PlanningScene()
        planning_scene.is_diff = True
        planning_scene.robot_state.is_diff = True

        attached_object = AttachedCollisionObject()
        attached_object.link_name = link_name
        attached_object.object.id = object_id
        attached_object.object.operation = CollisionObject.REMOVE

        planning_scene.robot_state.attached_collision_objects.append(attached_object)

        self._planning_scene_publisher.publish(planning_scene)

        self._node.get_logger().info(
            f"Detached collision object '{object_id}' from link '{link_name}'."
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