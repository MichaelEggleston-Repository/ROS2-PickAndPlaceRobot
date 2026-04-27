import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

import math
import time
import threading

from std_msgs.msg import String

from pick_place_robot.panda_arm_control import PandaArmControl
from pick_place_robot.panda_gripper_control import PandaGripperControl
from pick_place_interfaces.srv import PlanToTaskPose, ExecuteTaskPose
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
        
        self._execute_task_pose_service = self.create_service(
            ExecuteTaskPose,
            "execute_task_pose",
            self.execute_task_pose_callback,
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


        self._is_ready = False

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
        Wait for the planner service to become available.

        Inputs:
            None

        Returns:
            bool: True if the service became available, otherwise False.
        """
        self.get_logger().info("Waiting for plan_to_task_pose service...")

        while rclpy.ok():
            if self._plan_to_task_pose_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("plan_to_task_pose service is available.")
                return True

            self.get_logger().info(
                "plan_to_task_pose service not available yet, waiting again..."
            )

        return False

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

        self.get_logger().info("Coordinator motion sequence completed successfully.")
        return True
    
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
    ) -> PlanToTaskPose.Response | None:
        """
        Request a joint-trajectory plan for a task-space pose from the planner service.

        Inputs:
            pose: Target task-space pose message.
            use_orientation_constraint: True to request constrained planning.
            orientation_tolerance_rad: Allowed orientation error in radians about
                each axis when constrained planning is requested.

        Returns:
            PlanToTaskPose.Response | None: Service response if successful,
            otherwise None.
        """
        request = PlanToTaskPose.Request()
        request.pose = pose
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
    
    def plan_to_pose(self, pose: TaskSpacePoseMsg) -> bool:
        """
        Request a plan to a task-space pose and log the result.

        Inputs:
            pose: The target end-effector pose in task space.

        Returns:
            bool: True if planning succeeded, otherwise False.
        """

        if not self.require_ready():
            return False

        self.get_logger().info("Requesting a plan from the Panda planner service...")

        response = self.request_plan_to_task_pose(
            pose,
            use_orientation_constraint=False,
        )

        if response is None:
            self.get_logger().error("Planning request failed because no response was received.")
            return False

        self.get_logger().info(f"Planning success: {response.success}")
        self.get_logger().info(f"Planning message: {response.message}")

        if response.joint_trajectory.points:
            self.get_logger().info(
                f"Planned joint trajectory points: {len(response.joint_trajectory.points)}"
            )

        return response.success
    
    def plan_and_move_to_pose(
        self,
        pose: TaskSpacePoseMsg,
        speed_scale: float = 1.0,
    ) -> tuple[bool, str]:
        """
        Request a plan to a task-space pose and execute the returned joint trajectory.

        Inputs:
            pose: The target end-effector pose in task space.
            speed_scale: Motion speed scale factor for trajectory execution.

        Returns:
            tuple[bool, str]:
                Success flag and descriptive result message.
        """
        if not self.require_ready():
            return False, "Coordinator is not ready. Startup sequence is not complete."

        self.get_logger().info(
            "Requesting planning and arm execution for a task-space pose..."
        )

        response = self.request_plan_to_task_pose(
            pose,
            use_orientation_constraint=False,
        )

        if response is None:
            message = "Cannot execute motion because no planning response was received."
            self.get_logger().error(message)
            return False, message

        self.get_logger().info(f"Planning success: {response.success}")
        self.get_logger().info(f"Planning message: {response.message}")

        if not response.success:
            message = "Cannot execute motion because planning failed."
            self.get_logger().error(message)
            return False, message

        if not response.joint_trajectory.points:
            message = "Cannot execute motion because no joint trajectory was returned."
            self.get_logger().error(message)
            return False, message

        self.get_logger().info(
            f"Executing planned joint trajectory with "
            f"{len(response.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            response.joint_trajectory,
            speed_scale=speed_scale,
        )

        if not motion_succeeded:
            message = "Arm motion failed after successful planning."
            self.get_logger().error(message)
            return False, message

        message = "Planned arm motion completed successfully."
        self.get_logger().info(message)
        return True, message
    
    def plan_and_move_to_pose_with_orientation_constraint(
        self,
        pose: TaskSpacePoseMsg,
        orientation_tolerance_rad: float = 1.0,
    ) -> bool:
        """
        Request a constrained plan to a task-space pose and execute the returned
        joint trajectory.

        Inputs:
            pose: The target end-effector pose in task space.
            orientation_tolerance_rad: Maximum allowed orientation error in radians
                about each axis.

        Returns:
            bool: True if planning and execution both succeeded, otherwise False.
        """
        if not self.require_ready():
            return False
        
        self.get_logger().info(
            "Requesting constrained planning and arm execution for a task-space pose..."
        )

        response = self.request_plan_to_task_pose(
            pose,
            use_orientation_constraint=True,
            orientation_tolerance_rad=orientation_tolerance_rad,
        )

        if response is None:
            self.get_logger().error(
                "Cannot execute motion because no constrained planning response was received."
            )
            return False

        self.get_logger().info(f"Planning success: {response.success}")
        self.get_logger().info(f"Planning message: {response.message}")

        if not response.success:
            self.get_logger().error(
                "Cannot execute motion because constrained planning failed."
            )
            return False

        if not response.joint_trajectory.points:
            self.get_logger().error(
                "Cannot execute motion because no joint trajectory was returned."
            )
            return False

        self.get_logger().info(
            f"Executing planned joint trajectory with "
            f"{len(response.joint_trajectory.points)} points."
        )

        motion_succeeded = self._arm.move_to_joint_trajectory(
            response.joint_trajectory,
            speed_scale=1.0,
        )

        if not motion_succeeded:
            self.get_logger().error(
                "Arm motion failed after successful constrained planning."
            )
            return False

        self.get_logger().info("Constrained planned arm motion completed successfully.")
        return True
    
    def create_planning_smoke_test_pose(self) -> TaskSpacePoseMsg:
        """
        Create a simple task-space pose for planner/coordinator integration testing.

        Inputs:
            None

        Returns:
            TaskSpacePoseMsg: A reachable test pose for basic planning validation.
        """
        return TaskSpacePoseMsg(
            x=0.45,
            y=0.00,
            z=0.55,
            roll=3.14159,
            pitch=0.0,
            yaw=0.0,
        )

    def run_planning_smoke_test(self) -> bool:
        """
        Run a simple planner/coordinator integration smoke test.

        Inputs:
            None

        Returns:
            bool: True if the test motion completed successfully, otherwise False.
        """
        test_pose = self.create_planning_smoke_test_pose()

        self.get_logger().info("Starting planner/coordinator smoke test...")

        motion_succeeded = self.plan_and_move_to_pose(test_pose)

        if not motion_succeeded:
            self.get_logger().error(
                "Planner/coordinator smoke test failed."
            )
            return False

        self.get_logger().info(
            "Planner/coordinator smoke test completed successfully."
        )
        return True

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