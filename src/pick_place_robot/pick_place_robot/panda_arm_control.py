import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import CallbackGroup
from copy import deepcopy
import time

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

# Home pose values are kept as constants so they are easy to tune later.
HOME_JOINT_POSITIONS = [-1.5708, -0.4, 0.0, -2.0, 0.0, 1.6, 0.8]
HOME_MOVE_DURATION_SEC = 2

# Keep the joint ordering in one place so every goal uses the same mapping.
ARM_JOINT_NAMES = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]

class PandaArmControl:
    def __init__(self, node: Node, callback_group: CallbackGroup | None = None):
        """
        Create a reusable Panda arm control helper attached to an existing ROS node.

        Inputs:
            node: The ROS 2 node that owns logging and spinning context.

        Returns:
            None
        """
        # The control class uses an existing node instead of creating its own.
        # That keeps it reusable inside the coordinator.
        self._node = node

        self._arm_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/joint_trajectory_controller/follow_joint_trajectory",
            callback_group=callback_group,
        )

    def wait_for_server(self) -> bool:
        """
        Wait for the Panda arm action server to become available.

        Inputs:
            None

        Returns:
            bool: True if the action server is available, otherwise False.
        """
        self._node.get_logger().info("Waiting for Panda arm action server...")

        available = self._arm_client.wait_for_server(timeout_sec=30.0)

        if not available:
            self._node.get_logger().error("Panda arm action server was not found.")
            return False

        self._node.get_logger().info("Panda arm action server is ready.")
        return True

    def scale_joint_trajectory_timing(
        self,
        joint_trajectory: JointTrajectory,
        speed_scale: float,
    ) -> JointTrajectory:
        """
        Create a copy of a joint trajectory with scaled timing, velocities,
        and accelerations.

        Inputs:
            joint_trajectory: The original planned joint trajectory.
            speed_scale: Motion speed scale factor. Values above 1.0 speed up
                the motion. Values below 1.0 slow it down.

        Returns:
            JointTrajectory: A scaled copy of the input joint trajectory.
        """
        if speed_scale <= 0.0:
            raise ValueError("speed_scale must be greater than 0.0.")

        scaled_trajectory = deepcopy(joint_trajectory)

        for point in scaled_trajectory.points:
            total_sec = (
                float(point.time_from_start.sec)
                + float(point.time_from_start.nanosec) / 1e9
            )

            scaled_sec = total_sec / speed_scale
            scaled_whole_sec = int(scaled_sec)
            scaled_nanosec = int((scaled_sec - scaled_whole_sec) * 1e9)

            point.time_from_start.sec = scaled_whole_sec
            point.time_from_start.nanosec = scaled_nanosec

            if point.velocities:
                point.velocities = [float(v) * speed_scale for v in point.velocities]

            if point.accelerations:
                point.accelerations = [
                    float(a) * speed_scale * speed_scale for a in point.accelerations
                ]

        return scaled_trajectory

    def create_goal(
        self,
        joint_positions: list[float],
        duration_sec: int,
    ) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for a Panda arm joint target.

        Inputs:
            joint_positions: A list of 7 joint target values in Panda joint order.
            duration_sec: Target motion duration in seconds.

        Returns:
            FollowJointTrajectory.Goal: A goal containing the requested arm motion.
        """
        if len(joint_positions) != len(ARM_JOINT_NAMES):
            raise ValueError(
                f"Expected {len(ARM_JOINT_NAMES)} joint positions, got {len(joint_positions)}."
            )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = duration_sec

        self._node.get_logger().info(
            f"Creating arm goal with duration_sec={duration_sec}"
        )

        goal.trajectory.points.append(point)
        return goal
    
    def prepare_joint_trajectory_for_execution(
        self,
        joint_trajectory: JointTrajectory,
        terminal_hold_sec: float = 0.25,
    ) -> JointTrajectory:
        """
        Create a controller-friendly copy of a planned joint trajectory.

        Inputs:
            joint_trajectory: The planned joint trajectory.
            terminal_hold_sec: Extra hold time added at the final point.

        Returns:
            JointTrajectory: A modified trajectory ready for controller execution.
        """
        prepared_trajectory = deepcopy(joint_trajectory)

        if not prepared_trajectory.points:
            return prepared_trajectory

        final_point = prepared_trajectory.points[-1]

        if final_point.velocities:
            final_point.velocities = [0.0] * len(final_point.velocities)

        if final_point.accelerations:
            final_point.accelerations = [0.0] * len(final_point.accelerations)

        if terminal_hold_sec > 0.0:
            hold_point = deepcopy(final_point)

            if hold_point.velocities:
                hold_point.velocities = [0.0] * len(hold_point.velocities)

            if hold_point.accelerations:
                hold_point.accelerations = [0.0] * len(hold_point.accelerations)

            final_time_sec = (
                float(final_point.time_from_start.sec)
                + float(final_point.time_from_start.nanosec) / 1e9
            )
            hold_time_sec = final_time_sec + terminal_hold_sec

            hold_point.time_from_start.sec = int(hold_time_sec)
            hold_point.time_from_start.nanosec = int(
                (hold_time_sec - int(hold_time_sec)) * 1e9
            )

            prepared_trajectory.points.append(hold_point)

        return prepared_trajectory
    
    def _wait_for_future(
            self, 
            future, 
            timeout_sec: float, 
            description: str,
    ) -> bool:
        
        start_time = self._node.get_clock().now()

        while self._node.context.ok() and not future.done():
            elapsed_sec = (self._node.get_clock().now() - start_time).nanoseconds / 1e9

            if elapsed_sec > timeout_sec:
                self._node.get_logger().error(
                    f"Timed out waiting for {description} after {timeout_sec:.1f} seconds."
                )
                return False

            time.sleep(0.01)

        return future.done()

    def send_goal(
        self,
        goal: FollowJointTrajectory.Goal,
        timeout_sec: float = 10.0,
    ):
        self._node.get_logger().info("Sending arm goal to Panda arm controller...")

        send_goal_future = self._arm_client.send_goal_async(goal)

        if not self._wait_for_future(
            send_goal_future,
            timeout_sec=timeout_sec,
            description="arm goal acceptance",
        ):
            return None

        if send_goal_future.exception() is not None:
            self._node.get_logger().error(
                f"Arm send-goal failed with exception: {send_goal_future.exception()}"
            )
            return None

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "No goal handle was returned by the arm action server."
            )
            return None

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "Arm goal was rejected by the action server."
            )
            return None

        self._node.get_logger().info("Arm goal was accepted by the action server.")
        return goal_handle

    def wait_for_result(
        self,
        goal_handle,
        timeout_sec: float = 120.0,
    ) -> bool:
        self._node.get_logger().info("Waiting for Panda arm motion to finish...")

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(
            result_future,
            timeout_sec=timeout_sec,
            description="arm motion result",
        ):
            return False

        if result_future.exception() is not None:
            self._node.get_logger().error(
                f"Arm result future failed with exception: {result_future.exception()}"
            )
            return False

        result = result_future.result()

        if result is None:
            self._node.get_logger().error("No result was returned for the arm goal.")
            return False

        if result.status != 4:
            self._node.get_logger().error(
                f"Arm goal did not finish successfully. Status code: {result.status}"
            )
            return False

        self._node.get_logger().info("Panda arm motion completed successfully.")
        return True

    def move_to_joint_positions(
        self,
        joint_positions: list[float],
        duration_sec: int,
    ) -> bool:
        """
        Move the Panda arm to a specific joint target and wait for completion.

        Inputs:
            joint_positions: A list of 7 joint target values in Panda joint order.
            duration_sec: Target motion duration in seconds.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        goal = self.create_goal(joint_positions, duration_sec)
        goal_handle = self.send_goal(goal)

        if goal_handle is None:
            return False

        return self.wait_for_result(goal_handle)
    
    def move_to_joint_trajectory(
        self,
        joint_trajectory: JointTrajectory,
        speed_scale: float = 1.0,
    ) -> bool:
        """
        Send a full planned joint trajectory to the Panda arm controller and wait
        for execution to finish.

        Inputs:
            joint_trajectory: The full joint trajectory to execute.
            speed_scale: Motion speed scale factor. Values above 1.0 speed up
                the motion. Values below 1.0 slow it down.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        if not joint_trajectory.joint_names:
            self._node.get_logger().error(
                "Cannot execute joint trajectory because no joint names were provided."
            )
            return False

        if not joint_trajectory.points:
            self._node.get_logger().error(
                "Cannot execute joint trajectory because it contains no points."
            )
            return False
        
        scaled_trajectory = self.scale_joint_trajectory_timing(
            joint_trajectory,
            speed_scale,
        )
        prepared_trajectory = self.prepare_joint_trajectory_for_execution(
            scaled_trajectory,
            terminal_hold_sec=0.0,
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = prepared_trajectory
        # Do NOT set goal_time_tolerance here.  When goal_time_tolerance > 0
        # the arm JTC stays in an active monitoring state after the trajectory
        # clock expires, continuously running its update() loop on all 7 arm
        # joints until it decides the goal is complete.  That extended
        # transition period (up to 2 sim-seconds = ~20 wall-seconds at 10% RT)
        # appears to cause a race condition in the controller manager that
        # desynchronises the gripper finger joint trajectory initialisation on
        # the very next gripper command.  With goal_time_tolerance = 0 (the
        # default) the arm JTC transitions to IDLE immediately when the
        # trajectory clock expires — the same behaviour as move_home_unplanned
        # — which leaves the controller in a clean state for the gripper.

        self._node.get_logger().info(
            "Sending full joint trajectory to Panda arm controller: "
            f"{len(prepared_trajectory.points)} points, speed_scale={speed_scale:.2f}"
        )

        self._node.get_logger().info(
            "Prepared trajectory timing: "
            f"points={len(prepared_trajectory.points)}, "
            f"final_time_sec="
            f"{prepared_trajectory.points[-1].time_from_start.sec + prepared_trajectory.points[-1].time_from_start.nanosec / 1e9:.3f}"
        )

        final_point = prepared_trajectory.points[-1]
        self._node.get_logger().info(
            "Prepared final trajectory point: "
            f"time={final_point.time_from_start.sec + final_point.time_from_start.nanosec / 1e9:.3f}, "
            f"velocities={list(final_point.velocities) if final_point.velocities else []}, "
            f"accelerations={list(final_point.accelerations) if final_point.accelerations else []}"
        )

        goal_handle = self.send_goal(goal)

        if goal_handle is None:
            return False

        return self.wait_for_result(goal_handle)

    def move_home_unplanned(self) -> bool:
        """
        Move the Panda arm directly to the predefined home joint configuration
        WITHOUT collision-aware path planning.

        *** STARTUP USE ONLY ***
        This method sends a raw joint trajectory to the controller and does NOT
        invoke MoveIt / OMPL.  It is intentionally restricted to the coordinator
        startup sequence, where the arm is known to be in a safe, clear initial
        state.  For all post-startup homing (e.g. after a step failure) use
        plan_and_move_home() instead so that collision avoidance is active.

        Inputs:
            None

        Returns:
            bool: True if the home motion completed successfully, otherwise False.
        """
        return self.move_to_joint_positions(
            HOME_JOINT_POSITIONS,
            HOME_MOVE_DURATION_SEC,
        )
    
def main(args=None):
    """
    Run a simple manual test of PandaArmControl by commanding the home pose.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # This main is intended as a manual integration test
    rclpy.init(args=args)

    node = Node("panda_arm_control_test")
    arm = PandaArmControl(node)

    try:
        if not arm.wait_for_server():
            node.get_logger().error("Arm server was not available for the test.")
            raise SystemExit(1)

        node.get_logger().info("Starting PandaArmControl home-motion test...")

        if not arm.move_home():
            node.get_logger().error("PandaArmControl home-motion test failed.")
            raise SystemExit(1)

        node.get_logger().info("PandaArmControl home-motion test succeeded.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()