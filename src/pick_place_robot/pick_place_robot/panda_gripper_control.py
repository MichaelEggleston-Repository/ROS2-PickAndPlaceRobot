import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.callback_groups import CallbackGroup
import time

from control_msgs.action import FollowJointTrajectory
from control_msgs.msg import JointTolerance
from trajectory_msgs.msg import JointTrajectoryPoint

# Open and closed positions for each Panda finger joint.
GRIPPER_OPEN_POSITION = 0.04
GRIPPER_CLOSED_POSITION = 0.0

# Duration of the gripper trajectory in simulation seconds.
# The JTC reports success when this many sim-seconds have elapsed.
# Gazebo can run significantly slower than real-time after a heavy arm
# motion (observed ~10 % RT), so 1 sim-second ≈ 10 wall-seconds.
# Keep this short in sim-time; rely on the increased wait timeouts below
# to accommodate the slower-than-RT physics.
GRIPPER_MOVE_DURATION_SEC = 1

# Both finger joints are commanded directly by the trajectory controller.
# There is no mimic plugin — each joint must be driven explicitly by the JTC.
GRIPPER_JOINT_NAMES = [
    "panda_finger_joint1",
    "panda_finger_joint2",
]

class PandaGripperControl:
    def __init__(self, node: Node, callback_group: CallbackGroup | None = None):
        """
        Create a reusable Panda gripper control helper attached to an existing ROS node.

        Inputs:
            node: The ROS 2 node that owns logging and spinning context.

        Returns:
            None
        """
        # The control class uses an existing node instead of creating its own.
        # That keeps it reusable inside the coordinator.
        self._node = node

        self._gripper_client = ActionClient(
            node,
            FollowJointTrajectory,
            "/gripper_trajectory_controller/follow_joint_trajectory",
            callback_group=callback_group,
        )

    def wait_for_server(self) -> bool:
        """
        Wait for the Panda gripper action server to become available.

        Inputs:
            None

        Returns:
            bool: True if the action server is available, otherwise False.
        """
        self._node.get_logger().info("Waiting for Panda gripper action server...")

        available = self._gripper_client.wait_for_server(timeout_sec=30.0)

        if not available:
            self._node.get_logger().error("Panda gripper action server was not found.")
            return False

        self._node.get_logger().info("Panda gripper action server is ready.")
        return True
    
    def _wait_for_future(self, future, timeout_sec: float, description: str) -> bool:
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

    def create_goal(
        self,
        finger_position: float,
        duration_sec: int,
        start_positions: list[float] | None = None,
        goal_tolerance_m: float | None = None,
    ) -> FollowJointTrajectory.Goal:
        """
        Build a FollowJointTrajectory goal for the Panda gripper.

        Inputs:
            finger_position: Target position for each finger joint.
            duration_sec: Target motion duration in seconds.
            start_positions: Optional list of [joint1, joint2] actual current
                positions in metres.  When provided, a t=0 waypoint is
                prepended using the real measured positions so the JTC does
                not need to sample hardware state itself.

                Without this, the JTC reads both finger joint states from the
                hardware interface at the start of trajectory execution.  When
                called through a ROS service, the extra thread-dispatch
                latency means that sample can straddle two Gazebo physics
                steps.  Providing the actual /joint_states positions eliminates
                the hardware-sample race completely.  The values must be read
                from /joint_states immediately before calling this method;
                using stale or assumed constants defeats the purpose.
            goal_tolerance_m: Optional per-finger position tolerance (metres)
                embedded in the JTC goal message.  When None the JTC uses its
                controller-level default (tight).  Pass (object_half_width +
                margin) when closing on an object so the JTC accepts the
                fingers stopping at the object surface rather than at 0.0 m.

        Returns:
            FollowJointTrajectory.Goal: A goal containing the requested gripper motion.
        """
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = GRIPPER_JOINT_NAMES

        n = len(GRIPPER_JOINT_NAMES)

        if start_positions is not None and len(start_positions) == n:
            # Explicit t=0 waypoint anchors the joint(s) to the same reference
            # frame so the JTC never samples hardware state at trajectory start.
            start_point = JointTrajectoryPoint()
            start_point.positions = list(start_positions)
            start_point.velocities = [0.0] * n
            start_point.time_from_start.sec = 0
            start_point.time_from_start.nanosec = 0
            goal.trajectory.points.append(start_point)

        point = JointTrajectoryPoint()
        point.positions  = [finger_position] * n
        # Zero terminal velocities tell the controller to stop cleanly at the
        # target rather than extrapolating through it.
        point.velocities = [0.0] * n
        point.time_from_start.sec = duration_sec

        goal.trajectory.points.append(point)

        # Per-call goal tolerance — overrides the controller-level default when
        # the caller knows the fingers will stop against an object surface.
        if goal_tolerance_m is not None:
            for joint_name in GRIPPER_JOINT_NAMES:
                tol = JointTolerance()
                tol.name = joint_name
                tol.position = goal_tolerance_m
                goal.goal_tolerance.append(tol)

        return goal

    def send_goal(
        self,
        goal: FollowJointTrajectory.Goal,
        timeout_sec: float = 10.0,
    ):
        self._node.get_logger().info("Sending gripper goal to Panda gripper controller...")

        send_goal_future = self._gripper_client.send_goal_async(goal)

        if not self._wait_for_future(
            send_goal_future,
            timeout_sec=timeout_sec,
            description="gripper goal acceptance",
        ):
            return None

        if send_goal_future.exception() is not None:
            self._node.get_logger().error(
                f"Gripper send-goal failed with exception: {send_goal_future.exception()}"
            )
            return None

        goal_handle = send_goal_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "No goal handle was returned by the gripper action server."
            )
            return None

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "Gripper goal was rejected by the action server."
            )
            return None

        self._node.get_logger().info("Gripper goal was accepted by the action server.")
        return goal_handle

    def wait_for_result(self, goal_handle, timeout_sec: float = 120.0) -> bool:
        self._node.get_logger().info("Waiting for Panda gripper motion to finish...")

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(
            result_future,
            timeout_sec=timeout_sec,
            description="gripper motion result",
        ):
            return False

        if result_future.exception() is not None:
            self._node.get_logger().error(
                f"Gripper result future failed with exception: {result_future.exception()}"
            )
            return False

        result = result_future.result()

        if result is None:
            self._node.get_logger().error("No result was returned for the gripper goal.")
            return False

        if result.status != 4:
            self._node.get_logger().error(
                f"Gripper goal did not finish successfully. Status code: {result.status}"
            )
            return False

        self._node.get_logger().info("Panda gripper motion completed successfully.")
        return True
    
    def move_to_position(
        self,
        finger_position: float,
        duration_sec: int,
        start_positions: list[float] | None = None,
        goal_tolerance_m: float | None = None,
    ) -> bool:
        """
        Move the Panda gripper to a custom symmetric finger position.

        Inputs:
            finger_position: Target position for each finger joint.
            duration_sec: Target motion duration in seconds.
            start_positions: Optional [joint1, joint2] actual current
                positions in metres.  See create_goal for the full explanation
                of why this prevents the service-callback timing race
                condition.
            goal_tolerance_m: Optional per-finger position tolerance to embed
                in the JTC goal.  See create_goal for full explanation.

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        goal = self.create_goal(
            finger_position, duration_sec, start_positions, goal_tolerance_m
        )
        goal_handle = self.send_goal(goal)

        if goal_handle is None:
            return False

        return self.wait_for_result(goal_handle)

    def open_gripper(self) -> bool:
        """
        Open the Panda gripper and wait for completion.

        Inputs:
            None

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        return self.move_to_position(
            GRIPPER_OPEN_POSITION,
            GRIPPER_MOVE_DURATION_SEC,
        )

    def close_gripper(self) -> bool:
        """
        Close the Panda gripper and wait for completion.

        Inputs:
            None

        Returns:
            bool: True if the motion completed successfully, otherwise False.
        """
        return self.move_to_position(
            GRIPPER_CLOSED_POSITION,
            GRIPPER_MOVE_DURATION_SEC,
        )
    
def main(args=None):
    """
    Run a simple manual test of PandaGripperControl by opening and closing the gripper.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    # This main is intended as a manual integration test.
    rclpy.init(args=args)

    node = Node("panda_gripper_control_test")
    gripper = PandaGripperControl(node)

    try:
        if not gripper.wait_for_server():
            node.get_logger().error("Gripper server was not available for the test.")
            raise SystemExit(1)

        node.get_logger().info("Starting PandaGripperControl open/close test...")

        if not gripper.open_gripper():
            node.get_logger().error("Open-gripper test step failed.")
            raise SystemExit(1)

        if not gripper.close_gripper():
            node.get_logger().error("Close-gripper test step failed.")
            raise SystemExit(1)

        node.get_logger().info("PandaGripperControl open/close test succeeded.")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()