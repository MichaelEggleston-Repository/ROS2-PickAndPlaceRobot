import rclpy
from rclpy.node import Node
import rclpy.time
from tf2_ros import Buffer, TransformListener, TransformException
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
import yaml
import cv2
from cv_bridge import CvBridge
import numpy as np
import time

from pick_place_interfaces.srv import CaptureSnapshot


@dataclass
class CalibrationCaptureResult:
    """
    Describe the result of one calibration pose capture step.

    Inputs:
        pose_name: Name of the calibration pose.
        motion_success: True if the robot reached the requested pose.
        image_capture_success: True if a valid camera snapshot was acquired.
        image_frame_id: Frame id of the captured RGB image if available.
        snapshot_response: Full camera snapshot service response if available.
        requested_tcp_pose: Requested TCP pose for this calibration step.
        robot_base_frame: Base frame used for robot pose lookup.
        robot_tool_frame: Tool frame used for robot pose lookup.
        base_to_tool_transform: Current base-to-tool transform captured from TF.

    Returns:
        None
    """
    pose_name: str
    motion_success: bool
    image_capture_success: bool
    image_frame_id: str | None
    snapshot_response: CaptureSnapshot.Response | None
    requested_tcp_pose: object | None
    robot_base_frame: str | None
    robot_tool_frame: str | None
    base_to_tool_transform: object | None

class CalibrationDataCollection:
    def __init__(self, node: Node):
        """
        Create a reusable calibration data collection helper for snapshot capture
        and dataset saving.

        Inputs:
            node: Parent ROS node that owns logging, timing, and clients.

        Returns:
            None
        """
        self._node = node

        self._settle_time_sec = 2.0
        self._capture_timeout_sec = 10.0

        self._capture_snapshot_client = self._node.create_client(
            CaptureSnapshot,
            "capture_snapshot",
        )

        self._robot_base_frame = "panda_link0"
        self._robot_tool_frame = "panda_hand_tcp"

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self._node)

        self._output_root_dir = (
            Path.home() / "Projects" / "ROS2-PickAndPlaceRobot" / "calibration_data"
        )

        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")
        self._session_output_dir = self._output_root_dir / session_name
        self._session_output_dir.mkdir(parents=True, exist_ok=True)

        self._node.get_logger().info(
            f"Calibration data session directory: {self._session_output_dir}"
        )

        self._node.get_logger().info("Calibration data collection helper ready.")

    def wait_for_future(self, future, timeout_sec: float, description: str) -> bool:
        """
        Wait for an asynchronous operation to complete while allowing this standalone
        calibration node to process callbacks.

        Inputs:
            future: The ROS future being waited on.
            timeout_sec: Maximum time to wait in seconds.
            description: Human-readable description for logging.

        Returns:
            bool: True if the future completed before timeout, otherwise False.
        """
        start_time = self._node.get_clock().now()

        while self._node.context.ok() and not future.done():
            elapsed_sec = (self._node.get_clock().now() - start_time).nanoseconds / 1e9

            if elapsed_sec > timeout_sec:
                self._node.get_logger().error(
                    f"Timed out waiting for {description} after {timeout_sec:.1f} seconds."
                )
                return False

            rclpy.spin_once(self._node, timeout_sec=0.1)

        return future.done()
    
    def wait_for_capture_service(self) -> bool:
        """
        Wait for the camera snapshot service to become available.

        Inputs:
            None

        Returns:
            bool: True if the service became available, otherwise False.
        """
        self._node.get_logger().info("Waiting for capture_snapshot service...")

        while rclpy.ok():
            if self._capture_snapshot_client.wait_for_service(timeout_sec=1.0):
                self._node.get_logger().info("capture_snapshot service is available.")
                return True

            self._node.get_logger().info(
                "capture_snapshot service not available yet, waiting again..."
            )

        return False
    
    def request_camera_snapshot(
        self,
        require_depth: bool = False,
    ) -> CaptureSnapshot.Response | None:
        """
        Request a camera snapshot from the camera acquisition service.

        Inputs:
            require_depth: True if a depth image is required, otherwise False.

        Returns:
            CaptureSnapshot.Response | None: Service response if successful,
            otherwise None.
        """
        request = CaptureSnapshot.Request()
        request.require_depth = require_depth
        request.timeout_sec = self._capture_timeout_sec

        future = self._capture_snapshot_client.call_async(request)

        if not self.wait_for_future(
            future,
            timeout_sec=self._capture_timeout_sec + 2.0,
            description="capture snapshot service response",
        ):
            return None

        if future.exception() is not None:
            self._node.get_logger().error(
                f"Capture snapshot service call raised an exception: {future.exception()}"
            )
            return None

        response = future.result()

        if response is None:
            self._node.get_logger().warn("Capture snapshot service returned no response.")
            return None

        return response
    
    def wait_for_robot_settle(self) -> None:
        """
        Pause briefly after robot motion before attempting image capture.

        Inputs:
            None

        Returns:
            None
        """
        self._node.get_logger().info(
            f"Waiting {self._settle_time_sec:.1f} s for robot to settle."
        )
        self._node.get_clock().sleep_for(
            rclpy.duration.Duration(seconds=self._settle_time_sec)
        )
    
    def lookup_current_tool_transform(self):
        """
        Look up the current robot base-to-tool transform from TF.

        Inputs:
            None

        Returns:
            TransformStamped | None:
                The current base-to-tool transform if available, otherwise None.
        """
        start_time = self._node.get_clock().now()
        timeout_sec = 2.0

        while self._node.context.ok():
            try:
                transform = self._tf_buffer.lookup_transform(
                    self._robot_base_frame,
                    self._robot_tool_frame,
                    rclpy.time.Time(),
                )
                return transform

            except TransformException:
                elapsed_sec = (self._node.get_clock().now() - start_time).nanoseconds / 1e9

                if elapsed_sec > timeout_sec:
                    break

                rclpy.spin_once(self._node, timeout_sec=0.1)

        self._node.get_logger().warn(
            "Failed to look up robot base-to-tool transform: "
            f"{self._robot_base_frame} -> {self._robot_tool_frame}."
        )
        return None
    
    def capture_at_pose(self, pose_name: str, requested_tcp_pose) -> CalibrationCaptureResult:
        """
        Capture a camera snapshot after the robot reaches a calibration pose.

        Inputs:
            pose_name: Name of the calibration pose being captured.
            requested_tcp_pose: Requested TCP pose for this calibration step.

        Returns:
            CalibrationCaptureResult: Summary of the capture attempt.
        """
        self.wait_for_robot_settle()

        base_to_tool_transform = self.lookup_current_tool_transform()

        if base_to_tool_transform is None:
            self._node.get_logger().warn(
                f"Robot pose lookup failed at pose '{pose_name}'."
            )
            return CalibrationCaptureResult(
                pose_name=pose_name,
                motion_success=True,
                image_capture_success=False,
                image_frame_id=None,
                snapshot_response=None,
                requested_tcp_pose=requested_tcp_pose,
                robot_base_frame=self._robot_base_frame,
                robot_tool_frame=self._robot_tool_frame,
                base_to_tool_transform=None,
            )

        response = self.request_camera_snapshot(require_depth=False)

        if response is None:
            self._node.get_logger().warn(
                f"Image capture service call failed at pose '{pose_name}'."
            )
            return CalibrationCaptureResult(
                pose_name=pose_name,
                motion_success=True,
                image_capture_success=False,
                image_frame_id=None,
                snapshot_response=None,
                requested_tcp_pose=requested_tcp_pose,
                robot_base_frame=self._robot_base_frame,
                robot_tool_frame=self._robot_tool_frame,
                base_to_tool_transform=base_to_tool_transform,
            )

        if not response.success:
            self._node.get_logger().warn(
                f"Image capture failed at pose '{pose_name}': {response.message}"
            )
            return CalibrationCaptureResult(
                pose_name=pose_name,
                motion_success=True,
                image_capture_success=False,
                image_frame_id=None,
                snapshot_response=None,
                requested_tcp_pose=requested_tcp_pose,
                robot_base_frame=self._robot_base_frame,
                robot_tool_frame=self._robot_tool_frame,
                base_to_tool_transform=base_to_tool_transform,
            )

        self._node.get_logger().info(
            f"Image captured at pose '{pose_name}' "
            f"from frame '{response.rgb_image.header.frame_id}'."
        )

        return CalibrationCaptureResult(
            pose_name=pose_name,
            motion_success=True,
            image_capture_success=True,
            image_frame_id=response.rgb_image.header.frame_id,
            snapshot_response=response,
            requested_tcp_pose=requested_tcp_pose,
            robot_base_frame=self._robot_base_frame,
            robot_tool_frame=self._robot_tool_frame,
            base_to_tool_transform=base_to_tool_transform,
        )
    
    def transform_to_dict(self, transform) -> dict | None:
        """
        Convert a ROS transform message into a YAML-friendly dictionary.

        Inputs:
            transform: TransformStamped message to convert.

        Returns:
            dict | None: Serialized transform dictionary if available, otherwise None.
        """
        if transform is None:
            return None

        return {
            "header": {
                "stamp": {
                    "sec": int(transform.header.stamp.sec),
                    "nanosec": int(transform.header.stamp.nanosec),
                },
                "frame_id": transform.header.frame_id,
            },
            "child_frame_id": transform.child_frame_id,
            "translation_m": {
                "x": float(transform.transform.translation.x),
                "y": float(transform.transform.translation.y),
                "z": float(transform.transform.translation.z),
            },
            "quaternion_xyzw": {
                "x": float(transform.transform.rotation.x),
                "y": float(transform.transform.rotation.y),
                "z": float(transform.transform.rotation.z),
                "w": float(transform.transform.rotation.w),
            },
        }
    
    def task_space_pose_to_dict(self, pose) -> dict | None:
        """
        Convert a task-space pose message into a YAML-friendly dictionary.

        Inputs:
            pose: Task-space pose message to convert.

        Returns:
            dict | None: Serialized pose dictionary if available, otherwise None.
        """
        if pose is None:
            return None

        return {
            "x": float(pose.x),
            "y": float(pose.y),
            "z": float(pose.z),
            "roll": float(pose.roll),
            "pitch": float(pose.pitch),
            "yaw": float(pose.yaw),
        }

    def make_yaml_safe(self, value):
        """
        Recursively convert nested data into YAML-safe built-in Python types.

        Inputs:
            value: Any nested value that may contain NumPy scalars, lists,
                tuples, or dictionaries.

        Returns:
            The same logical value converted into plain Python types that
            yaml.safe_dump can serialize safely.
        """
        if isinstance(value, dict):
            return {
                str(key): self.make_yaml_safe(item_value)
                for key, item_value in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [self.make_yaml_safe(item) for item in value]

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, np.floating):
            return float(value)

        if isinstance(value, np.ndarray):
            return [self.make_yaml_safe(item) for item in value.tolist()]

        return value

    def save_capture_result(
        self,
        result: CalibrationCaptureResult,
    ) -> None:
        """
        Save a calibration capture result to the current session output directory.

        Inputs:
            result: Completed calibration capture result containing the requested
                pose name, robot motion status, and captured camera data.

        Returns:
            None
        """
        if not result.image_capture_success:
            self._node.get_logger().warn(
                f"Skipping save for pose '{result.pose_name}' because image capture failed."
            )
            return

        if result.snapshot_response is None:
            self._node.get_logger().warn(
                f"Skipping save for pose '{result.pose_name}' because no snapshot response is available."
            )
            return

        bridge = CvBridge()
        pose_prefix = result.pose_name

        snapshot = result.snapshot_response

        rgb_image_path = self._session_output_dir / f"{pose_prefix}_rgb.png"
        camera_info_path = self._session_output_dir / f"{pose_prefix}_camera_info.yaml"
        metadata_path = self._session_output_dir / f"{pose_prefix}_metadata.yaml"

        rgb_cv_image = bridge.imgmsg_to_cv2(snapshot.rgb_image, desired_encoding="bgr8")
        cv2.imwrite(str(rgb_image_path), rgb_cv_image)

        if snapshot.depth_image.header.frame_id:
            depth_image_path = self._session_output_dir / f"{pose_prefix}_depth.exr"
            depth_cv_image = bridge.imgmsg_to_cv2(snapshot.depth_image, desired_encoding="passthrough")
            cv2.imwrite(str(depth_image_path), depth_cv_image)
        else:
            depth_image_path = None

        camera_info_data = {
            "header": {
                "stamp": {
                    "sec": int(snapshot.camera_info.header.stamp.sec),
                    "nanosec": int(snapshot.camera_info.header.stamp.nanosec),
                },
                "frame_id": snapshot.camera_info.header.frame_id,
            },
            "height": int(snapshot.camera_info.height),
            "width": int(snapshot.camera_info.width),
            "distortion_model": snapshot.camera_info.distortion_model,
            "d": list(snapshot.camera_info.d),
            "k": list(snapshot.camera_info.k),
            "r": list(snapshot.camera_info.r),
            "p": list(snapshot.camera_info.p),
            "binning_x": int(snapshot.camera_info.binning_x),
            "binning_y": int(snapshot.camera_info.binning_y),
        }

        metadata = {
            "pose_name": result.pose_name,
            "motion_success": result.motion_success,
            "image_capture_success": result.image_capture_success,
            "image_frame_id": result.image_frame_id,
            "rgb_image_path": str(rgb_image_path),
            "depth_image_path": str(depth_image_path) if depth_image_path is not None else None,
            "camera_info_path": str(camera_info_path),
            "snapshot_stamp": {
                "sec": int(snapshot.rgb_image.header.stamp.sec),
                "nanosec": int(snapshot.rgb_image.header.stamp.nanosec),
            },
            "robot_base_frame": result.robot_base_frame,
            "robot_tool_frame": result.robot_tool_frame,
            "requested_tcp_pose": self.task_space_pose_to_dict(result.requested_tcp_pose),
            "base_to_tool": self.transform_to_dict(result.base_to_tool_transform),
            "tool_to_tag": {
                "translation_m": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.01,
                },
                "quaternion_xyzw": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                    "w": 1.0,
                },
            },
            "calibration_tag_id": 0,
        }

        camera_info_data = self.make_yaml_safe(camera_info_data)
        metadata = self.make_yaml_safe(metadata)

        with open(camera_info_path, "w", encoding="utf-8") as camera_info_file:
            yaml.safe_dump(camera_info_data, camera_info_file, sort_keys=False)

        with open(metadata_path, "w", encoding="utf-8") as metadata_file:
            yaml.safe_dump(metadata, metadata_file, sort_keys=False)

        self._node.get_logger().info(
            f"Saved calibration sample for pose '{result.pose_name}' to {self._session_output_dir}"
        )