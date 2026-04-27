import rclpy
from rclpy.node import Node
from dataclasses import dataclass
import time

from sensor_msgs.msg import CameraInfo, Image
from pick_place_interfaces.srv import CaptureSnapshot

@dataclass
class CameraSnapshot:
    """
    Describe one captured set of camera messages.

    Inputs:
        rgb_image: Latest RGB image message.
        depth_image: Latest depth image message.
        camera_info: Latest camera info message.

    Returns:
        None
    """
    rgb_image: Image
    depth_image: Image | None
    camera_info: CameraInfo

class CameraAcquisitionNode(Node):
    def __init__(self):
        """
        Create a camera acquisition node for validating the simulated camera streams.

        Inputs:
            None

        Returns:
            None
        """
        super().__init__("camera_acquisition_node")

        self._latest_rgb = None
        self._latest_depth = None
        self._latest_camera_info = None

        self._last_rgb_received_time = None
        self._last_depth_received_time = None
        self._last_camera_info_received_time = None

        self._rgb_subscription = self.create_subscription(
            Image,
            "/conveyor_camera/image",
            self.rgb_callback,
            10,
        )

        self._depth_subscription = self.create_subscription(
            Image,
            "/conveyor_camera/depth_image",
            self.depth_callback,
            10,
        )

        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/conveyor_camera/camera_info",
            self.camera_info_callback,
            10,
        )

        self._capture_snapshot_service = self.create_service(
            CaptureSnapshot,
            "capture_snapshot",
            self.capture_snapshot_callback,
        )

        # Periodic status logging keeps acquisition checks in one place before
        # calibration logic is added.
        self._status_timer = self.create_timer(2.0, self.log_camera_status)

        self.get_logger().info("Camera acquisition node started.")

    def rgb_callback(self, msg: Image) -> None:
        """
        Store the latest RGB image message.

        Inputs:
            msg: Latest RGB image message.

        Returns:
            None
        """
        self._latest_rgb = msg
        self._last_rgb_received_time = self.get_clock().now()

    def depth_callback(self, msg: Image) -> None:
        """
        Store the latest depth image message.

        Inputs:
            msg: Latest depth image message.

        Returns:
            None
        """
        self._latest_depth = msg
        self._last_depth_received_time = self.get_clock().now()

    def camera_info_callback(self, msg: CameraInfo) -> None:
        """
        Store the latest camera info message.

        Inputs:
            msg: Latest camera info message.

        Returns:
            None
        """
        self._latest_camera_info = msg
        self._last_camera_info_received_time = self.get_clock().now()

    def rgb_stream_ready(self, stale_after_sec: float = 2.0) -> bool:
        """
        Check whether the RGB stream is ready for capture.

        Inputs:
            stale_after_sec: Maximum allowed message age in seconds.

        Returns:
            bool: True if the RGB stream has a recent message, otherwise False.
        """
        return (
            self._latest_rgb is not None
            and self.stream_is_fresh(
                self._last_rgb_received_time,
                stale_after_sec,
            )
        )

    def depth_stream_ready(self, stale_after_sec: float = 2.0) -> bool:
        """
        Check whether the depth stream is ready for capture.

        Inputs:
            stale_after_sec: Maximum allowed message age in seconds.

        Returns:
            bool: True if the depth stream has a recent message, otherwise False.
        """
        return (
            self._latest_depth is not None
            and self.stream_is_fresh(
                self._last_depth_received_time,
                stale_after_sec,
            )
        )

    def camera_info_ready(self, stale_after_sec: float = 2.0) -> bool:
        """
        Check whether the camera info stream is ready for capture.

        Inputs:
            stale_after_sec: Maximum allowed message age in seconds.

        Returns:
            bool: True if the camera info stream has a recent message, otherwise False.
        """
        return (
            self._latest_camera_info is not None
            and self.stream_is_fresh(
                self._last_camera_info_received_time,
                stale_after_sec,
            )
        )
    
    def camera_streams_healthy(
        self,
        stale_after_sec: float = 2.0,
        require_depth: bool = True,
    ) -> bool:
        """
        Check whether the required camera streams are currently healthy.

        Inputs:
            stale_after_sec: Maximum allowed message age in seconds.
            require_depth: True if the depth stream is required, otherwise False.

        Returns:
            bool: True if all required streams are fresh, otherwise False.
        """
        rgb_ready = self.rgb_stream_ready(stale_after_sec)
        camera_info_ready = self.camera_info_ready(stale_after_sec)

        if not require_depth:
            return rgb_ready and camera_info_ready

        depth_ready = self.depth_stream_ready(stale_after_sec)
        return rgb_ready and depth_ready and camera_info_ready

    def stream_is_fresh(
        self,
        last_received_time,
        stale_after_sec: float = 2.0,
    ) -> bool:
        """
        Check whether a stream has received a message recently enough.

        Inputs:
            last_received_time: The node time when the latest message was received.
            stale_after_sec: Maximum allowed age in seconds before the stream is stale.

        Returns:
            bool: True if the stream is fresh, otherwise False.
        """
        if last_received_time is None:
            return False

        elapsed_sec = (
            self.get_clock().now() - last_received_time
        ).nanoseconds / 1e9

        return elapsed_sec <= stale_after_sec
    
    def camera_streams_healthy(
        self,
        stale_after_sec: float = 2.0,
        require_depth: bool = True,
    ) -> bool:
        """
        Check whether the required camera streams are currently healthy.

        Inputs:
            stale_after_sec: Maximum allowed message age in seconds.
            require_depth: True if the depth stream is required, otherwise False.

        Returns:
            bool: True if all required streams are fresh, otherwise False.
        """
        rgb_ready = self.rgb_stream_ready(stale_after_sec)
        camera_info_ready = self.camera_info_ready(stale_after_sec)

        if not require_depth:
            return rgb_ready and camera_info_ready

        depth_ready = self.depth_stream_ready(stale_after_sec)
        return rgb_ready and depth_ready and camera_info_ready

    def log_camera_status(self) -> None:
        """
        Log the current camera stream status for acquisition verification.

        Inputs:
            None

        Returns:
            None
        """
        rgb_ready = self._latest_rgb is not None
        depth_ready = self._latest_depth is not None
        camera_info_ready = self._latest_camera_info is not None

        rgb_fresh = self.stream_is_fresh(self._last_rgb_received_time)
        depth_fresh = self.stream_is_fresh(self._last_depth_received_time)
        camera_info_fresh = self.stream_is_fresh(
            self._last_camera_info_received_time
        )

        streams_healthy = self.camera_streams_healthy()

        self.get_logger().info(
            "Camera status: "
            f"streams_healthy={streams_healthy}, "
            f"rgb_ready={rgb_ready}, "
            f"rgb_fresh={rgb_fresh}, "
            f"depth_ready={depth_ready}, "
            f"depth_fresh={depth_fresh}, "
            f"camera_info_ready={camera_info_ready}, "
            f"camera_info_fresh={camera_info_fresh}"
        )

        if self._latest_rgb is not None:
            self.get_logger().info(
                "RGB image: "
                f"frame_id={self._latest_rgb.header.frame_id}, "
                f"width={self._latest_rgb.width}, "
                f"height={self._latest_rgb.height}, "
                f"encoding={self._latest_rgb.encoding}"
            )

        if self._latest_depth is not None:
            self.get_logger().info(
                "Depth image: "
                f"frame_id={self._latest_depth.header.frame_id}, "
                f"width={self._latest_depth.width}, "
                f"height={self._latest_depth.height}, "
                f"encoding={self._latest_depth.encoding}"
            )

        if self._latest_camera_info is not None:
            self.get_logger().info(
                "Camera info: "
                f"frame_id={self._latest_camera_info.header.frame_id}, "
                f"width={self._latest_camera_info.width}, "
                f"height={self._latest_camera_info.height}"
            )

    def get_latest_snapshot(
        self,
        require_depth: bool = True,
        stale_after_sec: float = 2.0,
    ) -> CameraSnapshot | None:
        """
        Return the latest valid camera snapshot if all required data is available.

        Inputs:
            require_depth: True if a depth image is required, otherwise False.
            stale_after_sec: Maximum allowed message age in seconds.

        Returns:
            CameraSnapshot | None: The latest valid snapshot, otherwise None.
        """
        if not self.camera_streams_healthy(
            stale_after_sec=stale_after_sec,
            require_depth=require_depth,
        ):
            return None

        return CameraSnapshot(
            rgb_image=self._latest_rgb,
            depth_image=self._latest_depth if require_depth else None,
            camera_info=self._latest_camera_info,
        )
    
    def wait_for_fresh_snapshot(
        self,
        timeout_sec: float = 5.0,
        stale_after_sec: float = 2.0,
        require_depth: bool = True,
    ) -> CameraSnapshot | None:
        """
        Wait until a fresh camera snapshot is available or the timeout expires.

        Inputs:
            timeout_sec: Maximum time to wait for valid camera data.
            stale_after_sec: Maximum allowed message age in seconds.
            require_depth: True if a depth image is required, otherwise False.

        Returns:
            CameraSnapshot | None: A fresh snapshot if available, otherwise None.
        """
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)

        while rclpy.ok():
            snapshot = self.get_latest_snapshot(
                require_depth=require_depth,
                stale_after_sec=stale_after_sec,
            )

            if snapshot is not None:
                return snapshot

            if self.get_clock().now().nanoseconds >= deadline:
                return None

            time.sleep(0.1)

        return None
    
    def capture_snapshot_callback(
        self,
        request: CaptureSnapshot.Request,
        response: CaptureSnapshot.Response,
    ) -> CaptureSnapshot.Response:
        """
        Handle a snapshot capture service request.

        Inputs:
            request: Capture request containing depth and timeout settings.
            response: Service response to populate with capture results.

        Returns:
            CaptureSnapshot.Response: The populated snapshot response.
        """
        self.get_logger().info(
            "Received capture snapshot request: "
            f"require_depth={request.require_depth}, "
            f"timeout_sec={request.timeout_sec:.2f}"
        )
        
        snapshot = self.wait_for_fresh_snapshot(
            timeout_sec=request.timeout_sec,
            stale_after_sec=2.0,
            require_depth=request.require_depth,
        )

        if snapshot is None:
            response.success = False
            response.message = "Failed to acquire a fresh camera snapshot."
            self.get_logger().warn(response.message)
            return response

        response.success = True
        response.message = "Camera snapshot acquired successfully."
        response.rgb_image = snapshot.rgb_image
        response.camera_info = snapshot.camera_info

        if request.require_depth and snapshot.depth_image is not None:
            response.depth_image = snapshot.depth_image

        self.get_logger().info(
            "Returning camera snapshot: "
            f"rgb_frame_id={response.rgb_image.header.frame_id}, "
            f"camera_info_frame_id={response.camera_info.header.frame_id}"
        )

        return response

def main(args=None):
    """
    Start the camera acquisition node and spin until shutdown.

    Inputs:
        args: Optional ROS argument list.

    Returns:
        None
    """
    rclpy.init(args=args)
    node = CameraAcquisitionNode()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()