import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CameraInfo, Image

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
    
    def camera_streams_healthy(self) -> bool:
        """
        Check whether all required camera streams are currently healthy.

        Inputs:
            None

        Returns:
            bool: True if RGB, depth, and camera info are all fresh, otherwise False.
        """
        return (
            self.stream_is_fresh(self._last_rgb_received_time)
            and self.stream_is_fresh(self._last_depth_received_time)
            and self.stream_is_fresh(self._last_camera_info_received_time)
        )

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