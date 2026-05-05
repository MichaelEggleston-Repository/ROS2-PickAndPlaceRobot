import rclpy
from rclpy.node import Node

import numpy as np
from cv_bridge import CvBridge
import cv2

from pick_place_vision.detected_cube import DetectedCube

from sensor_msgs.msg import Image, CameraInfo
from pick_place_interfaces.msg import DetectedObject
from pick_place_interfaces.srv import DetectedObjects




# Depth range (metres from camera) within which a pixel is considered to
# belong to a cube sitting on the conveyor surface.
#
# The camera is mounted at z ≈ 1.5 m (base frame).  The conveyor top surface
# is at z = 0.3 m, giving a nominal cube-top depth of ≈ 1.15 m.  A stack of
# three 5 cm cubes raises the top to ≈ 0.45 m, giving a minimum depth of
# ≈ 1.05 m.  A generous lower margin of 0.85 m keeps even taller stacks.
#
# The Panda arm passes through the camera field of view during pick/place
# sequences.  With the camera at z = 1.5 m and arm links at z ≈ 0.7–0.9 m,
# arm pixels appear at depth ≈ 0.6–0.8 m.  The lower bound of 0.85 m
# reliably rejects arm false-positives while keeping all cube detections.
_CUBE_DEPTH_MIN_M: float = 0.85
_CUBE_DEPTH_MAX_M: float = 1.20


class ObjectPerception(Node):
    def __init__(self) -> None:
        super().__init__("object_perception")

        self._cv_bridge = CvBridge()

        # Tracks whether the debug overlay has been shown once this session.
        # Only relevant when debug_visualization_enabled is True.
        self._show_debug_mask_once = False

        self.declare_parameter("debug_visualization_enabled", False)
        self._debug_visualization_enabled = (
            self.get_parameter("debug_visualization_enabled")
            .get_parameter_value()
            .bool_value
        )

        self._latest_rgb_image = None
        self._latest_depth_image = None
        self._latest_camera_info = None

        self._detected_cubes: dict[str, DetectedCube] = {}

        self._rgb_subscription = self.create_subscription(
            Image,
            "/conveyor_camera/image",
            self._handle_rgb_image,
            10,
        )

        self._depth_subscription = self.create_subscription(
            Image,
            "/conveyor_camera/depth_image",
            self._handle_depth_image,
            10,
        )

        self._camera_info_subscription = self.create_subscription(
            CameraInfo,
            "/conveyor_camera/camera_info",
            self._handle_camera_info,
            10,
        )

        self._detect_objects_service = self.create_service(
            DetectedObjects,
            "detect_objects",
            self._handle_detect_objects,
        )

        self.get_logger().info("Object perception node ready.")

    def _handle_rgb_image(self, msg: Image) -> None:
        self._latest_rgb_image = self._cv_bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="rgb8",
        )

    def _handle_depth_image(self, msg: Image) -> None:
        self._latest_depth_image = self._cv_bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="32FC1",
        )

    def _handle_camera_info(self, msg: CameraInfo) -> None:
        self._latest_camera_info = msg

    def _camera_data_ready(self) -> bool:
        return (
            self._latest_rgb_image is not None
            and self._latest_depth_image is not None
            and self._latest_camera_info is not None
        )

    def _find_largest_blob_contour(self, mask: np.ndarray) -> np.ndarray | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        return max(contours, key=cv2.contourArea)
    
    def _compute_contour_centroid(
        self,
        contour: np.ndarray,
    ) -> tuple[int, int] | None:
        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            return None

        centroid_u = int(moments["m10"] / moments["m00"])
        centroid_v = int(moments["m01"] / moments["m00"])

        return centroid_u, centroid_v

    def _show_debug_overlay(
        self,
        window_name: str,
        rgb_image: np.ndarray,
        mask: np.ndarray,
        object_contour: np.ndarray | None = None,
    ) -> None:
        if not self._debug_visualization_enabled:
            return

        if self._show_debug_mask_once:
            return

        self._show_debug_mask_once = True

        overlay_image = rgb_image.copy()
        overlay_image[mask > 0] = [0, 255, 0]

        display_image = cv2.addWeighted(rgb_image, 0.7, overlay_image, 0.3, 0.0)

        if object_contour is not None:
            cv2.drawContours(display_image, [object_contour], -1, (255, 255, 255), 2)

        display_image_bgr = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        cv2.imshow(window_name, display_image_bgr)

        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1:
            cv2.waitKey(50)

        cv2.destroyWindow(window_name)

    def _build_color_mask(
        self,
        rgb_image: np.ndarray,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
    ) -> np.ndarray:
        hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv_image, lower_hsv, upper_hsv)
        return mask
    
    def _project_pixel_to_camera_frame(
        self,
        u: int,
        v: int,
        depth_m: float,
    ) -> list[float]:
        fx = self._latest_camera_info.k[0]
        fy = self._latest_camera_info.k[4]
        cx = self._latest_camera_info.k[2]
        cy = self._latest_camera_info.k[5]

        x = (u - cx) * depth_m / fx
        y = (v - cy) * depth_m / fy
        z = depth_m

        return [x, y, z]
    
    def _detect_cube_by_color(
        self,
        cube_id: str,
        lower_hsv: np.ndarray,
        upper_hsv: np.ndarray,
        debug_window_name: str,
        extra_mask: np.ndarray | None = None,
    ) -> DetectedCube | None:
        """
        Detect a single coloured cube in the latest RGB-D frame.

        Returns a DetectedCube containing all image-space and 3D data,
        or None if the cube is not found or geometry estimation fails.

        The full colour mask (including any visible side faces) is used for
        height estimation only. All other geometry — contour, centroid, width
        — is derived from the top face mask so side face pixels do not skew
        the results.

        Inputs:
            cube_id: Identifier string for the detected object.
            lower_hsv: Lower HSV threshold for the colour range.
            upper_hsv: Upper HSV threshold for the colour range.
            debug_window_name: Name to use for the optional debug overlay window.
            extra_mask: Optional additional mask to OR with the colour mask.
                        Used for colours like red that wrap around the HSV boundary.

        Returns:
            DetectedCube | None: Detection result, or None if not found.
        """
        color_mask = self._build_color_mask(
            self._latest_rgb_image, lower_hsv, upper_hsv
        )

        if extra_mask is not None:
            color_mask = cv2.bitwise_or(color_mask, extra_mask)

        top_face_mask, top_depth_m = self._extract_top_face_mask(color_mask)

        # Bail out immediately if no valid depth pixels were found in the
        # expected range.  top_depth_m == 0.0 is the sentinel for this case.
        if top_depth_m <= 0.0:
            return None

        top_face_contour = self._find_largest_blob_contour(top_face_mask)

        self._show_debug_overlay(
            debug_window_name,
            self._latest_rgb_image,
            top_face_mask,
            top_face_contour,
        )

        if top_face_contour is None:
            return None

        centroid = self._compute_contour_centroid(top_face_contour)

        if centroid is None:
            return None

        u, v = centroid

        # Use the minimum depth from the validated top-face pixels rather than
        # sampling the depth image at the centroid pixel.  The per-pixel depth
        # value at the centroid can be noisy or interpolated; the minimum over
        # the whole top-face region is more robust and corresponds to the
        # actual top-surface distance of the cube.

        position_xyz = self._project_pixel_to_camera_frame(u, v, top_depth_m)

        estimated_half_height_m = self._estimate_cube_half_height_m(
            color_mask, top_depth_m
        )
        estimated_half_width_m = self._estimate_cube_half_width_m(
            top_face_contour, top_depth_m
        )

        return DetectedCube(
            id=cube_id,
            color_mask=color_mask,
            top_face_mask=top_face_mask,
            top_face_contour=top_face_contour,
            centroid_uv=(u, v),
            top_depth_m=top_depth_m,
            position_camera_xyz=position_xyz,
            estimated_half_height_m=estimated_half_height_m,
            estimated_half_width_m=estimated_half_width_m,
            confidence=1.0,
        )
    
    def _detect_all_cubes(self) -> None:
        """
        Run detection for all known cube colours and update the internal
        _detected_cubes world model.

        Cubes that are successfully detected are added or updated in the dict.
        Cubes that are not detected are removed so the dict always reflects
        the current observed state rather than stale previous detections.
        """
        red_lower_1 = np.array([0, 100, 100], dtype=np.uint8)
        red_upper_1 = np.array([10, 255, 255], dtype=np.uint8)
        red_lower_2 = np.array([170, 100, 100], dtype=np.uint8)
        red_upper_2 = np.array([179, 255, 255], dtype=np.uint8)

        green_lower = np.array([50, 100, 100], dtype=np.uint8)
        green_upper = np.array([90, 255, 255], dtype=np.uint8)

        blue_lower = np.array([90, 60, 40], dtype=np.uint8)
        blue_upper = np.array([135, 255, 255], dtype=np.uint8)

        red_extra_mask = self._build_color_mask(
            self._latest_rgb_image, red_lower_2, red_upper_2
        )

        cube_configs = [
            ("red_cube",   red_lower_1,  red_upper_1,  "red_cube_debug",   red_extra_mask),
            ("green_cube", green_lower,  green_upper,  "green_cube_debug", None),
            ("blue_cube",  blue_lower,   blue_upper,   "blue_cube_debug",  None),
        ]

        for cube_id, lower_hsv, upper_hsv, window_name, extra_mask in cube_configs:
            result = self._detect_cube_by_color(
                cube_id, lower_hsv, upper_hsv, window_name, extra_mask
            )

            if result is not None:
                self._detected_cubes[cube_id] = result
            elif cube_id in self._detected_cubes:
                del self._detected_cubes[cube_id]
    
    def _extract_top_face_mask(
        self,
        color_mask: np.ndarray,
        depth_tolerance_m: float = 0.015,
    ) -> tuple[np.ndarray, float]:
        """
        Filter a colour mask down to pixels belonging to the top face only.

        Side faces are at a greater depth than the top face.  By keeping only
        pixels within depth_tolerance_m of the minimum depth found in the
        mask, we discard side face pixels and retain only the top surface.

        A depth range gate of [_CUBE_DEPTH_MIN_M, _CUBE_DEPTH_MAX_M] is
        applied before selecting the minimum.  This rejects pixels belonging
        to the robot arm or other near-field objects that can fall inside the
        colour mask's HSV range and produce false detections.

        Inputs:
            color_mask: Binary mask from HSV colour detection (may include sides).
            depth_tolerance_m: Depth window above the top surface to keep.

        Returns:
            tuple[np.ndarray, float]:
                - top_face_mask: Binary mask containing only top face pixels.
                  An all-zero mask is returned when no valid depths are found.
                - min_depth_m: Minimum depth of the valid top-face pixels
                  (metres).  0.0 is returned when no valid depths are found,
                  which the caller must treat as a detection failure.
        """
        depths_in_mask = self._latest_depth_image[color_mask > 0]
        valid_depths = depths_in_mask[
            np.isfinite(depths_in_mask)
            & (depths_in_mask >= _CUBE_DEPTH_MIN_M)
            & (depths_in_mask <= _CUBE_DEPTH_MAX_M)
        ]

        if len(valid_depths) == 0:
            self.get_logger().warn(
                "Could not extract top face mask: no pixels with depth in "
                f"[{_CUBE_DEPTH_MIN_M:.2f}, {_CUBE_DEPTH_MAX_M:.2f}] m. "
                "Possible robot-arm false-positive or cube out of range."
            )
            return np.zeros_like(color_mask), 0.0

        min_depth_m = float(np.min(valid_depths))

        top_face_mask = np.zeros_like(color_mask)
        top_face_mask[
            (color_mask > 0)
            & np.isfinite(self._latest_depth_image)
            & (self._latest_depth_image >= _CUBE_DEPTH_MIN_M)
            & (self._latest_depth_image <= _CUBE_DEPTH_MAX_M)
            & (self._latest_depth_image >= min_depth_m)
            & (self._latest_depth_image <= min_depth_m + depth_tolerance_m)
        ] = 255

        return top_face_mask, min_depth_m
    
    def _estimate_cube_half_height_m(
        self,
        color_mask: np.ndarray,
        cube_top_depth_m: float,
    ) -> float:
        """
        Estimate half the cube height from the depth difference between the
        cube top surface and the surrounding conveyor surface.

        A ring of pixels just outside the full colour mask (which covers the
        entire visible cube footprint including sides) is used to sample the
        conveyor depth. The original colour mask is used here rather than the
        top face mask so the dilation ring sits well clear of the cube.

        Inputs:
            color_mask: Full HSV colour mask including any visible side faces.
            cube_top_depth_m: Depth at the top face centroid in metres.

        Returns:
            float: Estimated half-height in metres, or 0.0 if estimation fails.
        """
        kernel = np.ones((15, 15), np.uint8)
        dilated_mask = cv2.dilate(color_mask, kernel, iterations=1)
        border_mask = cv2.subtract(dilated_mask, color_mask)

        border_depths = self._latest_depth_image[border_mask > 0]
        valid_depths = border_depths[
            np.isfinite(border_depths)
            & (border_depths >= _CUBE_DEPTH_MIN_M)
            & (border_depths <= _CUBE_DEPTH_MAX_M)
        ]

        if len(valid_depths) == 0:
            self.get_logger().warn(
                "Could not estimate cube height: no valid depths in border region."
            )
            return 0.0

        conveyor_depth_m = float(np.median(valid_depths))
        cube_height_m = conveyor_depth_m - cube_top_depth_m

        if cube_height_m <= 0.0:
            self.get_logger().warn(
                f"Cube height estimate was non-positive ({cube_height_m:.4f}m). "
                "Cube top depth should be less than conveyor depth."
            )
            return 0.0

        return cube_height_m / 2.0
    
    def _estimate_cube_half_width_m(
        self,
        top_face_contour: np.ndarray,
        cube_top_depth_m: float,
    ) -> float:
        """
        Estimate half the cube width by projecting the top face contour
        bounding box into 3D using the pinhole camera model.

        Using the top face contour (rather than the full colour mask contour)
        ensures side face pixels do not inflate the bounding box.

        The pinhole projection simplifies to:
            metric_width = pixel_width * depth / focal_length_x

        Inputs:
            top_face_contour: Contour of the top face mask.
            cube_top_depth_m: Depth at the top face centroid in metres.

        Returns:
            float: Estimated half-width in metres.
        """
        fx = self._latest_camera_info.k[0]

        _, _, w_pixels, _ = cv2.boundingRect(top_face_contour)
        width_m = w_pixels * cube_top_depth_m / fx

        return width_m / 2.0
    
    def _detected_cube_to_ros_msg(self, cube: DetectedCube) -> DetectedObject:
        """
        Convert an internal DetectedCube to a DetectedObject ROS message.

        This is the only place in the vision node that constructs a
        DetectedObject message, keeping the internal representation and
        the ROS interface cleanly separated.

        Inputs:
            cube: Internal DetectedCube from the world model.

        Returns:
            DetectedObject: Populated ROS message ready to send to the manager.
        """
        msg = DetectedObject()

        msg.id = cube.id
        msg.confidence = cube.confidence

        msg.pose_camera.x = cube.position_camera_xyz[0]
        msg.pose_camera.y = cube.position_camera_xyz[1]
        msg.pose_camera.z = cube.position_camera_xyz[2]
        msg.pose_camera.roll = 0.0
        msg.pose_camera.pitch = 0.0
        msg.pose_camera.yaw = 0.0

        msg.estimated_half_height_m = cube.estimated_half_height_m
        msg.estimated_half_width_m = cube.estimated_half_width_m

        return msg

    def _handle_detect_objects(self, request, response):
        """
        Handle a detect_objects service request.

        Runs detection to update the internal world model, then converts
        each detected cube to a ROS message for the response.
        """
        self.get_logger().info("Received detect_objects request.")

        if not self._camera_data_ready():
            response.success = False
            response.message = "Camera data is not ready yet."
            response.detections = []

            self.get_logger().warn("Detection requested before camera data was ready.")
            return response

        self._detect_all_cubes()

        response.success = True
        response.message = f"Detected {len(self._detected_cubes)} cube(s)."
        response.detections = [
            self._detected_cube_to_ros_msg(cube)
            for cube in self._detected_cubes.values()
        ]

        for cube in self._detected_cubes.values():
            self.get_logger().info(
                f"Detected {cube.id}: "
                f"x={cube.position_camera_xyz[0]:.3f}, "
                f"y={cube.position_camera_xyz[1]:.3f}, "
                f"z={cube.position_camera_xyz[2]:.3f}, "
                f"half_height={cube.estimated_half_height_m:.4f}m, "
                f"half_width={cube.estimated_half_width_m:.4f}m"
            )

        return response

def main(args=None) -> None:
    rclpy.init(args=args)

    node = ObjectPerception()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()