from pathlib import Path
from dataclasses import dataclass
import yaml
import cv2
import numpy as np
import argparse

DEFAULT_TAG_ID = 0
DEFAULT_TAG_FAMILY = "tag36h11"
DEFAULT_TAG_SIZE_M = 0.21118

@dataclass
class AprilTagPoseResult:
    """
    Describe the result of one AprilTag detection and pose estimation step.

    Inputs:
        image_path: Path to the RGB image used for detection.
        camera_info_path: Path to the camera info YAML used for intrinsics.
        detected: True if the requested tag was detected.
        tag_id: Detected AprilTag id if available.
        tag_family: Detected AprilTag family if available.
        decision_margin: Detector confidence-style score if available.
        camera_frame_id: Camera frame id loaded from camera info if available.
        translation_m: Estimated camera-to-tag translation in meters.
        rotation_matrix: Estimated camera-to-tag rotation matrix.
        tag_size_m: Physical tag size used for pose estimation.

    Returns:
        None
    """
    image_path: str
    camera_info_path: str
    detected: bool
    tag_id: int | None
    tag_family: str | None
    decision_margin: float | None
    camera_frame_id: str | None
    translation_m: list[float] | None
    rotation_matrix: list[list[float]] | None
    tag_size_m: float

class AprilTagTargetDetection:
    def __init__(
        self,
        tag_family: str = DEFAULT_TAG_FAMILY,
        tag_size_m: float = DEFAULT_TAG_SIZE_M,
        target_tag_id: int = DEFAULT_TAG_ID,
    ) -> None:
        """
        Create a reusable AprilTag detector helper for one-image and batch
        calibration target detection.

        Inputs:
            tag_family: AprilTag family expected in the dataset.
            tag_size_m: Physical tag edge length in meters.
            target_tag_id: The specific tag id to detect and estimate pose for.

        Returns:
            None
        """
        self._tag_family = tag_family
        self._tag_size_m = tag_size_m
        self._target_tag_id = target_tag_id

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        self._detector_params = cv2.aruco.DetectorParameters_create()

    def load_camera_info(self, camera_info_path: Path) -> dict:
        """
        Load camera intrinsics from a saved camera info YAML file.

        Inputs:
            camera_info_path: Path to the camera info YAML file.

        Returns:
            dict: Parsed YAML camera info data.
        """
        with open(camera_info_path, "r", encoding="utf-8") as camera_info_file:
            return yaml.safe_load(camera_info_file)

    def extract_camera_params(self, camera_info_data: dict) -> tuple[float, float, float, float]:
        """
        Extract pinhole intrinsics from saved camera info data.

        Inputs:
            camera_info_data: Parsed camera info YAML dictionary.

        Returns:
            tuple[float, float, float, float]:
                Camera parameters as (fx, fy, cx, cy).
        """
        k = camera_info_data["k"]

        fx = float(k[0])
        fy = float(k[4])
        cx = float(k[2])
        cy = float(k[5])

        return fx, fy, cx, cy
    
    def load_grayscale_image(self, image_path: Path) -> np.ndarray:
        """
        Load an image from disk and convert it to grayscale for AprilTag detection.

        Inputs:
            image_path: Path to the RGB image file.

        Returns:
            np.ndarray: Grayscale image array.
        """
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        return image
    
    def isolate_black_regions(
        self,
        grayscale_image: np.ndarray,
        black_threshold: int = 80,
    ) -> np.ndarray:
        """
        Convert a grayscale image so that dark pixels remain black and all other
        pixels become white.

        Inputs:
            grayscale_image: Grayscale image array.
            black_threshold: Maximum grayscale value still treated as black.

        Returns:
            np.ndarray: Black-and-white image.
        """
        _threshold_value, binary_image = cv2.threshold(
            grayscale_image,
            black_threshold,
            255,
            cv2.THRESH_BINARY,
        )

        return binary_image

    def binarize_grayscale_image(
        self,
        grayscale_image: np.ndarray,
    ) -> np.ndarray:
        """
        Binarize a grayscale image using adaptive thresholding.

        Inputs:
            grayscale_image: Grayscale image array.

        Returns:
            np.ndarray: Binarized image array.
        """
        return cv2.adaptiveThreshold(
            grayscale_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
    
    def detect_target_tag(
        self,
        grayscale_image: np.ndarray,
        camera_params: tuple[float, float, float, float],
    ):
        """
        Detect AprilTags in a grayscale image.

        Inputs:
            grayscale_image: Grayscale image array.
            camera_params: Camera intrinsics as (fx, fy, cx, cy).

        Returns:
            tuple[list, np.ndarray | None, list]:
                Detected corners, detected ids, and rejected candidates.
        """
        corners, ids, rejected = cv2.aruco.detectMarkers(
            grayscale_image,
            self._aruco_dict,
            parameters=self._detector_params,
        )

        print(f"Detected marker ids: {ids}")
        print(f"Number of detected markers: {len(corners)}")
        print(f"Number of rejected candidates: {len(rejected)}")

        return corners, ids, rejected

    def select_requested_tag(self, detections):
        """
        Select the configured target tag id from OpenCV AprilTag detections.

        Inputs:
            detections: Tuple containing detected corners, ids, and rejected candidates.

        Returns:
            tuple[np.ndarray, int] | None:
                Matching detected corners and index if found, otherwise None.
        """
        corners, ids, _rejected = detections

        if ids is None:
            return None

        for index, detected_id in enumerate(ids.flatten()):
            if int(detected_id) == self._target_tag_id:
                return corners[index], index

        return None

    def draw_board_xy_axes(
        self,
        color_image: np.ndarray,
        detected_corners: np.ndarray,
    ) -> None:
        """
        Draw the board-local X and Y axes directly in the image plane using the
        detected outer board corners.

        Inputs:
            color_image: Color image on which to draw.
            detected_corners: Detected OpenCV marker corners.

        Returns:
            None
        """
        image_points = detected_corners.reshape(4, 2).astype(np.float32)

        corner_0 = image_points[0]
        corner_1 = image_points[1]
        corner_2 = image_points[2]
        corner_3 = image_points[3]

        board_center = np.mean(image_points, axis=0)

        left_midpoint = 0.5 * (corner_0 + corner_3)
        right_midpoint = 0.5 * (corner_1 + corner_2)
        top_midpoint = 0.5 * (corner_0 + corner_1)
        bottom_midpoint = 0.5 * (corner_2 + corner_3)

        x_direction = right_midpoint - left_midpoint
        y_direction = bottom_midpoint - top_midpoint

        axis_scale = 0.5
        x_tip = board_center + axis_scale * x_direction
        y_tip = board_center + axis_scale * y_direction

        center_point = (
            int(round(board_center[0])),
            int(round(board_center[1])),
        )
        x_tip_point = (
            int(round(x_tip[0])),
            int(round(x_tip[1])),
        )
        y_tip_point = (
            int(round(y_tip[0])),
            int(round(y_tip[1])),
        )

        cv2.circle(color_image, center_point, 6, (255, 255, 0), -1)

        cv2.arrowedLine(
            color_image,
            center_point,
            x_tip_point,
            (0, 0, 255),
            2,
            tipLength=0.15,
        )
        cv2.arrowedLine(
            color_image,
            center_point,
            y_tip_point,
            (0, 255, 0),
            2,
            tipLength=0.15,
        )

        cv2.putText(
            color_image,
            "X",
            (x_tip_point[0] + 8, x_tip_point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            color_image,
            "Y",
            (y_tip_point[0] + 8, y_tip_point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    
    def detection_to_result(
        self,
        image_path: Path,
        camera_info_path: Path,
        camera_info_data: dict,
        detection,
    ) -> AprilTagPoseResult:
        """
        Convert one OpenCV AprilTag detection into a serializable detection result.

        Inputs:
            image_path: Path to the RGB image used for detection.
            camera_info_path: Path to the camera info YAML used for intrinsics.
            camera_info_data: Parsed camera info YAML data.
            detection: Selected OpenCV detection tuple containing corners and index.

        Returns:
            AprilTagPoseResult: Serialized detection result.
        """
        detected_corners, _detection_index = detection

        translation_m, rotation_matrix = self.estimate_pose_from_detection(
            detected_corners,
            camera_info_data,
        )

        return AprilTagPoseResult(
            image_path=str(image_path),
            camera_info_path=str(camera_info_path),
            detected=True,
            tag_id=int(self._target_tag_id),
            tag_family=str(self._tag_family),
            decision_margin=None,
            camera_frame_id=str(camera_info_data["header"]["frame_id"]),
            translation_m=translation_m,
            rotation_matrix=rotation_matrix,
            tag_size_m=float(self._tag_size_m),
        )

    def create_tag_object_points(self) -> np.ndarray:
        """
        Create the 3D object points for the four AprilTag corners in the tag frame.

        Inputs:
            None

        Returns:
            np.ndarray: Tag corner coordinates as a 4x3 float array.
        """
        half_size = float(self._tag_size_m) * 0.5

        return np.array(
            [
                [-half_size, -half_size, 0.0],
                [ half_size, -half_size, 0.0],
                [ half_size,  half_size, 0.0],
                [-half_size,  half_size, 0.0],
            ],
            dtype=np.float32,
        )

    def create_camera_matrix_and_distortion(
        self,
        camera_info_data: dict,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create the camera matrix and distortion vector for OpenCV pose estimation.

        Inputs:
            camera_info_data: Parsed camera info YAML dictionary.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                Camera matrix and distortion vector.
        """
        fx, fy, cx, cy = self.extract_camera_params(camera_info_data)

        camera_matrix = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        distortion_coeffs = np.array(
            camera_info_data.get("d", []),
            dtype=np.float64,
        ).reshape(-1, 1)

        return camera_matrix, distortion_coeffs

    def estimate_pose_from_detection(
        self,
        detected_corners: np.ndarray,
        camera_info_data: dict,
    ) -> tuple[list[float] | None, list[list[float]] | None]:
        """
        Estimate the camera-to-tag pose from detected AprilTag image corners.

        Inputs:
            detected_corners: Detected tag corners from OpenCV.
            camera_info_data: Parsed camera info YAML dictionary.

        Returns:
            tuple[list[float] | None, list[list[float]] | None]:
                Translation vector and rotation matrix if pose estimation succeeds,
                otherwise (None, None).
        """
        object_points = self.create_tag_object_points()
        camera_matrix, distortion_coeffs = self.create_camera_matrix_and_distortion(
            camera_info_data
        )

        image_points = detected_corners.reshape(4, 2).astype(np.float32)

        pose_succeeded, rotation_vector, translation_vector = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )

        if not pose_succeeded:
            return None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        translation_m = translation_vector.reshape(3).astype(float).tolist()
        rotation_matrix_list = rotation_matrix.astype(float).tolist()

        return translation_m, rotation_matrix_list

    def save_detection_visualization(
        self,
        image_path: Path,
        detections,
        selected_detection,
        camera_info_data: dict,
        output_path: Path,
    ) -> None:
        """
        Save a visualization image showing the detected AprilTag marker, corner
        locations, corner indices, and board frame axes.

        Inputs:
            image_path: Path to the source RGB image.
            detections: Tuple containing detected corners, ids, and rejected candidates.
            selected_detection: Selected OpenCV detection tuple containing corners and index.
            camera_info_data: Parsed camera info YAML dictionary.
            output_path: Output path for the visualization image.

        Returns:
            None
        """
        color_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if color_image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        corners, ids, _rejected = detections

        if ids is not None and len(corners) > 0:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)

        if selected_detection is not None:
            detected_corners, _detection_index = selected_detection
            image_points = detected_corners.reshape(4, 2).astype(np.float32)

            for corner_index, point in enumerate(image_points):
                x_coord = int(round(point[0]))
                y_coord = int(round(point[1]))

                cv2.circle(color_image, (x_coord, y_coord), 20, (0, 0, 255), -1)
                cv2.putText(
                    color_image,
                    str(corner_index),
                    (x_coord + 8, y_coord - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (255, 0, 0),
                    10,
                    cv2.LINE_AA,
                )

            self.draw_board_xy_axes(
                color_image=color_image,
                detected_corners=detected_corners,
            )

        write_succeeded = cv2.imwrite(str(output_path), color_image)

        if not write_succeeded:
            raise RuntimeError(f"Failed to save detection visualization: {output_path}")
    
    def create_not_found_result(
        self,
        image_path: Path,
        camera_info_path: Path,
        camera_info_data: dict,
    ) -> AprilTagPoseResult:
        """
        Create a result record for the case where the requested tag is not detected.

        Inputs:
            image_path: Path to the RGB image used for detection.
            camera_info_path: Path to the camera info YAML used for intrinsics.
            camera_info_data: Parsed camera info YAML data.

        Returns:
            AprilTagPoseResult: Serialized non-detection result.
        """
        return AprilTagPoseResult(
            image_path=str(image_path),
            camera_info_path=str(camera_info_path),
            detected=False,
            tag_id=None,
            tag_family=None,
            decision_margin=None,
            camera_frame_id=str(camera_info_data["header"]["frame_id"]),
            translation_m=None,
            rotation_matrix=None,
            tag_size_m=float(self._tag_size_m),
        )

    def result_to_dict(self, result: AprilTagPoseResult) -> dict:
        """
        Convert an AprilTag pose result into a YAML-friendly dictionary.

        Inputs:
            result: AprilTag pose result dataclass.

        Returns:
            dict: Serialized detection result dictionary.
        """
        return {
            "image_path": result.image_path,
            "camera_info_path": result.camera_info_path,
            "detected": bool(result.detected),
            "tag_id": result.tag_id,
            "tag_family": result.tag_family,
            "decision_margin": result.decision_margin,
            "camera_frame_id": result.camera_frame_id,
            "translation_m": result.translation_m,
            "rotation_matrix": result.rotation_matrix,
            "tag_size_m": float(result.tag_size_m),
        }
    
    def save_result(self, result: AprilTagPoseResult, output_path: Path) -> None:
        """
        Save one AprilTag pose result to YAML.

        Inputs:
            result: Detection result to save.
            output_path: Output YAML path.

        Returns:
            None
        """
        output_data = self.result_to_dict(result)

        with open(output_path, "w", encoding="utf-8") as output_file:
            yaml.safe_dump(output_data, output_file, sort_keys=False)

    def detect_from_files(
        self,
        image_path: Path,
        camera_info_path: Path,
        ) -> AprilTagPoseResult:
        """
        Detect the configured AprilTag from one saved calibration image.

        Inputs:
            image_path: Path to the saved RGB image.
            camera_info_path: Path to the saved camera info YAML.

        Returns:
            AprilTagPoseResult: Detection and pose estimation result.
        """
        camera_info_data = self.load_camera_info(camera_info_path)
        camera_params = self.extract_camera_params(camera_info_data)
        grayscale_image = self.load_grayscale_image(image_path)
        binary_image = self.isolate_black_regions(
            grayscale_image,
            black_threshold=80,
        )

        binary_output_path = image_path.with_name(
            f"{image_path.stem}_binary.png"
        )
        write_succeeded = cv2.imwrite(str(binary_output_path), binary_image)

        if not write_succeeded:
            raise RuntimeError(f"Failed to save binary image: {binary_output_path}")

        print(f"Saved binary image to {binary_output_path}")

        cv2.imshow("AprilTag Binary Debug", binary_image)
        cv2.waitKey(2000)
        cv2.destroyAllWindows()

        detections = self.detect_target_tag(
            binary_image,
            camera_params,
        )

        detection = self.select_requested_tag(detections)

        if detection is None:
            return self.create_not_found_result(
                image_path,
                camera_info_path,
                camera_info_data,
            )

        visualization_output_path = image_path.with_name(
            f"{image_path.stem}_detection_visualization.png"
        )
        self.save_detection_visualization(
            image_path=image_path,
            detections=detections,
            selected_detection=detection,
            camera_info_data=camera_info_data,
            output_path=visualization_output_path,
        )
        print(f"Saved detection visualization to {visualization_output_path}")

        return self.detection_to_result(
            image_path,
            camera_info_path,
            camera_info_data,
            detection,
        )
    
def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for one-image AprilTag detection.

    Inputs:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect AprilTag pose from one saved calibration image."
    )
    parser.add_argument("--image", required=True, help="Path to the saved RGB image.")
    parser.add_argument(
        "--camera-info",
        required=True,
        help="Path to the saved camera info YAML.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output YAML path for the detection result.",
    )
    parser.add_argument(
        "--tag-id",
        type=int,
        default=DEFAULT_TAG_ID,
        help="Target AprilTag id to detect.",
    )
    parser.add_argument(
        "--tag-size-m",
        type=float,
        default=DEFAULT_TAG_SIZE_M,
        help="Physical tag edge length in meters.",
    )
    parser.add_argument(
        "--tag-family",
        default=DEFAULT_TAG_FAMILY,
        help="AprilTag family string.",
    )
    return parser.parse_args()
    
def main() -> None:
    """
    Run one-image AprilTag detection and optionally save the result to YAML.

    Inputs:
        None

    Returns:
        None
    """
    args = parse_args()

    image_path = Path(args.image)
    camera_info_path = Path(args.camera_info)

    detector = AprilTagTargetDetection(
        tag_family=args.tag_family,
        tag_size_m=args.tag_size_m,
        target_tag_id=args.tag_id,
    )

    result = detector.detect_from_files(
        image_path=image_path,
        camera_info_path=camera_info_path,
    )

    output_data = detector.result_to_dict(result)

    print(yaml.safe_dump(output_data, sort_keys=False))

    if args.output:
        detector.save_result(result, Path(args.output))
        print(f"Saved detection result to {args.output}")


if __name__ == "__main__":
    main()