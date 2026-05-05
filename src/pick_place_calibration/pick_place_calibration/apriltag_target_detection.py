# src/pick_place_calibration/pick_place_calibration/apriltag_target_detection.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse

import cv2
import numpy as np
import yaml

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


@dataclass
class CalibrationImageInput:
    """
    Describe one saved calibration image and its matching camera info file.

    Inputs:
        image_path: Path to one saved RGB calibration image.
        camera_info_path: Path to the matching camera info YAML.

    Returns:
        None
    """

    image_path: Path
    camera_info_path: Path


class AprilTagTargetDetection:
    """
    Detect one configured AprilTag and estimate its pose from saved images.
    """

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
        if tag_family != DEFAULT_TAG_FAMILY:
            raise ValueError(
                f"Only {DEFAULT_TAG_FAMILY} is currently supported, got {tag_family}."
            )

        self._tag_family = tag_family
        self._tag_size_m = tag_size_m
        self._target_tag_id = target_tag_id

        self._aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_APRILTAG_36h11
        )
        self._detector_params = cv2.aruco.DetectorParameters_create()

    def detect_session(
        self,
        session_dir: Path,
        show_debug_windows: bool = False,
        verbose: bool = False,
    ) -> list[AprilTagPoseResult]:
        """
        Run AprilTag detection across one calibration session directory.

        Inputs:
            session_dir: Path to the session directory.
            show_debug_windows: True to show OpenCV debug windows.
            verbose: True to print additional pose diagnostics.

        Returns:
            list[AprilTagPoseResult]: Detection results for all collected inputs.
        """
        session_inputs = self.collect_session_inputs(session_dir)

        if not session_inputs:
            raise ValueError(f"No valid calibration inputs found in: {session_dir}")

        results: list[AprilTagPoseResult] = []

        for session_input in session_inputs:
            print(f"Processing {session_input.image_path.name}...")
            result = self.detect_from_files(
                image_path=session_input.image_path,
                camera_info_path=session_input.camera_info_path,
                show_debug_windows=show_debug_windows,
                verbose=verbose,
            )

            result_output_path = session_input.image_path.with_name(
                f"{session_input.image_path.stem}_detection.yaml"
            )
            self.save_result(result, result_output_path)
            print(f"Saved detection result to {result_output_path}")

            results.append(result)

        return results

    def collect_session_inputs(
        self,
        session_dir: Path,
    ) -> list[CalibrationImageInput]:
        """
        Collect saved calibration RGB images and matching camera info YAML files
        from one session directory.

        Inputs:
            session_dir: Path to the calibration session directory.

        Returns:
            list[CalibrationImageInput]:
                Sorted list of image/camera-info pairs ready for detection.
        """
        if not session_dir.exists():
            raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

        if not session_dir.is_dir():
            raise NotADirectoryError(f"Session path is not a directory: {session_dir}")

        inputs: list[CalibrationImageInput] = []

        for image_path in sorted(session_dir.glob("*.png")):
            image_name = image_path.name

            if image_name.endswith("_binary.png"):
                continue

            if image_name.endswith("_detection_visualization.png"):
                continue

            image_stem = image_path.stem
            base_stem = image_stem[:-4] if image_stem.endswith("_rgb") else image_stem
            camera_info_path = image_path.with_name(f"{base_stem}_camera_info.yaml")

            if not camera_info_path.exists():
                print(
                    f"Skipping {image_path.name} because matching camera info was not found: "
                    f"{camera_info_path.name}"
                )
                continue

            inputs.append(
                CalibrationImageInput(
                    image_path=image_path,
                    camera_info_path=camera_info_path,
                )
            )

        return inputs

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

    def extract_camera_params(
        self,
        camera_info_data: dict,
    ) -> tuple[float, float, float, float]:
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

    def detect_target_tag(
        self,
        grayscale_image: np.ndarray,
    ) -> tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
        """
        Detect AprilTags in a grayscale image.

        Inputs:
            grayscale_image: Grayscale image array.

        Returns:
            tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]]:
                Detected corners, detected ids, and rejected candidates.
        """
        corners, ids, rejected = cv2.aruco.detectMarkers(
            grayscale_image,
            self._aruco_dict,
            parameters=self._detector_params,
        )
        return corners, ids, rejected

    def select_requested_tag(
        self,
        detections: tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]],
    ) -> tuple[np.ndarray, int] | None:
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
                [-half_size, half_size, 0.0],
                [half_size, half_size, 0.0],
                [half_size, -half_size, 0.0],
                [-half_size, -half_size, 0.0],
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

    def compute_detected_tag_side_length_pixels(
        self,
        detected_corners: np.ndarray,
    ) -> float:
        """
        Estimate the detected tag side length in pixels from the four corner points.

        Inputs:
            detected_corners: Detected tag corners from OpenCV.

        Returns:
            float: Mean side length in pixels.
        """
        image_points = detected_corners.reshape(4, 2).astype(np.float64)

        side_lengths = [
            float(np.linalg.norm(image_points[1] - image_points[0])),
            float(np.linalg.norm(image_points[2] - image_points[1])),
            float(np.linalg.norm(image_points[3] - image_points[2])),
            float(np.linalg.norm(image_points[0] - image_points[3])),
        ]

        return float(np.mean(side_lengths))

    def estimate_tag_distance_from_pixel_size(
        self,
        detected_corners: np.ndarray,
        camera_info_data: dict,
    ) -> float:
        """
        Estimate tag distance using a simple pinhole approximation.

        Inputs:
            detected_corners: Detected tag corners from OpenCV.
            camera_info_data: Parsed camera info YAML dictionary.

        Returns:
            float: Approximate tag distance in meters.
        """
        fx, _fy, _cx, _cy = self.extract_camera_params(camera_info_data)
        tag_side_length_pixels = self.compute_detected_tag_side_length_pixels(
            detected_corners
        )

        if tag_side_length_pixels <= 1e-9:
            raise ValueError("Detected tag side length in pixels is too small.")

        return float(fx * self._tag_size_m / tag_side_length_pixels)

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
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not pose_succeeded:
            return None, None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

        translation_m = translation_vector.reshape(3).astype(float).tolist()
        rotation_matrix_list = rotation_matrix.astype(float).tolist()

        return translation_m, rotation_matrix_list

    def compute_reprojection_error_pixels(
        self,
        object_points: np.ndarray,
        image_points: np.ndarray,
        rotation_matrix: list[list[float]],
        translation_m: list[float],
        camera_info_data: dict,
    ) -> tuple[np.ndarray, float]:
        """
        Reproject the solved 3D tag corners into the image and compute pixel error.

        Inputs:
            object_points: 3D tag corner points.
            image_points: Detected 2D corner points.
            rotation_matrix: Solved tag rotation matrix.
            translation_m: Solved tag translation vector.
            camera_info_data: Parsed camera info YAML dictionary.

        Returns:
            tuple[np.ndarray, float]:
                Reprojected 2D image points and mean reprojection error in pixels.
        """
        camera_matrix, distortion_coeffs = self.create_camera_matrix_and_distortion(
            camera_info_data
        )

        rotation_matrix_np = np.array(rotation_matrix, dtype=np.float64)
        translation_vector = np.array(translation_m, dtype=np.float64).reshape(3, 1)
        rotation_vector, _ = cv2.Rodrigues(rotation_matrix_np)

        reprojected_points, _ = cv2.projectPoints(
            object_points.astype(np.float64),
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion_coeffs,
        )

        reprojected_points = reprojected_points.reshape(-1, 2)
        detected_points = image_points.reshape(-1, 2).astype(np.float64)

        point_errors = np.linalg.norm(reprojected_points - detected_points, axis=1)
        mean_error_pixels = float(np.mean(point_errors))

        return reprojected_points, mean_error_pixels

    def detection_to_result(
        self,
        image_path: Path,
        camera_info_path: Path,
        camera_info_data: dict,
        detection: tuple[np.ndarray, int],
        verbose: bool,
    ) -> AprilTagPoseResult:
        """
        Convert one OpenCV AprilTag detection into a serializable detection result.

        Inputs:
            image_path: Path to the RGB image used for detection.
            camera_info_path: Path to the camera info YAML used for intrinsics.
            camera_info_data: Parsed camera info YAML data.
            detection: Selected OpenCV detection tuple containing corners and index.
            verbose: True to print additional pose diagnostics.

        Returns:
            AprilTagPoseResult: Serialized detection result.
        """
        detected_corners, _detection_index = detection
        translation_m, rotation_matrix = self.estimate_pose_from_detection(
            detected_corners,
            camera_info_data,
        )

        if translation_m is None or rotation_matrix is None:
            return self.create_not_found_result(
                image_path,
                camera_info_path,
                camera_info_data,
            )

        if verbose:
            rough_distance_m = self.estimate_tag_distance_from_pixel_size(
                detected_corners,
                camera_info_data,
            )
            object_points = self.create_tag_object_points()
            reprojected_points, mean_reprojection_error_pixels = (
                self.compute_reprojection_error_pixels(
                    object_points=object_points,
                    image_points=detected_corners,
                    rotation_matrix=rotation_matrix,
                    translation_m=translation_m,
                    camera_info_data=camera_info_data,
                )
            )

            fx, fy, cx, cy = self.extract_camera_params(camera_info_data)
            print(f"fx={fx}, fy={fy}, cx={cx}, cy={cy}")
            print(f"d={camera_info_data.get('d', [])}")
            print(f"Approximate pinhole distance estimate: {rough_distance_m:.6f} m")
            print(f"PnP translation estimate: {translation_m}")
            print(f"Mean reprojection error: {mean_reprojection_error_pixels:.6f} px")
            print(f"Detected image points: {detected_corners.reshape(-1, 2).tolist()}")
            print(f"Reprojected image points: {reprojected_points.tolist()}")

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

    def save_detection_visualization(
        self,
        image_path: Path,
        detections: tuple[list[np.ndarray], np.ndarray | None, list[np.ndarray]],
        selected_detection: tuple[np.ndarray, int] | None,
        output_path: Path,
    ) -> None:
        """
        Save a visualization image showing the detected AprilTag marker, corner
        locations, corner indices, and board frame axes.

        Inputs:
            image_path: Path to the source RGB image.
            detections: Tuple containing detected corners, ids, and rejected candidates.
            selected_detection: Selected OpenCV detection tuple containing corners and index.
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
        show_debug_windows: bool = False,
        verbose: bool = False,
    ) -> AprilTagPoseResult:
        """
        Detect the configured AprilTag from one saved calibration image.

        Inputs:
            image_path: Path to the saved RGB image.
            camera_info_path: Path to the saved camera info YAML.
            show_debug_windows: True to show OpenCV debug windows.
            verbose: True to print additional pose diagnostics.

        Returns:
            AprilTagPoseResult: Detection and pose estimation result.
        """
        camera_info_data = self.load_camera_info(camera_info_path)
        grayscale_image = self.load_grayscale_image(image_path)

        if show_debug_windows:
            cv2.imshow("AprilTag Grayscale Debug", grayscale_image)
            cv2.waitKey(2000)
            cv2.destroyAllWindows()

        detections = self.detect_target_tag(grayscale_image)
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
            output_path=visualization_output_path,
        )
        print(f"Saved detection visualization to {visualization_output_path}")

        return self.detection_to_result(
            image_path=image_path,
            camera_info_path=camera_info_path,
            camera_info_data=camera_info_data,
            detection=detection,
            verbose=verbose,
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for one-image or batch AprilTag detection.

    Inputs:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Detect AprilTag pose from one saved calibration image or a full session."
    )

    parser.add_argument(
        "--image",
        required=False,
        help="Path to one saved RGB image.",
    )
    parser.add_argument(
        "--camera-info",
        required=False,
        help="Path to the matching saved camera info YAML.",
    )
    parser.add_argument(
        "--session-dir",
        required=False,
        help="Path to a calibration session directory for batch detection.",
    )
    parser.add_argument(
        "--output",
        required=False,
        help="Optional output YAML path for one-image detection.",
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
    parser.add_argument(
        "--show-debug-windows",
        action="store_true",
        help="Show OpenCV debug windows during detection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional pose diagnostics.",
    )

    args = parser.parse_args()

    single_image_mode = args.image is not None or args.camera_info is not None
    session_mode = args.session_dir is not None

    if session_mode and single_image_mode:
        parser.error("Use either --session-dir or --image/--camera-info, not both.")

    if not session_mode and not single_image_mode:
        parser.error("Provide either --session-dir or both --image and --camera-info.")

    if single_image_mode and (args.image is None or args.camera_info is None):
        parser.error("--image and --camera-info must be provided together.")

    return args


def main() -> None:
    """
    Run one-image or batch AprilTag detection and save results.

    Inputs:
        None

    Returns:
        None
    """
    args = parse_args()

    detector = AprilTagTargetDetection(
        tag_family=args.tag_family,
        tag_size_m=args.tag_size_m,
        target_tag_id=args.tag_id,
    )

    if args.session_dir is not None:
        session_dir = Path(args.session_dir)

        results = detector.detect_session(
            session_dir=session_dir,
            show_debug_windows=args.show_debug_windows,
            verbose=args.verbose,
        )

        detected_count = sum(1 for result in results if result.detected)
        print(
            f"Batch detection complete: {detected_count}/{len(results)} images detected successfully."
        )
        return

    image_path = Path(args.image)
    camera_info_path = Path(args.camera_info)

    result = detector.detect_from_files(
        image_path=image_path,
        camera_info_path=camera_info_path,
        show_debug_windows=args.show_debug_windows,
        verbose=args.verbose,
    )

    output_data = detector.result_to_dict(result)
    print(yaml.safe_dump(output_data, sort_keys=False))

    if args.output:
        detector.save_result(result, Path(args.output))
        print(f"Saved detection result to {args.output}")


if __name__ == "__main__":
    main()
