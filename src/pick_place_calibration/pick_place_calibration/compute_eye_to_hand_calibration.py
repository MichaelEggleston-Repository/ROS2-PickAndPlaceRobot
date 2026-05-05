# src/pick_place_calibration/pick_place_calibration/compute_eye_to_hand_calibration.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math

import numpy as np
import yaml

DEFAULT_TOOL_TO_TAG_TRANSLATION_M = [0.0, 0.0, 0.013]
DEFAULT_TOOL_TO_TAG_RPY_RADIANS = [0.0, 0.0, math.pi]
DEFAULT_CAMERA_LINK_TO_SENSOR_RPY_RADIANS = [-math.pi / 2.0, 0.0, -math.pi / 2.0]


@dataclass
class CalibrationSamplePaths:
    """
    Describe one calibration sample file set.

    Inputs:
        image_stem: Base sample stem such as image_01.
        metadata_path: Path to the saved metadata YAML.
        detection_path: Path to the saved detection YAML.

    Returns:
        None
    """

    image_stem: str
    metadata_path: Path
    detection_path: Path


@dataclass
class CalibrationSampleData:
    """
    Describe one fully loaded calibration sample.

    Inputs:
        image_stem: Base sample stem such as image_01.
        base_T_tool: Homogeneous transform from robot base to tool.
        tool_T_tag: Homogeneous transform from tool to calibration tag.
        camera_T_tag: Homogeneous transform from camera sensor frame to calibration tag.
        base_to_tool_translation_m: Recorded base-to-tool translation.
        camera_to_tag_translation_m: Detected camera-to-tag translation.
        requested_vs_actual_position_error_m: Requested-vs-actual TCP position error if available.

    Returns:
        None
    """

    image_stem: str
    base_T_tool: np.ndarray
    tool_T_tag: np.ndarray
    camera_T_tag: np.ndarray
    base_to_tool_translation_m: list[float]
    camera_to_tag_translation_m: list[float]
    requested_vs_actual_position_error_m: float | None


@dataclass
class CameraTransformEstimate:
    """
    Describe one computed base-to-camera estimate from a calibration sample.

    Inputs:
        image_stem: Base sample stem such as image_01.
        base_T_camera: Homogeneous transform from robot base to camera sensor frame.
        translation_m: Base-to-camera translation as [x, y, z].
        rpy_radians: Base-to-camera roll, pitch, yaw in radians.

    Returns:
        None
    """

    image_stem: str
    base_T_camera: np.ndarray
    translation_m: list[float]
    rpy_radians: list[float]

@dataclass
class EyeToHandCalibrationResult:
    """
    Describe the final computed eye-to-hand calibration result.

    Inputs:
        sample_count: Number of valid samples used.
        mean_translation_m: Mean base-to-camera translation.
        mean_rpy_radians: Mean base-to-camera roll, pitch, yaw.
        mean_base_T_camera: Mean homogeneous transform.

    Returns:
        None
    """
    sample_count: int
    mean_translation_m: list[float]
    mean_rpy_radians: list[float]
    mean_base_T_camera: np.ndarray


def load_yaml_file(path: Path) -> dict:
    """
    Load one YAML file.

    Inputs:
        path: YAML file path.

    Returns:
        dict: Parsed YAML data.
    """
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def collect_calibration_sample_paths(
    session_dir: Path,
) -> list[CalibrationSamplePaths]:
    """
    Collect matching metadata and detection YAML files from one session directory.

    Inputs:
        session_dir: Path to the calibration session directory.

    Returns:
        list[CalibrationSamplePaths]:
            Sorted list of calibration sample file sets.
    """
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory does not exist: {session_dir}")

    if not session_dir.is_dir():
        raise NotADirectoryError(f"Session path is not a directory: {session_dir}")

    sample_paths: list[CalibrationSamplePaths] = []

    for metadata_path in sorted(session_dir.glob("*_metadata.yaml")):
        image_stem = metadata_path.stem.removesuffix("_metadata")
        detection_path = session_dir / f"{image_stem}_rgb_detection.yaml"

        if not detection_path.exists():
            print(
                f"Skipping {metadata_path.name} because matching detection file "
                f"was not found: {detection_path.name}"
            )
            continue

        sample_paths.append(
            CalibrationSamplePaths(
                image_stem=image_stem,
                metadata_path=metadata_path,
                detection_path=detection_path,
            )
        )

    return sample_paths


def parse_translation_dict(translation_dict: dict) -> list[float]:
    """
    Convert a translation dictionary with x, y, z keys to a list.

    Inputs:
        translation_dict: Dictionary containing x, y, z values.

    Returns:
        list[float]: Translation as [x, y, z].
    """
    return [
        float(translation_dict["x"]),
        float(translation_dict["y"]),
        float(translation_dict["z"]),
    ]


def parse_quaternion_dict_xyzw(quaternion_dict: dict) -> list[float]:
    """
    Convert a quaternion dictionary with x, y, z, w keys to a list.

    Inputs:
        quaternion_dict: Dictionary containing x, y, z, w values.

    Returns:
        list[float]: Quaternion as [x, y, z, w].
    """
    return [
        float(quaternion_dict["x"]),
        float(quaternion_dict["y"]),
        float(quaternion_dict["z"]),
        float(quaternion_dict["w"]),
    ]


def normalize_quaternion_xyzw(
    quaternion_xyzw: list[float],
) -> np.ndarray:
    """
    Normalize a quaternion in XYZW order.

    Inputs:
        quaternion_xyzw: Quaternion as [x, y, z, w].

    Returns:
        np.ndarray: Normalized quaternion.
    """
    quaternion = np.array(quaternion_xyzw, dtype=np.float64)
    norm = np.linalg.norm(quaternion)

    if norm <= 1e-12:
        raise ValueError("Quaternion norm is too small to normalize.")

    return quaternion / norm


def quaternion_xyzw_to_rotation_matrix(
    quaternion_xyzw: list[float],
) -> np.ndarray:
    """
    Convert a quaternion in XYZW order to a 3x3 rotation matrix.

    Inputs:
        quaternion_xyzw: Quaternion as [x, y, z, w].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    x, y, z, w = normalize_quaternion_xyzw(quaternion_xyzw)

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def rpy_radians_to_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
) -> np.ndarray:
    """
    Convert roll, pitch, yaw in radians to a 3x3 rotation matrix.

    Inputs:
        roll: Rotation about X axis in radians.
        pitch: Rotation about Y axis in radians.
        yaw: Rotation about Z axis in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    cr = math.cos(roll)
    sr = math.sin(roll)
    cp = math.cos(pitch)
    sp = math.sin(pitch)
    cy = math.cos(yaw)
    sy = math.sin(yaw)

    rotation_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    rotation_y = np.array(
        [
            [cp, 0.0, sp],
            [0.0, 1.0, 0.0],
            [-sp, 0.0, cp],
        ],
        dtype=np.float64,
    )

    rotation_z = np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return rotation_z @ rotation_y @ rotation_x


def make_transform_matrix(
    translation_m: list[float],
    rotation_matrix: np.ndarray,
) -> np.ndarray:
    """
    Create a 4x4 homogeneous transform matrix.

    Inputs:
        translation_m: Translation as [x, y, z].
        rotation_matrix: 3x3 rotation matrix.

    Returns:
        np.ndarray: 4x4 transform matrix.
    """
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = np.array(translation_m, dtype=np.float64)
    return transform


def invert_transform_matrix(transform: np.ndarray) -> np.ndarray:
    """
    Invert a 4x4 rigid transform matrix.

    Inputs:
        transform: 4x4 homogeneous transform.

    Returns:
        np.ndarray: Inverted 4x4 transform.
    """
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    inverted = np.eye(4, dtype=np.float64)
    inverted[:3, :3] = rotation.T
    inverted[:3, 3] = -rotation.T @ translation
    return inverted


def rotation_matrix_to_rpy_radians(
    rotation: np.ndarray,
) -> tuple[float, float, float]:
    """
    Convert a rotation matrix to roll, pitch, yaw in radians.

    Inputs:
        rotation: 3x3 rotation matrix.

    Returns:
        tuple[float, float, float]: Roll, pitch, yaw.
    """
    sy = math.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-9

    if not singular:
        roll = math.atan2(rotation[2, 1], rotation[2, 2])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = math.atan2(-rotation[1, 2], rotation[1, 1])
        pitch = math.atan2(-rotation[2, 0], sy)
        yaw = 0.0

    return roll, pitch, yaw


def apply_camera_frame_rotation(
    base_T_camera_link: np.ndarray,
    camera_link_to_sensor_rotation: np.ndarray,
) -> np.ndarray:
    """
    Apply the fixed camera-link to camera-sensor rotation.

    Inputs:
        base_T_camera_link: Base-to-camera-link transform.
        camera_link_to_sensor_rotation: Fixed rotation from camera link to sensor frame.

    Returns:
        np.ndarray: Base-to-camera-sensor transform.
    """
    camera_frame_offset = np.eye(4, dtype=np.float64)
    camera_frame_offset[:3, :3] = camera_link_to_sensor_rotation
    return base_T_camera_link @ camera_frame_offset


def extract_requested_vs_actual_position_error_m(
    metadata: dict,
) -> float | None:
    """
    Extract the requested-vs-actual TCP position error from metadata if present.

    Inputs:
        metadata: Parsed metadata YAML dictionary.

    Returns:
        float | None: Position error in meters if present.
    """
    if "requested_vs_actual_position_error_m" in metadata:
        return float(metadata["requested_vs_actual_position_error_m"])

    if "position_error_m" in metadata:
        return float(metadata["position_error_m"])

    return None


def load_calibration_sample(
    sample_paths: CalibrationSamplePaths,
    tool_to_tag_translation_m: list[float],
    tool_to_tag_rpy_radians: list[float],
) -> CalibrationSampleData | None:
    """
    Load one calibration sample from its metadata and detection YAML files.

    Inputs:
        sample_paths: Metadata/detection file paths for one sample.
        tool_to_tag_translation_m: Tool-to-tag translation used in the solve.
        tool_to_tag_rpy_radians: Tool-to-tag roll, pitch, yaw used in the solve.

    Returns:
        CalibrationSampleData | None:
            Parsed sample if valid, otherwise None.
    """
    metadata = load_yaml_file(sample_paths.metadata_path)
    detection = load_yaml_file(sample_paths.detection_path)

    if not bool(metadata.get("motion_success", False)):
        print(f"Skipping {sample_paths.image_stem} because motion_success was false.")
        return None

    if not bool(metadata.get("image_capture_success", False)):
        print(
            f"Skipping {sample_paths.image_stem} because image_capture_success was false."
        )
        return None

    if not bool(detection.get("detected", False)):
        print(f"Skipping {sample_paths.image_stem} because tag detection failed.")
        return None

    base_to_tool = metadata["base_to_tool"]
    base_to_tool_translation_m = parse_translation_dict(base_to_tool["translation_m"])

    base_T_tool = make_transform_matrix(
        translation_m=base_to_tool_translation_m,
        rotation_matrix=quaternion_xyzw_to_rotation_matrix(
            parse_quaternion_dict_xyzw(base_to_tool["quaternion_xyzw"])
        ),
    )

    tool_T_tag = make_transform_matrix(
        translation_m=tool_to_tag_translation_m,
        rotation_matrix=rpy_radians_to_rotation_matrix(
            tool_to_tag_rpy_radians[0],
            tool_to_tag_rpy_radians[1],
            tool_to_tag_rpy_radians[2],
        ),
    )

    camera_to_tag_translation_m = [float(value) for value in detection["translation_m"]]
    camera_T_tag = make_transform_matrix(
        translation_m=camera_to_tag_translation_m,
        rotation_matrix=np.array(detection["rotation_matrix"], dtype=np.float64),
    )

    requested_vs_actual_position_error_m = extract_requested_vs_actual_position_error_m(
        metadata
    )

    return CalibrationSampleData(
        image_stem=sample_paths.image_stem,
        base_T_tool=base_T_tool,
        tool_T_tag=tool_T_tag,
        camera_T_tag=camera_T_tag,
        base_to_tool_translation_m=base_to_tool_translation_m,
        camera_to_tag_translation_m=camera_to_tag_translation_m,
        requested_vs_actual_position_error_m=requested_vs_actual_position_error_m,
    )


def compute_base_to_camera_transform(
    sample: CalibrationSampleData,
) -> CameraTransformEstimate:
    """
    Compute one base-to-camera estimate from a calibration sample.

    Inputs:
        sample: Loaded calibration sample.

    Returns:
        CameraTransformEstimate: Computed base-to-camera estimate.
    """
    base_T_camera = (
        sample.base_T_tool
        @ sample.tool_T_tag
        @ invert_transform_matrix(sample.camera_T_tag)
    )

    translation_m = base_T_camera[:3, 3].astype(float).tolist()
    roll, pitch, yaw = rotation_matrix_to_rpy_radians(base_T_camera[:3, :3])

    return CameraTransformEstimate(
        image_stem=sample.image_stem,
        base_T_camera=base_T_camera,
        translation_m=translation_m,
        rpy_radians=[float(roll), float(pitch), float(yaw)],
    )


def compute_session_estimates(
    session_dir: Path,
    tool_to_tag_translation_m: list[float],
    tool_to_tag_rpy_radians: list[float],
) -> list[CameraTransformEstimate]:
    """
    Compute base-to-camera estimates for all valid samples in one session.

    Inputs:
        session_dir: Path to the calibration session directory.
        tool_to_tag_translation_m: Tool-to-tag translation used in the solve.
        tool_to_tag_rpy_radians: Tool-to-tag roll, pitch, yaw used in the solve.

    Returns:
        list[CameraTransformEstimate]: Per-sample camera transform estimates.
    """
    sample_paths_list = collect_calibration_sample_paths(session_dir)

    if not sample_paths_list:
        raise ValueError(f"No calibration sample file pairs found in: {session_dir}")

    estimates: list[CameraTransformEstimate] = []

    for sample_paths in sample_paths_list:
        sample = load_calibration_sample(
            sample_paths,
            tool_to_tag_translation_m=tool_to_tag_translation_m,
            tool_to_tag_rpy_radians=tool_to_tag_rpy_radians,
        )

        if sample is None:
            continue

        estimate = compute_base_to_camera_transform(sample)
        estimates.append(estimate)

    if not estimates:
        raise ValueError(
            f"No valid calibration estimates could be computed from: {session_dir}"
        )

    return estimates


def compute_mean_translation(
    estimates: list[CameraTransformEstimate],
) -> np.ndarray:
    """
    Compute the mean base-to-camera translation across all estimates.

    Inputs:
        estimates: Per-sample camera transform estimates.

    Returns:
        np.ndarray: Mean translation as a 3-element vector.
    """
    translations = np.array(
        [estimate.translation_m for estimate in estimates],
        dtype=np.float64,
    )
    return np.mean(translations, axis=0)

def compute_eye_to_hand_calibration(
    session_dir: Path,
    tool_to_tag_translation_m: list[float] | None = None,
    tool_to_tag_rpy_radians: list[float] | None = None,
) -> EyeToHandCalibrationResult:
    """
    Compute the final eye-to-hand calibration result for one session.

    Inputs:
        session_dir: Path to the calibration session directory.
        tool_to_tag_translation_m: Optional tool-to-tag translation override.
        tool_to_tag_rpy_radians: Optional tool-to-tag rotation override.

    Returns:
        EyeToHandCalibrationResult: Final computed calibration result.
    """
    if tool_to_tag_translation_m is None:
        tool_to_tag_translation_m = [0.0, 0.0, 0.013]

    if tool_to_tag_rpy_radians is None:
        tool_to_tag_rpy_radians = [0.0, 0.0, math.pi]

    estimates = compute_session_estimates(
        session_dir=session_dir,
        tool_to_tag_translation_m=tool_to_tag_translation_m,
        tool_to_tag_rpy_radians=tool_to_tag_rpy_radians,
    )

    translations = np.array(
        [estimate.translation_m for estimate in estimates],
        dtype=np.float64,
    )
    mean_translation = np.mean(translations, axis=0)

    rotation_matrices = np.array(
        [estimate.base_T_camera[:3, :3] for estimate in estimates],
        dtype=np.float64,
    )
    mean_rotation = np.mean(rotation_matrices, axis=0)

    u_matrix, _singular_values, v_transpose = np.linalg.svd(mean_rotation)
    mean_rotation_orthonormal = u_matrix @ v_transpose

    if np.linalg.det(mean_rotation_orthonormal) < 0.0:
        u_matrix[:, -1] *= -1.0
        mean_rotation_orthonormal = u_matrix @ v_transpose

    mean_base_T_camera = make_transform_matrix(
        translation_m=mean_translation.astype(float).tolist(),
        rotation_matrix=mean_rotation_orthonormal,
    )

    roll, pitch, yaw = rotation_matrix_to_rpy_radians(mean_rotation_orthonormal)

    return EyeToHandCalibrationResult(
        sample_count=len(estimates),
        mean_translation_m=mean_translation.astype(float).tolist(),
        mean_rpy_radians=[float(roll), float(pitch), float(yaw)],
        mean_base_T_camera=mean_base_T_camera,
    )

def rotation_matrix_to_quaternion_xyzw(rotation: np.ndarray) -> list[float]:
    """
    Convert a 3x3 rotation matrix to a quaternion in XYZW order.

    Inputs:
        rotation: 3x3 rotation matrix.

    Returns:
        list[float]: Quaternion as [x, y, z, w].
    """
    trace = float(np.trace(rotation))

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * scale
        x_value = (rotation[2, 1] - rotation[1, 2]) / scale
        y_value = (rotation[0, 2] - rotation[2, 0]) / scale
        z_value = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w_value = (rotation[2, 1] - rotation[1, 2]) / scale
        x_value = 0.25 * scale
        y_value = (rotation[0, 1] + rotation[1, 0]) / scale
        z_value = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w_value = (rotation[0, 2] - rotation[2, 0]) / scale
        x_value = (rotation[0, 1] + rotation[1, 0]) / scale
        y_value = 0.25 * scale
        z_value = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w_value = (rotation[1, 0] - rotation[0, 1]) / scale
        x_value = (rotation[0, 2] + rotation[2, 0]) / scale
        y_value = (rotation[1, 2] + rotation[2, 1]) / scale
        z_value = 0.25 * scale

    quaternion = np.array(
        [x_value, y_value, z_value, w_value],
        dtype=np.float64,
    )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.astype(float).tolist()


def save_eye_to_hand_calibration_result(
    result: EyeToHandCalibrationResult,
    output_path: Path,
    robot_base_frame: str = "panda_link0",
    camera_frame: str = "camera_mount/camera_link/conveyor_rgbd",
) -> None:
    """
    Save the final eye-to-hand calibration result to YAML.

    Inputs:
        result: Final computed calibration result.
        output_path: YAML output path.
        robot_base_frame: Robot base frame name.
        camera_frame: Camera frame name.

    Returns:
        None
    """
    quaternion_xyzw = rotation_matrix_to_quaternion_xyzw(
        result.mean_base_T_camera[:3, :3]
    )

    output_data = {
        "robot_base_frame": robot_base_frame,
        "camera_frame": camera_frame,
        "sample_count": int(result.sample_count),
        "base_to_camera": {
            "translation_m": {
                "x": float(result.mean_translation_m[0]),
                "y": float(result.mean_translation_m[1]),
                "z": float(result.mean_translation_m[2]),
            },
            "quaternion_xyzw": {
                "x": float(quaternion_xyzw[0]),
                "y": float(quaternion_xyzw[1]),
                "z": float(quaternion_xyzw[2]),
                "w": float(quaternion_xyzw[3]),
            },
            "rpy_radians": {
                "roll": float(result.mean_rpy_radians[0]),
                "pitch": float(result.mean_rpy_radians[1]),
                "yaw": float(result.mean_rpy_radians[2]),
            },
        },
    }

    with open(output_path, "w", encoding="utf-8") as output_file:
        yaml.safe_dump(output_data, output_file, sort_keys=False)


def print_session_summary(
    estimates: list[CameraTransformEstimate],
) -> None:
    """
    Print a simple consistency summary for one calibration session.

    Inputs:
        estimates: Per-sample camera transform estimates.

    Returns:
        None
    """
    mean_translation = compute_mean_translation(estimates)

    print()
    print("Session summary")
    print(
        "Mean translation_m="
        f"[{mean_translation[0]:.6f}, {mean_translation[1]:.6f}, {mean_translation[2]:.6f}]"
    )
    print()

    for estimate in estimates:
        translation = np.array(estimate.translation_m, dtype=np.float64)
        translation_error_m = np.linalg.norm(translation - mean_translation)

        print(
            f"{estimate.image_stem}: "
            f"translation_error_from_mean_m={translation_error_m:.6f}"
        )


def print_ground_truth_translation_comparison(
    estimates: list[CameraTransformEstimate],
    ground_truth_translation_m: list[float],
) -> None:
    """
    Print translation error from the supplied ground-truth camera position.

    Inputs:
        estimates: Per-sample camera transform estimates.
        ground_truth_translation_m: Ground-truth camera translation as [x, y, z].

    Returns:
        None
    """
    ground_truth = np.array(ground_truth_translation_m, dtype=np.float64)

    print()
    print(
        "Ground-truth translation_m="
        f"[{ground_truth[0]:.6f}, {ground_truth[1]:.6f}, {ground_truth[2]:.6f}]"
    )

    for estimate in estimates:
        translation = np.array(estimate.translation_m, dtype=np.float64)
        translation_error_m = np.linalg.norm(translation - ground_truth)

        print(
            f"{estimate.image_stem}: "
            f"translation_error_from_ground_truth_m={translation_error_m:.6f}"
        )


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for session-based eye-to-hand computation.

    Inputs:
        None

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compute base-to-camera estimates from one calibration session."
    )
    parser.add_argument(
        "--session-dir",
        required=True,
        help="Path to the calibration session directory.",
    )
    parser.add_argument(
        "--tool-tag-tx",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_TRANSLATION_M[0],
        help="Tool-to-tag translation x in meters.",
    )
    parser.add_argument(
        "--tool-tag-ty",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_TRANSLATION_M[1],
        help="Tool-to-tag translation y in meters.",
    )
    parser.add_argument(
        "--tool-tag-tz",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_TRANSLATION_M[2],
        help="Tool-to-tag translation z in meters.",
    )
    parser.add_argument(
        "--tool-tag-roll",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_RPY_RADIANS[0],
        help="Tool-to-tag roll in radians.",
    )
    parser.add_argument(
        "--tool-tag-pitch",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_RPY_RADIANS[1],
        help="Tool-to-tag pitch in radians.",
    )
    parser.add_argument(
        "--tool-tag-yaw",
        type=float,
        default=DEFAULT_TOOL_TO_TAG_RPY_RADIANS[2],
        help="Tool-to-tag yaw in radians.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample debug details.",
    )
    parser.add_argument(
        "--ground-truth-tx",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link translation x in meters.",
    )
    parser.add_argument(
        "--ground-truth-ty",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link translation y in meters.",
    )
    parser.add_argument(
        "--ground-truth-tz",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link translation z in meters.",
    )
    parser.add_argument(
        "--ground-truth-roll",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link roll in radians.",
    )
    parser.add_argument(
        "--ground-truth-pitch",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link pitch in radians.",
    )
    parser.add_argument(
        "--ground-truth-yaw",
        type=float,
        required=False,
        help="Ground-truth base-to-camera-link yaw in radians.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Compute base-to-camera estimates for one calibration session and print a summary.

    Inputs:
        None

    Returns:
        None
    """
    args = parse_args()
    session_dir = Path(args.session_dir)

    tool_to_tag_translation_m = [
        args.tool_tag_tx,
        args.tool_tag_ty,
        args.tool_tag_tz,
    ]
    tool_to_tag_rpy_radians = [
        args.tool_tag_roll,
        args.tool_tag_pitch,
        args.tool_tag_yaw,
    ]

    estimates = compute_session_estimates(
        session_dir=session_dir,
        tool_to_tag_translation_m=tool_to_tag_translation_m,
        tool_to_tag_rpy_radians=tool_to_tag_rpy_radians,
    )
    print_session_summary(estimates)

    if (
        args.ground_truth_tx is not None
        and args.ground_truth_ty is not None
        and args.ground_truth_tz is not None
        and args.ground_truth_roll is not None
        and args.ground_truth_pitch is not None
        and args.ground_truth_yaw is not None
    ):
        ground_truth_base_T_camera_link = make_transform_matrix(
            translation_m=[
                args.ground_truth_tx,
                args.ground_truth_ty,
                args.ground_truth_tz,
            ],
            rotation_matrix=rpy_radians_to_rotation_matrix(
                args.ground_truth_roll,
                args.ground_truth_pitch,
                args.ground_truth_yaw,
            ),
        )

        ground_truth_base_T_camera = apply_camera_frame_rotation(
            ground_truth_base_T_camera_link,
            rpy_radians_to_rotation_matrix(
                DEFAULT_CAMERA_LINK_TO_SENSOR_RPY_RADIANS[0],
                DEFAULT_CAMERA_LINK_TO_SENSOR_RPY_RADIANS[1],
                DEFAULT_CAMERA_LINK_TO_SENSOR_RPY_RADIANS[2],
            ),
        )

        print_ground_truth_translation_comparison(
            estimates,
            ground_truth_base_T_camera[:3, 3].astype(float).tolist(),
        )


if __name__ == "__main__":
    main()
