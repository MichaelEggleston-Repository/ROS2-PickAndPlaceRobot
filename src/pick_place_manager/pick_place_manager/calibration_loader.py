from pathlib import Path
import yaml

def load_eye_to_hand_calibration(calibration_file: Path) -> dict:
    if not calibration_file.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_file}")

    with calibration_file.open("r", encoding="utf-8") as file:
        calibration_data = yaml.safe_load(file)

    if calibration_data is None:
        raise ValueError(f"Calibration file is empty: {calibration_file}")

    base_to_camera = calibration_data["base_to_camera"]
    translation = base_to_camera["translation_m"]
    quaternion = base_to_camera["quaternion_xyzw"]
    rpy = base_to_camera["rpy_radians"]

    return {
        "robot_base_frame": calibration_data["robot_base_frame"],
        "camera_frame": calibration_data["camera_frame"],
        "sample_count": calibration_data["sample_count"],
        "translation_xyz": [
            translation["x"],
            translation["y"],
            translation["z"],
        ],
        "quaternion_xyzw": [
            quaternion["x"],
            quaternion["y"],
            quaternion["z"],
            quaternion["w"],
        ],
        "rpy": [
            rpy["roll"],
            rpy["pitch"],
            rpy["yaw"],
        ],
    }