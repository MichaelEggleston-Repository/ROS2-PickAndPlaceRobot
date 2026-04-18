from dataclasses import dataclass

@dataclass
class CalibrationPose:
    """
    Describe a rigid pose using translation and quaternion rotation.

    Inputs:
        x: Position x in meters.
        y: Position y in meters.
        z: Position z in meters.
        qx: Quaternion x component.
        qy: Quaternion y component.
        qz: Quaternion z component.
        qw: Quaternion w component.

    Returns:
        None
    """
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

@dataclass
class CalibrationSample:
    """
    Describe one eye-to-hand calibration observation.

    Inputs:
        sample_id: Unique identifier for the collected sample.
        timestamp_sec: Time the sample was collected in seconds.
        tag_id: AprilTag identifier for the detected target.
        base_frame_id: Robot base reference frame, usually panda_link0.
        camera_frame_id: Camera frame used by the detector.
        tag_frame_id: Logical frame name for the calibration target.
        camera_to_tag: Detected tag pose relative to the camera frame.
        base_to_tag: Known tag pose relative to the robot base frame.

    Returns:
        None
    """
    sample_id: int
    timestamp_sec: float
    tag_id: int
    base_frame_id: str
    camera_frame_id: str
    tag_frame_id: str
    camera_to_tag: CalibrationPose
    base_to_tag: CalibrationPose