from dataclasses import dataclass
import numpy as np


@dataclass
class DetectedCube:
    """
    Internal vision node representation of a detected cube.

    Holds all raw and derived data produced during detection so the node
    has a persistent world model it can refer to between service calls.
    The ROS DetectedObject message is populated from this class rather
    than being constructed fresh on every request.

    All positional values are in the camera frame.
    All dimensions are in metres.
    """

    id: str

    # Image-space data
    color_mask: np.ndarray        # full HSV mask, may include side faces
    top_face_mask: np.ndarray     # filtered to top face pixels only
    top_face_contour: np.ndarray  # largest contour from top face mask
    centroid_uv: tuple[int, int]  # pixel coordinates of top face centroid

    # Depth and 3D position in camera frame
    top_depth_m: float
    position_camera_xyz: list[float]

    # Sensor-derived size estimates
    estimated_half_height_m: float
    estimated_half_width_m: float

    confidence: float = 1.0