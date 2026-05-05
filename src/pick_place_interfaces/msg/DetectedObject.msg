# DetectedObject.msg
string id
float64 confidence
TaskSpacePose pose_camera

# Sensor-derived size estimates computed from the depth image.
# Half-extents are stored directly since grasp and stack calculations
# both use half-dimensions (centroid ± half_extent).
float64 estimated_half_height_m
float64 estimated_half_width_m