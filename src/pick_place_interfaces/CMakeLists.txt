cmake_minimum_required(VERSION 3.8)
project(pick_place_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TaskSpacePose.msg"
  "msg/DetectedObject.msg"
  "msg/PddlAction.msg"
  "srv/CaptureSnapshot.srv"
  "srv/PlanToTaskPose.srv"
  "srv/ExecuteTaskPose.srv"
  "srv/PlanToJointPositions.srv"
  "srv/ExecuteHome.srv"
  "srv/DetectedObjects.srv"
  "srv/MoveGripper.srv"
  "srv/ComputeApproachJoints.srv"
  "srv/GeneratePlan.srv"
  DEPENDENCIES
    sensor_msgs
    trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()