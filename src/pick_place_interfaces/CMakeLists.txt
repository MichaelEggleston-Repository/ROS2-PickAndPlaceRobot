cmake_minimum_required(VERSION 3.8)
project(pick_place_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/TaskSpacePose.msg"
  "srv/CaptureSnapshot.srv"
  "srv/PlanToTaskPose.srv"
  "srv/ExecuteTaskPose.srv"
  DEPENDENCIES
    sensor_msgs
    trajectory_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()