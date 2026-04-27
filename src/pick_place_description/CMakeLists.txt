cmake_minimum_required(VERSION 3.8)
project(pick_place_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes materials
  DESTINATION share/${PROJECT_NAME}
)

ament_package()