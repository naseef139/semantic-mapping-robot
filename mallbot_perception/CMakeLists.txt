cmake_minimum_required(VERSION 3.8)
project(mallbot_perception)

# ament_cmake (not ament_python): registers reliably with colcon --symlink-install on Humble
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# the Python module: mallbot_perception/{semantic_mapper,notes,geometry,evaluate}.py
ament_python_install_package(${PROJECT_NAME})

# executables for `ros2 run mallbot_perception ...`
install(PROGRAMS scripts/semantic_mapper scripts/evaluate_notes
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
