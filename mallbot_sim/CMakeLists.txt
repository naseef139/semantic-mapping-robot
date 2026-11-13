cmake_minimum_required(VERSION 3.8)
project(mallbot_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf worlds config models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS scripts/cmd_vel_timeout.py scripts/colorlog.py scripts/frontier_explorer.py scripts/frontier_core.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES scripts/mlaunch.bash
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/auto_keepout.py
  scripts/room_core.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
