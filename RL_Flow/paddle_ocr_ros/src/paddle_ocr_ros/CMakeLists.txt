cmake_minimum_required(VERSION 3.0.2)
project(paddle_ocr_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  cv_bridge
  geometry_msgs
  message_generation
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 添加服务文件
add_service_files(
  FILES
  OCRService.srv
)

# 生成服务消息
generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
  geometry_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs sensor_msgs cv_bridge geometry_msgs message_runtime
)

# 安装Python脚本
catkin_install_python(PROGRAMS
  scripts/paddle_ocr_node.py
  scripts/ocr_client_test.py
  scripts/service_monitor.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装启动文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# 安装配置文件
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)
