cmake_minimum_required(VERSION 2.8.3)
project(signals)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  message_generation
)

include_directories(${catkin_INCLUDE_DIRS})

generate_messages(
   DEPENDENCIES
   std_msgs
)

catkin_package(
  CATKIN_DEPENDS openface_tracker
)

install(PROGRAMS
  nodes/body.py
  nodes/brain.py
  nodes/interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES
  launch/signals.launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
