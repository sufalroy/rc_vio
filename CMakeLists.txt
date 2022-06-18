cmake_minimum_required(VERSION 3.16.0)
project(rcvio VERSION 0.0.0)

# Set build type
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

# Enable debug flags (if you want to debug in gdb)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -g3")

# Find catkin (the ROS build system)
# Add this for finding ros dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp roslib tf sensor_msgs geometry_msgs nav_msgs cv_bridge eigen_conversions
)

# Describe catkin project
catkin_package(CATKIN_DEPENDS
  roscpp roslib tf std_msgs sensor_msgs geometry_msgs nav_msgs cv_bridge eigen_conversions
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

# Include libraries
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED)

# Include directories of headers
include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${EIGEN3_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}
)

# #################################################
# Make binary for the main library
# #################################################
add_library(${PROJECT_NAME} SHARED
  src/system.cpp
  src/updater.cpp
  src/tracker.cpp
  src/pre_integrator.cpp
  src/ransac.cpp
  src/input_buffer.cpp
  src/feature_detector.cpp
)

target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
)

# #################################################
# Make binary for main method ROS
# #################################################
add_executable(rcvio_mono src/rcvio_mono.cpp)
target_link_libraries(rcvio_mono
  ${PROJECT_NAME}
  ${OpenCV_LIBRARIES}
  ${catkin_LIBRARIES}
)
