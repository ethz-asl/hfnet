cmake_minimum_required(VERSION 3.0.2)
project(hfnet_ros)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(CUDA)
include(FindProtobuf)
find_package(Protobuf REQUIRED "3.6.0")

if (Protobuf_FOUND)
    INCLUDE_DIRECTORIES(${PROTOBUF_INCLUDE_DIR})
else (Protobuf_FOUND)
    MESSAGE(FATAL_ERROR "PROTOBUF NOT FOUNED")
endif (Protobuf_FOUND)

find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_transport
  roscpp
  rospy
  std_msgs
)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES 
  CATKIN_DEPENDS
#  DEPENDS system_lib
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}
  "/usr/local/include/eigen3"
  ${OpenCV_INCLUDE_DIRS}
)

# tensorflow
include_directories(/usr/local/include/tf)
include_directories(/usr/local/include/tf/tensorflow)
#include_directories(/usr/local/include/tf/bazel-tensorflow/external/eigen_archive)
#include_directories(/usr/local/include/tf/bazel-tensorflow/external/protobuf_archive/src)
include_directories(/usr/local/include/tf/absl)
include_directories(/usr/local/include/tf/bazel-genfiles)
include_directories(/usr/local/include/tf/tensorflow/cc)
include_directories(/usr/local/include/tf/tensorflow/c)
include_directories(/usr/local/include/tf/tensorflow/core)

link_directories(/usr/local/lib)

add_executable(hfnet_ros_node src/hfnet_ros_node.cpp)
find_package(CUDA)
if(CUDA_FOUND)
  message("FIND_CUDA")
  target_link_libraries(hfnet_ros_node 
                        ${CUDA_LIBRARIES}
                        ${PROTOBUF_LIBRARY}
                        ${OpenCV_LIBS}
                        /usr/local/lib/libtensorflow_cc.so
                        /usr/local/lib/libtensorflow_framework.so
			${catkin_LIBRARIES}
                        )

endif()


