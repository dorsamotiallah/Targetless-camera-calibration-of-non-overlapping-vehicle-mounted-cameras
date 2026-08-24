cmake_minimum_required(VERSION 3.0.0)
project(calib_nmc3d)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native")

add_definitions(-w)

find_package(catkin REQUIRED COMPONENTS roscpp tf sensor_msgs std_msgs image_transport cv_bridge)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

# calib.cpp's #include "edge.hpp" resolves against ../calib below; edge.cpp/
# edge.hpp are intentionally not duplicated here -- see README.md.
include_directories(
   ${catkin_INCLUDE_DIRS}
   ${EIGEN3_INCLUDE_DIR}
   ${OpenCV_INCLUDE_DIRS}
   ${PROJECT_SOURCE_DIR}/../orbslam3
   ${PROJECT_SOURCE_DIR}/../orbslam3/include
   ${PROJECT_SOURCE_DIR}/../orbslam3/include/CameraModels
   ${PROJECT_SOURCE_DIR}/../orbslam3/Thirdparty/Sophus
   ${PROJECT_SOURCE_DIR}/../calib
)

# NMC3D-variant calibration node: same CamMap matching/optimization as
# ../calib/calib.cpp, extended with depth- and image-grid-balanced
# ("uniformly distributed") match selection before the final Sim3 fit.
add_executable(calib_nmc3d
   main.cpp
   calib.cpp
   ${PROJECT_SOURCE_DIR}/../calib/edge.cpp
)

target_link_libraries(calib_nmc3d
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ORB_SLAM3
)

add_executable(atlas_ground_export
   atlas_ground_export.cpp
)

target_link_libraries(atlas_ground_export
   ${catkin_LIBRARIES}
   ${OpenCV_LIBS}
   ORB_SLAM3
)
