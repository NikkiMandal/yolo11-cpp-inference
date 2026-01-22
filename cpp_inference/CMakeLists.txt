cmake_minimum_required(VERSION 3.10)
project(yolo_inference)

set(CMAKE_CXX_STANDARD 14)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# ONNX Runtime
set(ONNXRUNTIME_DIR "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-1.16.3")
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Original executable (with webcam)
add_executable(yolo_inference main.cpp)
target_link_libraries(yolo_inference ${OpenCV_LIBS} onnxruntime)

# Benchmark executable (no webcam needed)
add_executable(yolo_benchmark main_benchmark.cpp)
target_link_libraries(yolo_benchmark ${OpenCV_LIBS} onnxruntime)

# Video test executable
add_executable(yolo_video main_video.cpp)
target_link_libraries(yolo_video ${OpenCV_LIBS} onnxruntime)