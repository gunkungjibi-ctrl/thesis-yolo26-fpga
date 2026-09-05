cmake_minimum_required(VERSION 3.10)
project(yolo_dpu_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV (มากับ Vitis AI container และ Ubuntu-on-Kria)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

add_executable(yolo_dpu_infer yolo_dpu_infer.cpp)

target_include_directories(yolo_dpu_infer PRIVATE ${OpenCV_INCLUDE_DIRS})

# VART / XIR / glog — ลิงก์ตรงด้วยชื่อ .so ของ Vitis AI 3.0.
# หมายเหตุ (ยืนยันจาก container 3.0.0.106): RunnerExt::create_runner อยู่ใน
# libvart-runner.so — ไม่มี libvart-runner-ext.so และไม่มี libunilog.so แยก.
target_link_libraries(yolo_dpu_infer
  PRIVATE
    ${OpenCV_LIBS}
    vart-runner
    xir
    glog
    pthread
)
