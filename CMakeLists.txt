cmake_minimum_required(VERSION 4.1)
project(code)

set(CMAKE_CXX_STANDARD 26)

if(APPLE AND NOT OpenCV_DIR)
  set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)

add_executable(code HW/hw001/main.cpp
        HW/hw001/Cuboid3d.cpp
        HW/hw001/Cuboid3d.h
        HW/hw001/scane.h
        HW/hw001/scane.cpp)
target_include_directories(code PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(code PRIVATE ${OpenCV_LIBS})
