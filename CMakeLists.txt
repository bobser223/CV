cmake_minimum_required(VERSION 4.1)
project(code)

set(CMAKE_CXX_STANDARD 26)

if(APPLE AND NOT OpenCV_DIR)
  set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)

add_executable(code
#        HW/hw002/main.cpp

        HW/hw003/utils003.h
        HW/hw003/utils003.cpp
        HW/hw003/main.cpp
        HW/hw003/regressionDrawer.cpp
        HW/hw003/regressionDrawer.h
)
target_include_directories(code PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(code PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres
)
