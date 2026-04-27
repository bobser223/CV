cmake_minimum_required(VERSION 4.1)
project(code)

set(CMAKE_CXX_STANDARD 26)

if(APPLE AND NOT OpenCV_DIR)
  set(OpenCV_DIR "/opt/homebrew/lib/cmake/opencv4")
endif()

find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Matplot++ REQUIRED)


add_executable(homework003
#        HW/hw002/main.cpp

        HW/hw003/utils003.h
        HW/hw003/utils003.cpp
        HW/hw003/main.cpp
        HW/hw003/regressionDrawer.cpp
        HW/hw003/regressionDrawer.h
        HW/hw003/task4.cpp
        HW/hw003/task4.h

)

target_include_directories(homework003 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(homework003 PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres
        Matplot++::matplot
)


add_executable( homework004
        HW/hw004/main.cpp
        HW/hw004/utils002.cpp
        HW/hw004/utils002.h

)
target_include_directories(homework004 PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(homework004 PRIVATE
        ${OpenCV_LIBS}
        Ceres::ceres
        Matplot++::matplot
)

