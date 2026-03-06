# hw001

Невеликий C++ проєкт з OpenCV, який рендерить 2D-зображення сцени з кількох текстурованих 3D кубоїдів (apple/cherry/pear) через піксельне трасування променів.

## Що робить
- формує сцену з об'єктів у системі координат NED;
- задає камеру (матриця `K`, поза `P`);
- для кожного пікселя знаходить перетин променя з найближчим об'єктом;
- бере колір із відповідної текстури та показує результат у вікні `render`.


## Дані
Текстури читаються з шляху відносно файлу `scane.h`:
`../../data/hw001/{apple,cherry,pear}`.
> Вони в гітігнорі, бо це не найкраща ідея завантажувати зображення на гітхаб

apple1.jpg  - 612 x 408
apple2.jpg  - 612 x 408
apple3.jpg  - 612 x 612

cherry1.jpg - 640 x 420
cherry2.jpg - 640 x 420
cherry3.jpg - 640 x 560

pear1.jpg   - 704 x 448
pear2.jpg   - 704 x 448
pear3.jpg   - 704 x 512

## Cmakelists.txt

```cmake_minimum_required(VERSION 4.1)
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
```