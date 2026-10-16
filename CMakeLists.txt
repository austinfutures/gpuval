cmake_minimum_required(VERSION 3.16)
project(gpuval CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)
add_executable(gpuval
  src/main.cpp src/fp16.cpp src/reference.cpp src/raster.cpp
  src/test_fp16.cpp src/test_raster.cpp)
target_compile_options(gpuval PRIVATE -Wall -Wextra)
target_link_libraries(gpuval PRIVATE Threads::Threads)
