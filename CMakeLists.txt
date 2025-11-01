cmake_minimum_required(VERSION 3.12.0)
project(LMD VERSION 0.1.0 LANGUAGES C CXX)

#exe output
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_RELEASE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message("In debug mode!")
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message("In release mode!")
endif()

set(DIR_SRCS "./benchmark/experiment.cpp")

add_executable(LMD ${DIR_SRCS})
add_executable(SIMPLE_TEST "./main.cpp")
