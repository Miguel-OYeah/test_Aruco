# ----------------------------------------------------------------------------
#   Basic Configuration
# ----------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.0)
project(test_aruco VERSION "0.1" LANGUAGES CXX)
set(PROJECT_SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 11) # C++11...
set(CMAKE_CXX_STANDARD_REQUIRED ON) #...is required...
set(CMAKE_CXX_EXTENSIONS ON) #...with compiler extensions like gnu++11

# ----------------------------------------------------------------------------
#   Find Dependencies
# ----------------------------------------------------------------------------

find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )

message(STATUS "OpenCV found, version: ${OpenCV_VERSION} in dir ${OpenCV_INCLUDE_DIRS}")

include(cmake/options.cmake)
include(cmake/compilerOptions.cmake)

add_subdirectory(src)

include(cmake/printInfo.cmake)

