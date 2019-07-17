cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wduplicated-cond")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wduplicated-branches")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wlogical-op")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wold-style-cast")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wuseless-cast")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} -Og -g3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

list(APPEND CMAKE_PREFIX_PATH $ENV{ROOTSYS})
find_package(ROOT REQUIRED COMPONENTS RIO Net)
# include(${ROOT_USE_FILE})
include_directories(SYSTEM ${ROOT_INCLUDE_DIRS})
link_directories(${ROOT_LIBRARY})

add_definitions(
    -DCL_HPP_ENABLE_EXCEPTIONS
    -DCL_TARGET_OPENCL_VERSION=200
    -DCL_HPP_TARGET_OPENCL_VERSION=200

    -Dspan_FEATURE_MAKE_SPAN=1
    -Dspan_FEATURE_MEMBER_BACK_FRONT=1
    -Dspan_FEATURE_MEMBER_AT=2
    -Dspan_CONFIG_INDEX_TYPE=std::size_t
)

include_directories(src)
include_directories(SYSTEM external)
include_directories(cl)
add_subdirectory(src)
