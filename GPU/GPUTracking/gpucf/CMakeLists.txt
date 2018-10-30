cmake_minimum_required(VERSION 3.5)


set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wduplicated-cond")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wduplicated-branches")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wlogical-op")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wold-style-cast")
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} -Wuseless-cast")
set(CMAKE_CXX_FLAGS_DEBUG   "-Og -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
