# Copyright 2019-2026 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Standalone build of Framework/AODNative without the O2 stack, for quick
# iteration: minimal shims for the o2_* CMake functions, Arrow C++ from a
# pyarrow wheel, Catch2 from its amalgamated sources. See ../../DEVELOPMENT.md.
#
#   cmake -S Framework/AODNative/test/standalone -B build -G Ninja \
#         -DPYARROW_DIR=<venv>/lib/python3.X/site-packages/pyarrow \
#         -DCATCH2_DIR=<dir with catch_amalgamated.hpp/.cpp>
cmake_minimum_required(VERSION 3.20)
project(AODNativeStandalone CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
enable_testing()

option(SANITIZE "Build with ASan/UBSan" OFF)
if(SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()
add_compile_options(-Wall -Wextra -Wno-unused-parameter)

get_filename_component(O2_SOURCE_DEFAULT ${CMAKE_CURRENT_LIST_DIR}/../../../.. ABSOLUTE)
set(O2_SOURCE ${O2_SOURCE_DEFAULT} CACHE PATH "O2 source tree")
set(PYARROW_DIR "" CACHE PATH "pyarrow package directory, with include/ and libarrow.so.*")
set(CATCH2_DIR "" CACHE PATH "Directory with catch_amalgamated.hpp and catch_amalgamated.cpp")

file(GLOB ARROW_LIB ${PYARROW_DIR}/libarrow.so.*)
add_library(Arrow::arrow_shared SHARED IMPORTED)
set_target_properties(Arrow::arrow_shared PROPERTIES
  IMPORTED_LOCATION ${ARROW_LIB}
  INTERFACE_INCLUDE_DIRECTORIES ${PYARROW_DIR}/include)

add_library(catch2 STATIC ${CATCH2_DIR}/catch_amalgamated.cpp)
target_include_directories(catch2 PUBLIC ${CATCH2_DIR})
add_library(O2::Catch2 ALIAS catch2)

find_package(ZLIB REQUIRED)
find_package(LibLZMA REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(zstd REQUIRED IMPORTED_TARGET libzstd)
pkg_check_modules(lz4 REQUIRED IMPORTED_TARGET liblz4)

function(o2_add_library name)
  cmake_parse_arguments(PARSE_ARGV 1 A "" "TARGETVARNAME"
    "SOURCES;PUBLIC_INCLUDE_DIRECTORIES;PUBLIC_LINK_LIBRARIES;PRIVATE_INCLUDE_DIRECTORIES;PRIVATE_LINK_LIBRARIES")
  set(target O2lib-${name})
  add_library(${target} SHARED ${A_SOURCES})
  add_library(O2::${name} ALIAS ${target})
  target_include_directories(${target} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include PRIVATE ${A_PRIVATE_INCLUDE_DIRECTORIES})
  target_link_libraries(${target} PUBLIC ${A_PUBLIC_LINK_LIBRARIES} PRIVATE ${A_PRIVATE_LINK_LIBRARIES})
  if(A_TARGETVARNAME)
    set(${A_TARGETVARNAME} ${target} PARENT_SCOPE)
  endif()
endfunction()

function(o2_add_executable name)
  cmake_parse_arguments(PARSE_ARGV 1 A "IS_TEST;NO_INSTALL;IS_BENCHMARK" "COMPONENT_NAME;TARGETVARNAME" "SOURCES;PUBLIC_LINK_LIBRARIES")
  set(target o2-${A_COMPONENT_NAME}-${name})
  add_executable(${target} ${A_SOURCES})
  target_link_libraries(${target} PRIVATE ${A_PUBLIC_LINK_LIBRARIES})
endfunction()

add_subdirectory(${O2_SOURCE}/Framework/AODNative AODNative)
