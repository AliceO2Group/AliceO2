# Copyright 2019-2020 CERN and copyright holders of ALICE O2.
# See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
# All rights not expressly granted are reserved.
#
# This software is distributed under the terms of the GNU General Public
# License v3 (GPL Version 3), copied verbatim in the file "COPYING".
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization
# or submit itself to any jurisdiction.

# Some general CMake settings
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(GPUTrackingStandalone)
include(FeatureSummary)
include(GNUInstallDirs)
set(CMAKE_INSTALL_MESSAGE LAZY)

set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

set(GPU_DIR "${CMAKE_SOURCE_DIR}/../../")
set(GPUTRACKING_DIR "${GPU_DIR}/GPUTracking")
set(O2_DIR "${CMAKE_SOURCE_DIR}/../../../")

set(CMAKE_MODULE_PATH ${O2_DIR}/dependencies ${CMAKE_MODULE_PATH})

# Copy and include Config File
if(NOT EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  file(COPY "${CMAKE_SOURCE_DIR}/cmake/config.cmake" DESTINATION "${CMAKE_BINARY_DIR}")
endif()
include("${CMAKE_BINARY_DIR}/config.cmake")
if(DEFINED GPUCA_CONFIG_COMPILER)
  if(GPUCA_CONFIG_COMPILER STREQUAL "clang")
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
  elseif(GPUCA_CONFIG_COMPILER STREQUAL "gcc")
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "c++")
  else()
    set(CMAKE_C_COMPILER "${GPUCA_CONFIG_COMPILER}")
    set(CMAKE_CXX_COMPILER "${GPUCA_CONFIG_COMPILER}")
  endif()
endif()

# Set Build and Compiler settings
set(ALIGPU_BUILD_TYPE "Standalone")
add_definitions(-DGPUCA_STANDALONE)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(GPUCA_BUILD_DEBUG)
  set(CMAKE_CXX_FLAGS "-O0 -ggdb")
  set(CMAKE_BUILD_TYPE DEBUG)
else()
  set(CMAKE_CXX_FLAGS "-O3 -march=native -ggdb")
  set(CMAKE_BUILD_TYPE RELEASE)
  add_definitions(-DNDEBUG)
endif()
if (GPUCA_BUILD_DEBUG_SANITIZE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize=vptr")
  if(CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -shared-libasan")
  endif()
endif()
if(GPUCA_CONFIG_WERROR)
  string(APPEND CMAKE_CXX_FLAGS " -Werror -Wno-unknown-warning-option")
else()
  string(APPEND CMAKE_CXX_FLAGS " -Wno-error")
endif()
string(APPEND CMAKE_CXX_FLAGS " -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unknown-pragmas -Wno-write-strings")
string(APPEND CMAKE_SHARED_LINKER_FLAGS " -rdynamic -Wl,--no-undefined")
if(CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
  string(APPEND CMAKE_CXX_FLAGS " -Wno-vla-cxx-extension")
endif()

# Find mandatory packages
find_package(TBB REQUIRED)

if(GPUCA_CONFIG_VC)
  find_package(Vc REQUIRED)
else()
  set(Vc_FOUND OFF)
endif()

if(GPUCA_BUILD_EVENT_DISPLAY)
  find_package(GLFW NAMES glfw3 CONFIG REQUIRED)
  find_package(GLEW REQUIRED)
  find_package(GLUT REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(X11 REQUIRED)
  if(GPUCA_BUILD_EVENT_DISPLAY_FREETYPE)
    find_package(Freetype REQUIRED)
    find_package(Fontconfig REQUIRED)
  endif()
  if(GPUCA_BUILD_EVENT_DISPLAY_VULKAN)
    find_package(Vulkan REQUIRED)
    if (Vulkan_GLSLC_EXECUTABLE STREQUAL "Vulkan_GLSLC_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR ${Vulkan_GLSLC_EXECUTABLE})
    endif()
    if(GPUCA_BUILD_EVENT_DISPLAY_WAYLAND)
      find_package(O2GPUWayland REQUIRED)
    endif()
  else()
    set(Vulkan_FOUND OFF)
  endif()
  if(GPUCA_BUILD_EVENT_DISPLAY_QT)
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
  endif()
else()
  set(OpenGL_FOUND OFF)
endif()

if(GPUCA_CONFIG_ROOT)
  find_package(ROOT CONFIG REQUIRED)
elseif(GPUCA_BUILD_EVENT_DISPLAY)
  message(FATAL_ERROR "Cannot build event display without ROOT")
endif()
find_package(Microsoft.GSL REQUIRED HINTS "$ENV{MS_GSL_ROOT}/share/cmake")

if(GPUCA_CONFIG_FMT)
  find_package(fmt REQUIRED HINTS $ENV{FMT_ROOT})
else()
  set(fmt_FOUND OFF)
endif()

# Detect GPU Backends
find_package(O2GPU REQUIRED)

if(GPUCA_CONFIG_ONNX)
  find_package(onnxruntime REQUIRED)
  if(HIP_ENABLED AND NOT DEFINED ORT_ROCM_BUILD)
    set(ORT_ROCM_BUILD ON)
  elseif(CUDA_ENABLED AND NOT DEFINED ORT_CUDA_BUILD)
    set(ORT_CUDA_BUILD ON)
  endif()
else()
  set(onnxruntime_FOUND OFF)
endif()

# Create main targets
add_subdirectory(../../ GPU)
add_library(standalone_support SHARED
            ${O2_DIR}/Common/Field/src/MagFieldFast.cxx
            ${O2_DIR}/Common/Utils/src/StringUtils.cxx
            ${O2_DIR}/DataFormats/Detectors/TPC/src/CompressedClusters.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/TrackParametrization.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/TrackParametrizationWithError.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/Vertex.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/TrackLTIntegral.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/TrackParametrization.cxx
            ${O2_DIR}/DataFormats/Reconstruction/src/TrackParametrizationWithError.cxx
            ${O2_DIR}/Detectors/TRD/base/src/GeometryBase.cxx
            ${O2_DIR}/Detectors/Base/src/MatLayerCylSet.cxx
            ${O2_DIR}/Detectors/Base/src/MatLayerCyl.cxx
            ${O2_DIR}/Detectors/Base/src/Ray.cxx
            ${O2_DIR}/Detectors/Base/src/Propagator.cxx)

target_compile_definitions(standalone_support PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_include_directories(standalone_support PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,INCLUDE_DIRECTORIES>)

target_include_directories(standalone_support PUBLIC
                           ${GPU_DIR}/Common
                           ${GPU_DIR}/Utils
                           ${O2_DIR}/Common/Field/include
                           ${O2_DIR}/Common/Constants/include
                           ${O2_DIR}/Common/MathUtils/include
                           ${O2_DIR}/Common/Utils/include
                           ${O2_DIR}/Common/ML/include
                           ${O2_DIR}/DataFormats/common/include
                           ${O2_DIR}/DataFormats/Detectors/Common/include
                           ${O2_DIR}/DataFormats/Detectors/ITSMFT/common/include
                           ${O2_DIR}/DataFormats/Detectors/ITSMFT/ITS/include
                           ${O2_DIR}/DataFormats/Detectors/TOF/include
                           ${O2_DIR}/DataFormats/Detectors/TPC/include
                           ${O2_DIR}/DataFormats/Detectors/TRD/include
                           ${O2_DIR}/DataFormats/Headers/include
                           ${O2_DIR}/DataFormats/MemoryResources/include
                           ${O2_DIR}/DataFormats/Reconstruction/include
                           ${O2_DIR}/DataFormats/Reconstruction/src
                           ${O2_DIR}/DataFormats/simulation/include
                           ${O2_DIR}/Detectors/Base/include
                           ${O2_DIR}/Detectors/Base/src
                           ${O2_DIR}/Detectors/ITSMFT/common/base/include
                           ${O2_DIR}/Detectors/ITSMFT/ITS/base/include
                           ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/include
                           ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/GPU
                           ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/GPU/cuda
                           ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/GPU/hip
                           ${O2_DIR}/Detectors/Raw/include
                           ${O2_DIR}/Detectors/TOF/base/include
                           ${O2_DIR}/Detectors/TPC/base/include
                           ${O2_DIR}/Detectors/TRD/base/include
                           ${O2_DIR}/Detectors/TRD/base/src
                           ${O2_DIR}/Framework/Foundation/3rdparty/include)

target_link_libraries(standalone_support PUBLIC#
                      dl
                      pthread
                      Microsoft.GSL::GSL)

target_link_libraries(GPUTracking PUBLIC standalone_support)
target_link_libraries(TPCFastTransformation PUBLIC standalone_support)

if(GPUCA_CONFIG_O2)
  target_compile_definitions(GPUTracking PRIVATE GPUCA_TPC_GEOMETRY_O2)
endif()

if(GPUCA_CONFIG_FMT)
  target_link_libraries(standalone_support PUBLIC fmt::fmt)
else()
  target_compile_definitions(standalone_support PUBLIC GPUCA_NO_FMT)
endif()

if(GPUCA_CONFIG_VC)
  target_link_libraries(standalone_support PUBLIC Vc::Vc)
else()
  target_compile_definitions(standalone_support PUBLIC GPUCA_NO_VC)
endif()

if(GPUCA_CONFIG_ROOT)
  target_link_libraries(standalone_support PUBLIC
                        ROOT::Core
                        ROOT::RIO
                        ROOT::Hist
                        ROOT::Gui
                        ROOT::Tree)
endif()

if(GPUCA_CONFIG_ONNX)
  target_link_libraries(standalone_support PRIVATE onnxruntime::onnxruntime)
  target_sources(standalone_support PRIVATE ${O2_DIR}/Common/ML/src/OrtInterface.cxx)
endif()

if (GPUCA_BUILD_DEBUG_SANITIZE AND CMAKE_CXX_COMPILER MATCHES "clang\\+\\+")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libclang_rt.asan-x86_64.so OUTPUT_VARIABLE CLANG_ASAN_SO_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
  get_filename_component(CLANG_ASAN_SO_PATH "${CLANG_ASAN_SO_PATH}" DIRECTORY)
  get_filename_component(CLANG_ASAN_SO_PATH "${CLANG_ASAN_SO_PATH}" ABSOLUTE)
  target_link_options(ca PRIVATE "-Wl,-rpath,${CLANG_ASAN_SO_PATH}")
endif()

# Installation
install(TARGETS ca TPCFastTransformation standalone_support)
install(FILES "cmake/makefile" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${O2_DIR} ${CMAKE_INSTALL_PREFIX}/src)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/config.cmake ${CMAKE_INSTALL_PREFIX}/config.cmake)")
install(DIRECTORY tools DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/GPU)
