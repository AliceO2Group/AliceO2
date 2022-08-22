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
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(GPUTrackingStandalone)
include(FeatureSummary)
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
if(DEFINED CONFIG_COMPILER)
  if(CONFIG_COMPILER STREQUAL "clang")
    set(CMAKE_C_COMPILER "clang")
    set(CMAKE_CXX_COMPILER "clang++")
  elseif(CONFIG_COMPILER STREQUAL "gcc")
    set(CMAKE_C_COMPILER "gcc")
    set(CMAKE_CXX_COMPILER "c++")
  else()
    set(CMAKE_C_COMPILER "${CONFIG_COMPILER}")
    set(CMAKE_CXX_COMPILER "${CONFIG_COMPILER}")
  endif()
endif()

# Set Build and Compiler settings
set(ALIGPU_BUILD_TYPE "Standalone")
add_definitions(-DGPUCA_STANDALONE)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(BUILD_DEBUG)
  set(CMAKE_CXX_FLAGS "-Og -ggdb")
  set(CMAKE_BUILD_TYPE DEBUG)
else()
  set(CMAKE_CXX_FLAGS "-O3 -march=native -ggdb -minline-all-stringops -funroll-loops -ffast-math -fno-stack-protector")
  if (NOT CMAKE_CXX_COMPILER STREQUAL "clang++")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftracer -fprefetch-loop-arrays")
  endif()
  set(CMAKE_RELEASE_TYPE RELEASE)
  add_definitions(-DNDEBUG)
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unknown-pragmas -Wno-write-strings")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -Wl,--no-undefined")

# Check and set settings for optional packages
if(CONFIG_OPENMP)
  find_package(OpenMP REQUIRED)
else()
  set(OpenMP_FOUND OFF)
  set(OpenMP_CXX_FOUND OFF)
endif()

if(CONFIG_VC)
  find_package(Vc REQUIRED)
else()
  set(Vc_FOUND OFF)
  add_definitions(-DGPUCA_NO_VC)
endif()

if(BUILD_EVENT_DISPLAY)
  find_package(GLFW NAMES glfw3 CONFIG REQUIRED)
  find_package(GLEW REQUIRED)
  find_package(GLUT REQUIRED)
  find_package(OpenGL REQUIRED)
  find_package(X11 REQUIRED)
  if(BUILD_EVENT_DISPLAY_FREETYPE)
    find_package(Freetype REQUIRED)
    find_package(Fontconfig REQUIRED)
  endif()
  if(BUILD_EVENT_DISPLAY_VULKAN)
    find_package(Vulkan REQUIRED)
    if (Vulkan_GLSLC_EXECUTABLE STREQUAL "Vulkan_GLSLC_EXECUTABLE-NOTFOUND")
      message(FATAL_ERROR ${Vulkan_GLSLC_EXECUTABLE})
    endif()
    if(BUILD_EVENT_DISPLAY_WAYLAND)
      find_package(O2GPUWayland REQUIRED)
    endif()
  else()
    set(Vulkan_FOUND OFF)
  endif()
  if(BUILD_EVENT_DISPLAY_QT)
    find_package(Qt5 COMPONENTS Widgets REQUIRED)
  endif()
else()
  set(OpenGL_FOUND OFF)
endif()

if(CONFIG_O2)
  add_definitions(-DGPUCA_TPC_GEOMETRY_O2)
endif()

if(CONFIG_ROOT)
  find_package(ROOT CONFIG REQUIRED)
else()
  add_definitions(-DGPUCA_NO_ROOT)
endif()
if(CONFIG_O2_EXTENSIONS)
  add_definitions(-DGPUCA_HAVE_O2HEADERS)
  find_package(Microsoft.GSL REQUIRED HINTS "$ENV{MS_GSL_ROOT}/share/cmake")
  if(CONFIG_O2_ITS_TRAITS)
    if(NOT CONFIG_ROOT)
      message(FATAL_ERROR "Can not build ITS traits without ROOT")
    endif()
    find_package(Boost REQUIRED)
  else()
    add_definitions(-DGPUCA_NO_ITS_TRAITS)
  endif()
endif()

if(CONFIG_FMT)
  find_package(fmt REQUIRED HINTS $ENV{FMT_ROOT})
else()
  add_definitions(-DGPUCA_NO_FMT)
endif()

# Detect GPU Backends
find_package(O2GPU)

# Global include directories
include_directories(${GPU_DIR}/Common
                    ${GPU_DIR}/Utils
                    ${GPU_DIR}/TPCFastTransformation
                    ${GPUTRACKING_DIR}
                    ${GPUTRACKING_DIR}/Debug
                    ${GPUTRACKING_DIR}/Definitions
                    ${GPUTRACKING_DIR}/DataTypes
                    ${GPUTRACKING_DIR}/Base
                    ${GPUTRACKING_DIR}/Base/opencl-common
                    ${GPUTRACKING_DIR}/dEdx
                    ${GPUTRACKING_DIR}/TPCConvert
                    ${GPUTRACKING_DIR}/Global
                    ${GPUTRACKING_DIR}/HLTHeaders
                    ${GPUTRACKING_DIR}/Merger
                    ${GPUTRACKING_DIR}/Refit
                    ${GPUTRACKING_DIR}/qa
                    ${GPUTRACKING_DIR}/SliceTracker
                    ${GPUTRACKING_DIR}/DataCompression
                    ${GPUTRACKING_DIR}/TRDTracking)

if(CONFIG_O2_EXTENSIONS)
include_directories(${GPUTRACKING_DIR}/TPCClusterFinder
                    ${GPUTRACKING_DIR}/ITS
                    ${GPUTRACKING_DIR}/Interface
                    ${O2_DIR}/Common/Field/include
                    ${O2_DIR}/Common/Constants/include
                    ${O2_DIR}/Common/MathUtils/include
                    ${O2_DIR}/Common/Utils/include
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
                    ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/include
                    ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/cuda/src
                    ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/hip/include
                    ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/hip/src
                    ${O2_DIR}/Detectors/Raw/include
                    ${O2_DIR}/Detectors/TOF/base/include
                    ${O2_DIR}/Detectors/TPC/base/include
                    ${O2_DIR}/Detectors/TRD/base/include
                    ${O2_DIR}/Detectors/TRD/base/src
                    ${O2_DIR}/Framework/Foundation/3rdparty/include)
endif()

# Create main targets
add_subdirectory(../../ GPU)
add_library(standalone_support SHARED ${GPUTRACKING_DIR}/utils/EmptyFile.cxx)

target_link_libraries(GPUTracking PUBLIC TPCFastTransformation standalone_support dl)
target_link_libraries(ca PUBLIC GPUTracking)
target_link_libraries(standalone_support PUBLIC dl pthread)
target_compile_definitions(ca PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_compile_definitions(standalone_support PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

# Add all sources and dependencies to to support based on Config File
if(CONFIG_O2_EXTENSIONS)
  target_sources(standalone_support PRIVATE
               ${O2_DIR}/Common/Field/src/MagFieldFast.cxx
               ${O2_DIR}/DataFormats/Detectors/TPC/src/CompressedClusters.cxx
               ${O2_DIR}/DataFormats/simulation/src/MCCompLabel.cxx
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
               ${O2_DIR}/Detectors/Base/src/Propagator.cxx
               ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/Road.cxx)
  if(CONFIG_O2_ITS_TRAITS)
    target_sources(standalone_support PRIVATE
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/PrimaryVertexContext.cxx
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/Cluster.cxx
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/ClusterLines.cxx
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/TrackerTraitsCPU.cxx
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/VertexerTraits.cxx
        ${O2_DIR}/Detectors/ITSMFT/ITS/tracking/src/ROframe.cxx)
    target_link_libraries(standalone_support PUBLIC Boost::boost)
  endif()
endif()

if(CONFIG_FMT)
  target_link_libraries(standalone_support PUBLIC fmt::fmt)
  target_link_libraries(TPCFastTransformation PUBLIC fmt::fmt)
endif()

if(CONFIG_VC)
  target_link_libraries(standalone_support PUBLIC Vc::Vc)
  target_link_libraries(TPCFastTransformation PUBLIC Vc::Vc)
endif()

if(BUILD_EVENT_DISPLAY)
  if(CONFIG_GL3W)
    target_sources(GPUTrackingDisplay PRIVATE ${GPUTRACKING_DIR}/display/3rdparty/gl3w.c)
  endif()
  target_sources(GPUTracking PRIVATE ${GPUTRACKING_DIR}/display/3rdparty/HandMadeMath/HandMadeMathImpl.cxx)
  target_include_directories(GPUTracking SYSTEM PUBLIC ${GPUTRACKING_DIR}/display/3rdparty)
endif()

if(CONFIG_ROOT)
  target_link_libraries(standalone_support PUBLIC
                             ROOT::Core
                             ROOT::RIO
                             ROOT::Hist
                             ROOT::Gui
                             ROOT::Tree)
endif()
if(CONFIG_O2_EXTENSIONS)
  target_link_libraries(standalone_support PUBLIC Microsoft.GSL::GSL TPCFastTransformation)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(ca PUBLIC OpenMP::OpenMP_CXX)
  if (CMAKE_CXX_COMPILER STREQUAL "clang++")
    target_link_libraries(ca PUBLIC -fopenmp)
    target_link_libraries(GPUTracking PUBLIC -fopenmp)
  endif()
endif()

# Installation
install(TARGETS ca TPCFastTransformation standalone_support)
install(FILES "cmake/makefile" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${O2_DIR} ${CMAKE_INSTALL_PREFIX}/src)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/config.cmake ${CMAKE_INSTALL_PREFIX}/config.cmake)")
