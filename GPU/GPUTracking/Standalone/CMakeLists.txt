# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

# Some general CMake settings
cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(GPUTrackingStandalone)
include(FeatureSummary)
set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/dependencies ${CMAKE_MODULE_PATH})

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
  set(CMAKE_CXX_FLAGS "-O0 -ggdb")
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
  add_definitions(-DHAVE_O2HEADERS)
  if(NOT DEFINED MS_GSL_INCLUDE_DIR AND DEFINED ENV{MS_GSL_ROOT})
    set(MS_GSL_INCLUDE_DIR $ENV{MS_GSL_ROOT}/include)
  endif()
  find_package(ms_gsl REQUIRED)
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
include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Definitions
                    ${CMAKE_SOURCE_DIR}/DataTypes
                    ${CMAKE_SOURCE_DIR}/Base
                    ${CMAKE_SOURCE_DIR}/Base/opencl-common
                    ${CMAKE_SOURCE_DIR}/Common
                    ${CMAKE_SOURCE_DIR}/GPUUtils
                    ${CMAKE_SOURCE_DIR}/dEdx
                    ${CMAKE_SOURCE_DIR}/TPCConvert
                    ${CMAKE_SOURCE_DIR}/TPCFastTransformation
                    ${CMAKE_SOURCE_DIR}/display
                    ${CMAKE_SOURCE_DIR}/Global
                    ${CMAKE_SOURCE_DIR}/HLTHeaders
                    ${CMAKE_SOURCE_DIR}/Merger
                    ${CMAKE_SOURCE_DIR}/Refit
                    ${CMAKE_SOURCE_DIR}/qa
                    ${CMAKE_SOURCE_DIR}/SliceTracker
                    ${CMAKE_SOURCE_DIR}/DataCompression
                    ${CMAKE_SOURCE_DIR}/TRDTracking)

if(CONFIG_O2_EXTENSIONS)
include_directories(${CMAKE_SOURCE_DIR}/TPCClusterFinder
                    ${CMAKE_SOURCE_DIR}/ITS
                    ${CMAKE_SOURCE_DIR}/../../../Common/Field/include
                    ${CMAKE_SOURCE_DIR}/../../../Common/Constants/include
                    ${CMAKE_SOURCE_DIR}/../../../Common/MathUtils/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/common/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/Common/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/ITSMFT/ITS/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/TPC/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/TRD/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Headers/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/MemoryResources/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Reconstruction/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Reconstruction/src
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/simulation/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/Base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/Base/src
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/cuda/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/cuda/src
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/hip/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/hip/src
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/Raw/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TPC/base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TRD/base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TRD/base/src)
endif()

# Create main targets
add_subdirectory(../../ GPU)
add_library(standalone_support SHARED dummy.cxx)

target_link_libraries(GPUTracking PUBLIC TPCFastTransformation standalone_support dl)
target_link_libraries(ca PUBLIC GPUTracking)
target_link_libraries(standalone_support PUBLIC dl pthread)
target_compile_definitions(ca PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_compile_definitions(standalone_support PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

# Add all sources and dependencies to to support based on Config File
if(CONFIG_O2_EXTENSIONS)
  target_sources(standalone_support PRIVATE
               ../../../Common/Field/src/MagFieldFast.cxx
               ../../../DataFormats/Detectors/TPC/src/CompressedClusters.cxx
               ../../../DataFormats/simulation/src/MCCompLabel.cxx
               ../../../DataFormats/Reconstruction/src/TrackParametrization.cxx
               ../../../DataFormats/Reconstruction/src/TrackParametrizationWithError.cxx
               ../../../DataFormats/Reconstruction/src/Vertex.cxx
               ../../../DataFormats/Reconstruction/src/TrackLTIntegral.cxx
               ../../../DataFormats/Reconstruction/src/TrackParametrization.cxx
               ../../../DataFormats/Reconstruction/src/TrackParametrizationWithError.cxx
               ../../../Detectors/TRD/base/src/GeometryBase.cxx
               ../../../Detectors/Base/src/MatLayerCylSet.cxx
               ../../../Detectors/Base/src/MatLayerCyl.cxx
               ../../../Detectors/Base/src/Ray.cxx
               ../../../Detectors/Base/src/Propagator.cxx
               ../../../Detectors/ITSMFT/ITS/tracking/src/Road.cxx)
  if(CONFIG_O2_ITS_TRAITS)
    target_sources(standalone_support PRIVATE
        ../../../Detectors/ITSMFT/ITS/tracking/src/PrimaryVertexContext.cxx
        ../../../Detectors/ITSMFT/ITS/tracking/src/Cluster.cxx
        ../../../Detectors/ITSMFT/ITS/tracking/src/ClusterLines.cxx
        ../../../Detectors/ITSMFT/ITS/tracking/src/TrackerTraitsCPU.cxx
        ../../../Detectors/ITSMFT/ITS/tracking/src/VertexerTraits.cxx
        ../../../Detectors/ITSMFT/ITS/tracking/src/ROframe.cxx)
    target_link_libraries(standalone_support PUBLIC Boost::boost)
  endif()
endif()

if(CONFIG_FMT)
  target_link_libraries(standalone_support PUBLIC fmt::fmt)
endif()

if(CONFIG_VC)
  target_link_libraries(standalone_support PUBLIC Vc::Vc)
endif()

if(BUILD_EVENT_DISPLAY)
  target_link_libraries(standalone_support PUBLIC glfw)
  if(CONFIG_GL3W)
    target_sources(GPUTracking PRIVATE display/3rdparty/gl3w.c)
    target_compile_definitions(GPUTracking PUBLIC GPUCA_DISPLAY_GL3W)
  endif()
  target_sources(GPUTracking PRIVATE display/3rdparty/HandMadeMath/HandMadeMathImpl.cxx)
  target_include_directories(GPUTracking SYSTEM PUBLIC display/3rdparty)
endif()

if(CONFIG_ROOT)
  target_link_libraries(standalone_support PUBLIC
                             ROOT::Core
                             ROOT::RIO
                             ROOT::Hist
                             ROOT::Gui)
endif()
if(CONFIG_O2_EXTENSIONS)
  target_link_libraries(standalone_support PUBLIC ms_gsl::ms_gsl)
endif()

if(CONFIG_VC)
  target_link_libraries(TPCFastTransformation PUBLIC Vc::Vc)
endif()

if(OpenMP_CXX_FOUND)
  target_link_libraries(ca PUBLIC OpenMP::OpenMP_CXX)
  if (CMAKE_CXX_COMPILER STREQUAL "clang++")
    target_link_libraries(ca PUBLIC -fopenmp)
    target_link_libraries(GPUTracking PUBLIC -fopenmp)
  endif()
endif()

# Installation
install(TARGETS ca GPUTracking TPCFastTransformation standalone_support)
install(FILES "cmake/makefile" DESTINATION "${CMAKE_INSTALL_PREFIX}")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR} ${CMAKE_INSTALL_PREFIX}/src)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/config.cmake ${CMAKE_INSTALL_PREFIX}/config.cmake)")
