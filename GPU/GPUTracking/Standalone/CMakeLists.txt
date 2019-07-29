# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
include(FeatureSummary)

set(CMAKE_INSTALL_BINDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_LIBDIR "${CMAKE_INSTALL_PREFIX}")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}")

if(NOT EXISTS "${CMAKE_BINARY_DIR}/config.cmake")
  file(COPY "${CMAKE_SOURCE_DIR}/cmake/config.cmake" DESTINATION "${CMAKE_BINARY_DIR}")
endif()
include("${CMAKE_BINARY_DIR}/config.cmake")

install(FILES "cmake/makefile" DESTINATION "${CMAKE_INSTALL_PREFIX}")

set(ALIGPU_BUILD_TYPE "Standalone")
add_definitions(-DGPUCA_STANDALONE -DHAVE_O2HEADERS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(BUILD_DEBUG)
  set(CMAKE_CXX_FLAGS "-O0 -ggdb")
else()
  set(CMAKE_CXX_FLAGS "-O3 -march=native -ggdb -minline-all-stringops -ftracer -funroll-loops -fprefetch-loop-arrays -ffast-math -fno-stack-protector -Werror -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-write-strings")
  add_definitions(-DNDEBUG)
endif()
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -rdynamic -Wl,--no-undefined")

if(CONFIG_OPENMP)
  find_package(OpenMP REQUIRED)
else()
  set(OpenMP_FOUND OFF)
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
else()
  set(OpenGL_FOUND OFF)
endif()

if(CONFIG_O2)
  add_definitions(-DGPUCA_TPC_GEOMETRY_O2)
endif()

find_package(ROOT CONFIG REQUIRED)
find_package(Clang REQUIRED)
find_package(LLVM REQUIRED)
find_package(X11 REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/dependencies ${CMAKE_MODULE_PATH})
find_package(O2GPU)

include_directories(${CMAKE_SOURCE_DIR}
                    ${CMAKE_SOURCE_DIR}/Base
                    ${CMAKE_SOURCE_DIR}/Base/opencl-common
                    ${CMAKE_SOURCE_DIR}/Common
                    ${CMAKE_SOURCE_DIR}/DataCompression
                    ${CMAKE_SOURCE_DIR}/dEdx
                    ${CMAKE_SOURCE_DIR}/display
                    ${CMAKE_SOURCE_DIR}/Global
                    ${CMAKE_SOURCE_DIR}/HLTHeaders
                    ${CMAKE_SOURCE_DIR}/ITS
                    ${CMAKE_SOURCE_DIR}/Merger
                    ${CMAKE_SOURCE_DIR}/qa
                    ${CMAKE_SOURCE_DIR}/SliceTracker
                    ${CMAKE_SOURCE_DIR}/TPCConvert
                    ${CMAKE_SOURCE_DIR}/TPCFastTransformation
                    ${CMAKE_SOURCE_DIR}/TRDTracking
                    ${CMAKE_SOURCE_DIR}/../../../Common/Constants/include
                    ${CMAKE_SOURCE_DIR}/../../../Common/MathUtils/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/common/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/Common/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/ITSMFT/ITS/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Detectors/TPC/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/Reconstruction/include
                    ${CMAKE_SOURCE_DIR}/../../../DataFormats/simulation/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/Base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/Base/src
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/cuda/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/ITSMFT/ITS/tracking/cuda/src
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TPC/base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TRD/base/include
                    ${CMAKE_SOURCE_DIR}/../../../Detectors/TRD/base/src)

add_subdirectory(../../ GPU)

add_executable(ca standalone.cxx qa/genEvents.cxx
                  Base/GPUReconstructionTimeframe.cxx)

add_library(o2_support SHARED
  ../../..//DataFormats/simulation/src/MCCompLabel.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/PrimaryVertexContext.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/Cluster.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/ClusterLines.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/TrackerTraitsCPU.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/VertexerTraits.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/ROframe.cxx
  ../../..//Detectors/ITSMFT/ITS/tracking/src/Road.cxx
  ../../..//Detectors/TRD/base/src/TRDGeometryBase.cxx
  ../../..//Detectors/Base/src/MatLayerCylSet.cxx
  ../../..//Detectors/Base/src/MatLayerCyl.cxx
  ../../..//Detectors/Base/src/Ray.cxx)

add_library(standalone_config SHARED utils/qconfig.cxx)

target_sources(GPUTracking PRIVATE display/GPUDisplayBackendX11.cxx
               display/GPUDisplayBackendGlut.cxx)

target_link_libraries(o2_support
                      PUBLIC Vc::Vc
                             X11::X11
                             glfw
                             glut
                             ROOT::Core
                             ROOT::RIO
                             ROOT::Hist
                             ROOT::Gui)
if(CONFIG_VC)
  target_link_libraries(TPCFastTransformation PUBLIC Vc::Vc)
endif()
target_link_libraries(GPUTracking PUBLIC TPCFastTransformation o2_support standalone_config dl)

target_link_libraries(ca PUBLIC GPUTracking)
target_compile_definitions(ca PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
target_compile_definitions(standalone_config PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

install(TARGETS ca GPUTracking TPCFastTransformation o2_support standalone_config)
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_SOURCE_DIR} ${CMAKE_INSTALL_PREFIX}/src)")
install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_BINARY_DIR}/config.cmake ${CMAKE_INSTALL_PREFIX}/config.cmake)")
