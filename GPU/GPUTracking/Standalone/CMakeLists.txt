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

set(ALIGPU_BUILD_TYPE "Standalone")
add_definitions(-DGPUCA_STANDALONE -DHAVE_O2HEADERS)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(
  CMAKE_CXX_FLAGS
  "-O3 -march=native -ggdb -minline-all-stringops -ftracer -funroll-loops -fprefetch-loop-arrays -ffast-math -fno-stack-protector -Werror -Wall -Wextra -Wshadow -Wno-unused-function -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-write-strings"
  )
add_definitions(-DNDEBUG)

find_package(OpenMP REQUIRED)
find_package(GLFW NAMES glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Vc REQUIRED)
find_package(ROOT CONFIG REQUIRED)
find_package(fmt REQUIRED)
find_package(Clang REQUIRED)
find_package(LLVM REQUIRED)
find_package(X11 REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/dependencies ${CMAKE_MODULE_PATH})
find_package(O2GPU)
find_package(cub REQUIRED)

include_directories(.
                    SliceTracker
                    HLTHeaders
                    Merger
                    Base
                    Global
                    TRDTracking
                    ITS
                    dEdx
                    TPCConvert
                    DataCompression
                    Common
                    TPCFastTransformation
                    display
                    qa
                    ../../../Common/Constants/include
                    ../../../Common/MathUtils/include
                    ../../../DataFormats/common/include
                    ../../../Detectors/TPC/base/include
                    ../../../DataFormats/Detectors/TPC/include
                    ../../../DataFormats/common/include
                    ../../../Detectors/TRD/base/include
                    ../../../Detectors/TRD/base/src
                    ../../../Detectors/ITSMFT/ITS/tracking/include
                    ../../../Detectors/ITSMFT/ITS/tracking/cuda/include
                    ../../../Detectors/ITSMFT/ITS/tracking/cuda/src
                    ../../../DataFormats/Detectors/ITSMFT/ITS/include
                    ../../../DataFormats/Reconstruction/include
                    ../../../DataFormats/simulation/include
                    ../../../Detectors/Base/src
                    ../../../Detectors/Base/include
                    ../../../DataFormats/Detectors/Common/include)

add_subdirectory(../../ GPU)

add_executable(ca standalone.cxx utils/qconfig.cpp qa/genEvents.cpp
                  Base/GPUReconstructionTimeframe.cxx)

add_library(o2_support
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

target_sources(GPUTracking PRIVATE display/GPUDisplayBackendX11.cpp
               display/GPUDisplayBackendGlut.cpp)

target_link_libraries(o2_support
                      PUBLIC Vc::Vc
                             X11::X11
                             glfw
                             glut
                             ROOT::Core
                             ROOT::RIO
                             ROOT::Hist
                             ROOT::Gui)
target_link_libraries(GPUTracking PUBLIC TPCFastTransformation o2_support)

target_link_libraries(ca PUBLIC GPUTracking)
target_compile_definitions(
  ca PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

install(TARGETS ca GPUTracking TPCFastTransformation)
