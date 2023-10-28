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

# CUDA
if(CUDA_ENABLED)
find_package(CUDAToolkit)
message(STATUS "Building ITS CUDA tracker")

o2_add_library(ITStrackingCUDA
               SOURCES ClusterLinesGPU.cu
                       Context.cu
                       Stream.cu
                       TrackerTraitsGPU.cu
                       TimeFrameGPU.cu
                       TracerGPU.cu
                       VertexerTraitsGPU.cu
                       Utils.cu
               PUBLIC_INCLUDE_DIRECTORIES ../
               PRIVATE_INCLUDE_DIRECTORIES
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/cuda
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/DataTypes
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Global               # "GPUErrors.h"
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/SliceTracker
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TRDTracking          # GPUTRDTracker.h
                       ${CMAKE_SOURCE_DIR}/DataFormats/Reconstruction/src
                       ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TPC/include    # "DataFormatsTPC/dEdxInfo.h"
                       ${CMAKE_SOURCE_DIR}/DataFormats/Detectors/TRD/include    # "DataFormatsTRD/Constants.h"
                       ${CMAKE_SOURCE_DIR}/Detectors/Base/src
                       ${CMAKE_SOURCE_DIR}/Detectors/TRD/base/include           # "TRDBase/GeometryFlat.h"
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Definitions
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Merger
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TPCConvert
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/DataCompression
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/ITS
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/TPCClusterFinder
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Refit
               PUBLIC_LINK_LIBRARIES O2::ITStracking
                                     O2::SimConfig
                                     O2::SimulationDataFormat
                                     O2::ReconstructionDataFormats
                                     CUDA::nvToolsExt # TODO: change to CUDA::nvtx3 when CMake bump >= 3.25
               TARGETVARNAME targetName)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)
set_target_cuda_arch(${targetName})

endif()