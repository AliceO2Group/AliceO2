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

if(HIP_ENABLED)
  message(STATUS "Building ITS HIP tracker")
  set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")
  o2_add_hipified_library(ITStrackingHIP
                 SOURCES ../cuda/ClusterLinesGPU.cu
                         ../cuda/Context.cu
                         ../cuda/TimeFrameGPU.cu
                         ../cuda/Stream.cu
                         ../cuda/TrackerTraitsGPU.cu
                         ../cuda/TracerGPU.cu
                         ../cuda/VertexerTraitsGPU.cu
                         ../cuda/Utils.cu
                 PUBLIC_INCLUDE_DIRECTORIES ../
                 PRIVATE_INCLUDE_DIRECTORIES
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base
                       ${CMAKE_SOURCE_DIR}/GPU/GPUTracking/Base/hip
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
                                       O2::GPUTracking
                                       O2::SimulationDataFormat
                                       O2::ReconstructionDataFormats
                                       hip::host
                                       hip::device
                                       hip::hipcub
                 TARGETVARNAME targetName)

  if(O2_HIP_CMAKE_LINK_FLAGS)
    # Need to add gpu target also to link flags due to gpu-rdc option
    target_link_options(${targetName} PRIVATE ${O2_HIP_CMAKE_LINK_FLAGS})
  endif()
endif()
