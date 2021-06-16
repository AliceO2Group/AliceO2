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

message(STATUS "Building ITS CUDA tracker")

o2_add_library(ITStrackingCUDA
               SOURCES src/ClusterLinesGPU.cu
                       src/Context.cu
                       src/DeviceStoreNV.cu
                       src/DeviceStoreVertexerGPU.cu
                       src/Stream.cu
                       src/TrackerTraitsNV.cu
                       src/VertexerTraitsGPU.cu
                       src/Utils.cu
               PUBLIC_LINK_LIBRARIES O2::ITStracking
                                     O2::SimConfig
                                     O2::SimulationDataFormat
               TARGETVARNAME targetName)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(
  ${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)
