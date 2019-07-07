# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

find_package(cub)
set_package_properties(cub PROPERTIES TYPE REQUIRED)

message(STATUS "Building ITS CUDA tracker")

o2_add_library(ITStrackingCUDA
               SOURCES src/Context.cu
                       src/DeviceStoreNV.cu
                       src/Stream.cu
                       src/TrackerTraitsNV.cu
                       src/VertexerTraitsGPU.cu
                       src/Utils.cu
               PUBLIC_LINK_LIBRARIES O2::ITStracking cub::cub
               TARGETVARNAME targetName)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_compile_definitions(
  ${targetName} PRIVATE $<TARGET_PROPERTY:O2::ITStracking,COMPILE_DEFINITIONS>)
