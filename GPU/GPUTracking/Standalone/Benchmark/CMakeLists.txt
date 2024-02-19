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

message(STATUS "Building GPU Standalone Benchmark")

set(SRCS
    standalone.cxx
    ../../utils/qconfig.cxx
    ../../Base/GPUReconstructionTimeframe.cxx)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
    o2_add_executable(standalone-benchmark
                      COMPONENT_NAME gpu
                      TARGETVARNAME targetName
                      PUBLIC_LINK_LIBRARIES O2::GPUO2Interface O2::GPUTracking
                      SOURCES ${SRCS})

    target_compile_definitions(${targetName} PUBLIC $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    add_executable(ca ${SRCS})
    set(targetName ca)
endif()

if(ROOT_FOUND)
  target_sources(${targetName} PRIVATE ../../qa/genEvents.cxx)
endif()
