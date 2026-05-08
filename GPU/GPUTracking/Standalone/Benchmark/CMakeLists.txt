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
                      PUBLIC_LINK_LIBRARIES O2::GPUO2Interface O2::GPUTracking O2::TPCReconstruction
                      SOURCES ${SRCS})
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
    add_executable(ca ${SRCS})
    set(targetName ca)
    target_link_libraries(${targetName} PUBLIC GPUTracking standalone_support2)
    install(TARGETS ca)
endif()

install(DIRECTORY ../tools DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/GPU)
install(DIRECTORY ../../Definitions/Parameters/ DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/GPU/tools FILES_MATCHING REGEX "\\.(python|sh|cmake)")
target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)

if(GPUCA_DETERMINISTIC_MODE GREATER_EQUAL ${GPUCA_DETERMINISTIC_MODE_MAP_NO_FAST_MATH} AND GPUCA_DETERMINISTIC_NO_FTZ)
target_compile_definitions(${targetName} PRIVATE GPUCA_DETERMINISTIC_NO_FTZ)
endif()

if(ROOT_FOUND)
  target_sources(${targetName} PRIVATE ../../qa/genEvents.cxx)
endif()
