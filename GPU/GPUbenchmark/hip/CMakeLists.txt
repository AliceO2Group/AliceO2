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

message(STATUS "Building GPU HIP benchmark")
# Hipify-perl to generate HIP sources
set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
file(GLOB CUDA_SOURCES_FULL_PATH "../cuda/*.cu")
foreach(file ${CUDA_SOURCES_FULL_PATH})
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${file})
  get_filename_component(CUDA_SOURCE ${file} NAME)
  string(REPLACE ".cu" "" CUDA_SOURCE_NAME ${CUDA_SOURCE})
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip.cxx
    COMMAND ${HIPIFY_EXECUTABLE} --quiet-warnings ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE} | sed '1{/\#include \"hip\\/hip_runtime.h\"/d}' > ${CMAKE_CURRENT_SOURCE_DIR}/${CUDA_SOURCE_NAME}.hip.cxx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/${CUDA_SOURCE}
  )
endforeach()

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")
o2_add_executable(gpu-memory-benchmark-hip
                  SOURCES benchmark.hip.cxx
                          Kernels.hip.cxx
                  PUBLIC_LINK_LIBRARIES hip::host
                                        Boost::program_options
                                        ROOT::Tree
                  TARGETVARNAME targetName)
if(O2_HIP_CMAKE_LINK_FLAGS)
  # Need to add gpu target also to link flags due to gpu-rdc option
  target_link_options(${targetName} PUBLIC ${O2_HIP_CMAKE_LINK_FLAGS})
endif()
