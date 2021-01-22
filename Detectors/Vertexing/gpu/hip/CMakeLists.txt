# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")
# hipify-perl
set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
set(CU_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/src/SVertexer.cu)
set(HIP_KERNEL "${CMAKE_CURRENT_SOURCE_DIR}/src/SVertexer.hip.cxx")

if(EXISTS ${HIPIFY_EXECUTABLE})

if(EXISTS ${CU_KERNEL})
  message(STATUS "CU_KERNEL: " ${CU_KERNEL})
endif()
if(EXISTS ${HIP_KERNEL})
  message(STATUS "HIP_KERNEL: " ${HIP_KERNEL})
endif()

execute_process(COMMAND bash ${HIPIFY_EXECUTABLE} "${CU_KERNEL} > ${HIP_KERNEL}" RESULT_VARIABLE CMD_ERROR OUTPUT_FILE CMD_OUTPUT)
message(STATUS "CMD_ERROR:" ${CMD_ERROR})
message(FATAL_ERROR "CMD_OUTPUT:" ${CMD_OUTPUT})
endif()

o2_add_library(DetectorsVertexingHIP
               SOURCES src/SVertexer.hip.cxx
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     O2::ReconstructionDataFormats
                                     O2::DetectorsVertexing
                                     hip::host
                                     hip::device
               TARGETVARNAME targetName)
               target_compile_definitions(
                ${targetName} PRIVATE $<TARGET_PROPERTY:O2::DetectorsVertexing,COMPILE_DEFINITIONS>)
              
  if(HIP_AMDGPUTARGET)
    # Need to add gpu target also to link flags due to gpu-rdc option
    target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
  endif()