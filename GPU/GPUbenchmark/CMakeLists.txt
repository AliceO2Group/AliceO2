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

set(HDRS_INSTALL ../Shared/Kernels.h)

if(CUDA_ENABLED)
  # add_subdirectory(cuda)
  o2_add_executable(gpu-memory-benchmark-cuda
                  SOURCES benchmark.cxx
                          cuda/Kernels.cu
                  PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                        Boost::program_options
                  TARGETVARNAME targetName)
endif()

if(HIP_ENABLED)
  # Hipify-perl
  set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")

  set(HIP_KERNEL "Kernels.hip.cxx")
  set(CU_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/cuda/Kernels.cu)
  set(HIP_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/hip/${HIP_KERNEL}")

  if(EXISTS ${HIPIFY_EXECUTABLE})
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CU_KERNEL})
    message("Generating HIP kernel code ...")
    execute_process(COMMAND /bin/sh -c "${HIPIFY_EXECUTABLE} ${CU_KERNEL} > ${HIP_KERNEL_PATH}")
  elseif()
    message(STATUS "Could not generate ${HIP_KERNEL} HIP kernel, skipping...")
  endif()

  set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")

  o2_add_executable(gpu-memory-benchmark-hip
                    SOURCES benchmark.cxx
                            hip/Kernels.hip.cxx
                    PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                          hip::host
                                          Boost::program_options
                    TARGETVARNAME targetName)

  if(HIP_AMDGPUTARGET)
    # Need to add gpu target also to link flags due to gpu-rdc option
    target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
  endif()
endif()