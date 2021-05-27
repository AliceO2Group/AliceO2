# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(HDRS_INSTALL ../Shared/Kernels.h)

set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")
 
# Hipify-perl
set(HIPIFY_EXECUTABLE "/opt/rocm/bin/hipify-perl")
set(HIP_KERNEL "Kernels.hip.cxx")

set(CU_KERNEL ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/Kernels.cu)
set(HIP_KERNEL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${HIP_KERNEL}")

if(EXISTS ${HIPIFY_EXECUTABLE})

# Generate on-the-fly the HIP kernel
message("Generating HIP kernel code ....")
execute_process(COMMAND /bin/sh -c "${HIPIFY_EXECUTABLE} ${CU_KERNEL} | tee ${HIP_KERNEL_PATH}")
o2_add_library(HIPbenchmark
               SOURCES Kernels.hip.cxx
               PUBLIC_INCLUDE_DIRECTORIES ../Shared
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                     hip::host
                                     hip::device
               TARGETVARNAME targetName)

elseif()
  message(STATUS "Could not generate ${HIP_KERNEL} HIP kernel, skipping...")
endif()