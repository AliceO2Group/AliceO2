# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

if(CUDA_ENABLED)
  add_subdirectory(cuda)
  o2_add_executable(memory-benchmark-cuda
                  SOURCES benchmark.cxx
                  PUBLIC_LINK_LIBRARIES O2::CUDAbenchmark
                  TARGETVARNAME targetName)
endif()

if(HIP_ENABLED)
  add_subdirectory(hip)
  set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
  set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc")

  o2_add_executable(memory-benchmark-hip
                    SOURCES benchmark.cxx
                    PUBLIC_LINK_LIBRARIES O2::HIPbenchmark
                                          hip::host
                    TARGETVARNAME targetName)

if(HIP_AMDGPUTARGET)
# Need to add gpu target also to link flags due to gpu-rdc option
target_link_options(${targetName} PUBLIC --amdgpu-target=${HIP_AMDGPUTARGET})
endif()
endif()