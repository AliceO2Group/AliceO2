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

if(CUDA_ENABLED)
message(STATUS "Building GPU CUDA benchmark")
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
  o2_add_executable(gpu-memory-benchmark-cuda
                  SOURCES benchmark.cu
                          Kernels.cu
                  PUBLIC_LINK_LIBRARIES Boost::program_options
                                        ROOT::Tree
                  TARGETVARNAME targetName)
                  set_target_properties(${targeName} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()