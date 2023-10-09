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
set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} ${O2_HIP_CMAKE_CXX_FLAGS} -fgpu-rdc -isystem /opt/rocm/include -fPIC")

o2_add_hipified_executable(gpu-memory-benchmark-hip
                           SOURCES ../cuda/benchmark.cu
                                   ../cuda/Kernels.cu
                           PUBLIC_LINK_LIBRARIES hip::host
                                                 Boost::program_options
                           TARGETVARNAME targetName)