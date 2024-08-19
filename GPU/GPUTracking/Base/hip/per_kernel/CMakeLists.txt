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

add_library(GPUTrackingHIPKernels OBJECT $<JOIN:$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<LIST:TRANSFORM,$<TARGET_PROPERTY:O2_GPU_KERNELS,O2_GPU_KERNEL_NAMES>,REPLACE,[^A-Za-z0-9]+,_>,PREPEND,${O2_GPU_KERNEL_WRAPPER_FOLDER}/krnl_>,APPEND,.hip.cxx>, >)
set(CMAKE_CXX_COMPILER ${hip_HIPCC_EXECUTABLE})
set(CMAKE_CXX_FLAGS "${GPU_RTC_FLAGS} ${GPU_RTC_FLAGS_ARCH} --genco")
unset(CMAKE_CXX_FLAGS_${CMAKE_BUILD_TYPE_UPPER})
