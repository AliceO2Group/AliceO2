# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(HDRS_INSTALL ../GPUbenchmark.h)

o2_add_library(GPUbenchmarkCUDA
               SOURCES GPUbenchmark.cu
               PUBLIC_INCLUDE_DIRECTORIES ../
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
               TARGETVARNAME targetName)

               set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

install(FILES ${HDRS_INSTALL} DESTINATION include/GPU/GPUBenchmark)