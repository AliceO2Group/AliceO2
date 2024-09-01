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

# set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fgpu-rdc")
# o2_add_hipified_library(DCAFitterNKernelsHIP
#                TARGETVARNAME targetName
#                SOURCES ../cuda/DCAFitterNKernels.cu
#                PUBLIC_INCLUDE_DIRECTORIES ../cuda
#                PRIVATE_LINK_LIBRARIES O2::DCAFitter
#                                       O2::MathUtils
#                                       O2::ReconstructionDataFormats
#                                       O2::GPUCommon
#                                       O2::DetectorsBase
#                                       O2::GPUTrackingHIPExternalProvider)
# target_compile_options(DCAFitterNKernelsHIP PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)
# target_link_options(DCAFitterNKernelsHIP PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-fgpu-rdc>)

# o2_add_hipified_library(DCAFitterAPIHIP
#                SOURCES ../cuda/DCAFitterGPUAPI.cu
#                # src/FwdDCAFitterN.cxx
#                PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_SOURCE_DIR}/Common/DCAFitter/GPU
#                                           ../cuda
#                PUBLIC_LINK_LIBRARIES O2::GPUCommon
#                                      O2::ReconstructionDataFormats
#                PRIVATE_LINK_LIBRARIES O2::DCAFitter
#                                       O2::DCAFitterNKernelsHIP)

# o2_add_test(DCAFitterNHIP
#             SOURCES ../cuda/test/testDCAFitterNGPU.cxx
#             PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
#                                   O2::DCAFitterAPIHIP
#                                   O2::DCAFitter
#                                   ROOT::Core
#                                   ROOT::Physics
#             HIPIFIED test
#             COMPONENT_NAME gpu               
#             LABELS vertexing
#             ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
#             VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})