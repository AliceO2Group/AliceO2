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

set(CMAKE_HIP_FLAGS "${CMAKE_HIP_FLAGS} -fgpu-rdc")
o2_add_hipified_library(DCAFitterHIP
                        SOURCES ../cuda/DCAFitterN.cu
                        PUBLIC_INCLUDE_DIRECTORIES ../../include
                        PUBLIC_LINK_LIBRARIES O2::MathUtils
                                              O2::ReconstructionDataFormats
                                              O2::DetectorsBase
                                              hip::host
                        PRIVATE_LINK_LIBRARIES O2::GPUTrackingHIPExternalProvider
                        TARGETVARNAME targetNAme)

o2_add_test(DCAFitterNHIP
            SOURCES ../cuda/test/testDCAFitterNGPU.cxx
            PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                  O2::DCAFitterHIP
                                  O2::DCAFitter
                                  ROOT::Core
                                  ROOT::Physics
            HIPIFIED test
            COMPONENT_NAME gpu
            LABELS vertexing
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})