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

o2_add_library(DCAFitterCUDA
               TARGETVARNAME targetName
               SOURCES DCAFitterN.cu
                       GPUInterface.cu
               PUBLIC_INCLUDE_DIRECTORIES ../../include
               PUBLIC_INCLUDE_DIRECTORIES ../
               PUBLIC_LINK_LIBRARIES O2::MathUtils
                                     O2::ReconstructionDataFormats
                                     O2::DetectorsBase
               PRIVATE_LINK_LIBRARIES O2::GPUTrackingCUDAExternalProvider)
set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)
# add_compile_options(-lineinfo)

o2_add_test(DCAFitterNCUDA
            SOURCES test/testDCAFitterNGPU.cxx
            PUBLIC_LINK_LIBRARIES O2::ReconstructionDataFormats
                                  O2::DCAFitterCUDA
                                  O2::DCAFitter
                                  ROOT::Core
                                  ROOT::Physics
            COMPONENT_NAME gpu
            LABELS vertexing
            ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
            VMCWORKDIR=${CMAKE_BINARY_DIR}/stage/${CMAKE_INSTALL_DATADIR})