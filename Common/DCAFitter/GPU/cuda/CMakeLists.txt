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

o2_add_library(DCAFitterCUDAAPI
               TARGETVARNAME targetName
               SOURCES DCAFitterNKernels.cu
               # src/FwdDCAFitterN.cxx
               PUBLIC_INCLUDE_DIRECTORIES .
                                          ../../include
               PUBLIC_LINK_LIBRARIES O2::MathUtils
                                     O2::CommonUtils
                                     O2::ReconstructionDataFormats
                                     O2::DataFormatsParameters
                                     O2::GPUCommon
                                     O2::DetectorsBase
               PRIVATE_LINK_LIBRARIES O2::GPUTrackingCUDAExternalProvider)

set_property(TARGET ${targetName} PROPERTY CUDA_SEPARABLE_COMPILATION ON)