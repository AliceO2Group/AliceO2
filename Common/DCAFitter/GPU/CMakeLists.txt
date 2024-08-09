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
# o2_add_library(DCAFitterCUDA
#                TARGETVARNAME targetName
#                SOURCES DCAFitterN.cu
#                # src/FwdDCAFitterN.cxx
#                PUBLIC_INCLUDE_DIRECTORIES ../include
#                PUBLIC_LINK_LIBRARIES O2::MathUtils
#                                      O2::ReconstructionDataFormats
#                                      O2::DetectorsBase)

# o2_add_test(DCAFitterNCUDA NAME testDCAFitterNCUDA
#             SOURCES test/testDCAFitterNCUDA.cu
#             COMPONENT_NAME DCAFitterCUDA
#             PUBLIC_LINK_LIBRARIES O2::DCAFitter
#             COMPONENT_NAME GPU
#             LABELS gpu vertexing)
endif()
# if (HIP_ENABLED)
# o2_add_test(DCAFitterNHIP NAME testDCAFitterNHIP
#             SOURCES test/testDCAFitterNCUDA.cu
#             HIPIFIED test
#             PUBLIC_LINK_LIBRARIES O2::DCAFitterHIP
#             COMPONENT_NAME GPU
#             LABELS gpu vertexing)
# endif()