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

set(MODULE GPUCommon)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ../GPUTracking/utils/EmptyFile.cxx
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::FrameworkLogger)
  target_include_directories(${targetName}
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                    $<INSTALL_INTERFACE:include/GPU>)

  # cuda test, only compile if CUDA
  if(CUDA_ENABLED)
    o2_add_test(GPUsortCUDA NAME test_GPUsortCUDA
                SOURCES test/testGPUsortCUDA.cu
                PUBLIC_LINK_LIBRARIES O2::${MODULE}
                COMPONENT_NAME GPU
                LABELS gpu)
    o2_add_test(SMatrixImpCUDA NAME test_SMatrixImpCUDA
                SOURCES test/testSMatrixImp.cu
                PUBLIC_LINK_LIBRARIES O2::${MODULE}
                                      O2::MathUtils
                                      ROOT::Core
                COMPONENT_NAME GPU
                LABELS gpu)
  endif()
#  if (HIP_ENABLED)
#    o2_add_test(SMatrixImpHIP NAME test_SMatrixImpHIP
#                SOURCES test/testSMatrixImp.cu
#                HIPIFIED test
#                PUBLIC_LINK_LIBRARIES O2::${MODULE}
#                                      O2::MathUtils
#                                      ROOT::Core
#                COMPONENT_NAME GPU
#                LABELS gpu)
#  endif()
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU FILES_MATCHING PATTERN "*.h" PATTERN "test" EXCLUDE)
