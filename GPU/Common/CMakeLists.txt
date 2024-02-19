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

set(HDRS_INSTALL
    GPUCommonAlgorithm.h
    GPUCommonDef.h
    GPUCommonDefAPI.h
    GPUCommonDefSettings.h
    GPUCommonConstants.h
    GPUCommonLogger.h
    GPUCommonMath.h
    GPUCommonRtypes.h
    GPUCommonArray.h
    GPUCommonTypeTraits.h
    GPUCommonTransform3D.h
    GPUROOTSMatrixFwd.h
    GPUROOTCartesianFwd.h
    GPUDebugStreamer.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ../GPUTracking/utils/EmptyFile.cxx
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::FrameworkLogger)
  target_include_directories(${targetName}
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                    $<INSTALL_INTERFACE:include/GPU>)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB
                             GPUCA_TPC_GEOMETRY_O2 GPUCA_HAVE_O2HEADERS)

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
  if (HIP_ENABLED)
    o2_add_test(SMatrixImpHIP NAME test_SMatrixImpHIP
                SOURCES test/testSMatrixImp.cu
                HIPIFIED test
                PUBLIC_LINK_LIBRARIES O2::${MODULE}
                                      O2::MathUtils
                                      ROOT::Core
                COMPONENT_NAME GPU
                LABELS gpu)
  endif()
  install(FILES ${HDRS_INSTALL} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_ALIROOT_LIB)

  set(SRCS ${SRCS} ../GPUTracking/utils/EmptyFile.cxx)

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS})

  # Additional compilation flags
  set_target_properties(Ali${MODULE} PROPERTIES COMPILE_FLAGS "")

  # System dependent: Modify the way the library is build
  if(${CMAKE_SYSTEM} MATCHES Darwin)
    set_target_properties(Ali${MODULE}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(${CMAKE_SYSTEM} MATCHES Darwin)

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS_INSTALL} DESTINATION include)
endif()
