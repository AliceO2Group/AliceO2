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

set(MODULE TPCFastTransformation)

set(SRCS
    TPCFastTransformGeo.cxx
    TPCFastSpaceChargeCorrection.cxx
    TPCFastSpaceChargeCorrectionMap.cxx
    TPCFastTransform.cxx
    TPCFastTransformPOD.cxx
)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(SRCS ${SRCS}
      CorrectionMapsHelper.cxx
      ChebyshevFit1D.cxx
      devtools/IrregularSpline1D.cxx
      devtools/IrregularSpline2D3D.cxx
      devtools/SemiregularSpline2D3D.cxx
      devtools/IrregularSpline2D3DCalibrator.cxx
  )
endif()

string(REPLACE ".cxx" ".h" HDRS_CINT "${SRCS}")
set(HDRS_CINT ${HDRS_CINT}
              devtools/RegularSpline1D.h
              CorrectionMapsTypes.h
)

if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 SOURCES ${SRCS}
                 PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}
                                            devtools
                                            ${CMAKE_SOURCE_DIR}/GPU/Common
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       O2::GPUUtils
                                       O2::TPCSpaceCharge
                                       Vc::Vc
                                       ROOT::Core ROOT::Matrix ROOT::Tree ROOT::Gpad ROOT::Minuit
                             )
  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT}
                            LINKDEF TPCFastTransformationLinkDef_O2.h)

  file(COPY ${HDRS_CINT} DESTINATION ${CMAKE_BINARY_DIR}/stage/include/GPU)

  o2_add_test(${MODULE}
              PUBLIC_LINK_LIBRARIES O2::${MODULE}
              SOURCES test/testSplines.cxx
              COMPONENT_NAME GPU
              LABELS gpu)

  o2_add_test(MultivarPolynomials
              COMPONENT_NAME GPU
              PUBLIC_LINK_LIBRARIES O2::${MODULE}
              SOURCES test/testMultivarPolynomials.cxx
              ENVIRONMENT O2_ROOT=${CMAKE_BINARY_DIR}/stage
              LABELS gpu
              CONFIGURATIONS RelWithDebInfo Release MinSizeRel)

  foreach(m
          SplineDemo.C
          SplineRecoveryDemo.C
          fastTransformQA.C
          TPCFastTransformInit.C
          generateTPCCorrectionNTuple.C)
    o2_add_test_root_macro(macro/${m}
                           PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation
                                                 O2::DataFormatsTPC
                                                 O2::TPCSimulation
                                                 O2::TPCReconstruction
                                                 O2::TPCCalibration
                                                 O2::SpacePoints
                          PUBLIC_INCLUDE_DIRECTORIES
                                                 ${CMAKE_BINARY_DIR}/stage/include
                           LABELS gpu tpc COMPILE_ONLY)
  endforeach()
    foreach(m
          IrregularSpline1DTest.C
          IrregularSpline2D3DCalibratorTest.C
          IrregularSpline2D3DTest.C
          RegularSpline1DTest.C
          SemiregularSpline2D3DTest.C
          ChebFitTest.C )
    o2_add_test_root_macro(devtools/${m}
                           PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation
                                                 O2::DataFormatsTPC
                                                 O2::TPCSimulation
                                                 O2::TPCReconstruction
                                                 PUBLIC_INCLUDE_DIRECTORIES
                                                 ${CMAKE_BINARY_DIR}/stage/include
                           LABELS gpu tpc)
  endforeach()

  install(FILES macro/TPCFastTransformInit.C
          DESTINATION share/macro/)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
  set(targetName ${MODULE})
  target_include_directories(${targetName} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_link_libraries(${targetName} PUBLIC GPUUtils)
  install(TARGETS ${MODULE})
endif()

install(FILES ${HDRS_CINT} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)

#
# FIXME: this one is misplaced : it depends (at least) on TPCSimulation which is
# built after GPU...
#
# o2_add_test_root_macro(macro/generateTPCCorrectionNTuple.C
# PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation O2::DataFormatsTPC
# O2::TPCSimulation PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/stage/include
# LABELS gpu tpc)
