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
    SplineSpec.cxx
    Spline.cxx
    SplineHelper.cxx
    Spline1DSpec.cxx
    Spline1D.cxx
    Spline1DHelperOld.cxx
    Spline2DSpec.cxx
    Spline2D.cxx
    ChebyshevFit1D.cxx
    TPCFastTransformGeo.cxx
    TPCFastSpaceChargeCorrection.cxx
    TPCFastSpaceChargeCorrectionMap.cxx
    TPCFastTransform.cxx
    CorrectionMapsHelper.cxx
)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(SRCS ${SRCS}
      Spline1DHelper.cxx
      Spline2DHelper.cxx
      SymMatrixSolver.cxx
      BandMatrixSolver.cxx
      MultivariatePolynomial.cxx
      MultivariatePolynomialHelper.cxx
      NDPiecewisePolynomials.cxx
      devtools/IrregularSpline1D.cxx
      devtools/IrregularSpline2D3D.cxx
      devtools/SemiregularSpline2D3D.cxx
      devtools/IrregularSpline2D3DCalibrator.cxx
  )
endif()

string(REPLACE ".cxx" ".h" HDRS_CINT_O2 "${SRCS}")
set(HDRS_CINT_O2 ${HDRS_CINT_O2} SplineUtil.h devtools/RegularSpline1D.h)

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
                            HEADERS ${HDRS_CINT_O2}
                            LINKDEF TPCFastTransformationLinkDef_O2.h)

  install(FILES ${HDRS_CINT_O2} DESTINATION include/GPU)
  file(COPY ${HDRS_CINT_O2} DESTINATION ${CMAKE_BINARY_DIR}/stage/include/GPU)

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
              CONFIGURATIONS RelWithDebInfo Release MinRelSize)

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

endif()

if(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_ALIROOT_LIB -DGPUCA_NO_FMT)

  set(SRCS ${SRCS} TPCFastTransformManager.cxx TPCFastTransformQA.cxx
           ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
           ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx)
  #set(HDRS_CINT ${HDRS_CINT_O2} TPCFastTransformManager.h TPCFastTransformQA.h )
  set(HDRS_CINT TPCFastTransformManager.h TPCFastTransformQA.h )

  # Enable Vc
  alice_usevc()

  include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
  include_directories(${AliRoot_SOURCE_DIR}/GPU/TPCFastTransformation
                      ${AliRoot_SOURCE_DIR}/GPU/TPCFastTransformation/devtools
                      ${AliRoot_SOURCE_DIR}/GPU/Common
                      ${AliRoot_SOURCE_DIR}/GPU/Utils
                      ${AliRoot_SOURCE_DIR}/HLT/BASE
                      ${AliRoot_SOURCE_DIR}/HLT/TPCLib
                      ${AliRoot_SOURCE_DIR}/TPC/TPCbase
                      ${AliRoot_SOURCE_DIR}/STEER/STEERBase)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary_flat("Ali${MODULE}" "TPCFastTransformationLinkDef_AliRoot.h"
                      "${HDRS_CINT}" "${incdirs}")

  # Generate the ROOT map Dependecies
  set(LIBDEPS STEERBase HLTbase TPCbase)
  generate_rootmap(
    "Ali${MODULE}" "${LIBDEPS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/TPCFastTransformationLinkDef_AliRoot.h")
  # Don't pass Vc to root
  set(LIBDEPS ${LIBDEPS} Vc)

  # Add a library to the project using the specified source files
  add_library_tested(Ali${MODULE} SHARED ${SRCS} G__Ali${MODULE}.cxx)
  target_link_libraries(Ali${MODULE} ${LIBDEPS})

  # Additional compilation flags
  set_target_properties(Ali${MODULE} PROPERTIES COMPILE_FLAGS "")

  # System dependent: Modify the way the library is build
  if(${CMAKE_SYSTEM} MATCHES Darwin)
    set_target_properties(Ali${MODULE}
                          PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
  endif(${CMAKE_SYSTEM} MATCHES Darwin)

  # Installation
  install(TARGETS Ali${MODULE} ARCHIVE DESTINATION lib LIBRARY DESTINATION lib)

  install(FILES ${HDRS_CINT_O2} DESTINATION include)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
endif()

#
# FIXME: this one is misplaced : it depends (at least) on TPCSimulation which is
# built after GPU...
#
# o2_add_test_root_macro(macro/generateTPCCorrectionNTuple.C
# PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation O2::DataFormatsTPC
# O2::TPCSimulation PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/stage/include
# LABELS gpu tpc)
