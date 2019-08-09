# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE TPCFastTransformation)

set(SRCS
    IrregularSpline1D.cxx
    IrregularSpline2D3D.cxx
    SemiregularSpline2D3D.cxx
    IrregularSpline2D3DCalibrator.cxx
    TPCFastTransformGeo.cxx
    TPCDistortionIRS.cxx
    TPCFastTransform.cxx)

string(REPLACE ".cxx" ".h" HDRS_CINT_O2 "${SRCS}")
set(HDRS_CINT_O2 ${HDRS_CINT_O2} RegularSpline1D.h)

if(${ALIGPU_BUILD_TYPE} STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 SOURCES ${SRCS}
                 PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon Vc::Vc ROOT::Core)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT_O2}
                            LINKDEF TPCFastTransformationLinkDef_O2.h)

  install(FILES ${HDRS_CINT_O2} DESTINATION include/GPU)
  file(COPY ${HDRS_CINT_O2} DESTINATION ${CMAKE_BINARY_DIR}/stage/include/GPU)

  foreach(m
          IrregularSpline1DTest.C
          IrregularSpline2D3DCalibratorTest.C
          IrregularSpline2D3DTest.C
          RegularSpline1DTest.C
          SemiregularSpline2D3DTest.C
          fastTransformQA.C
          generateTPCDistortionNTupleAliRoot.C)
    o2_add_test_root_macro(macro/${m}
                           PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation
                                                 PUBLIC_INCLUDE_DIRECTORIES
                                                 ${CMAKE_BINARY_DIR}/stage/include
                           LABELS gpu tpc)
  endforeach()
endif()

if(${ALIGPU_BUILD_TYPE} STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_ALIROOT_LIB)

  set(SRCS ${SRCS} TPCFastTransformManager.cxx TPCFastTransformQA.cxx
           ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCGeometry.cxx
           ${AliRoot_SOURCE_DIR}/HLT/TPCLib/AliHLTTPCLog.cxx)
  set(HDRS_CINT ${HDRS_CINT} TPCFastTransformManager.h TPCFastTransformQA.h)

  # Enable Vc
  alice_usevc()

  include_directories(SYSTEM ${ROOT_INCLUDE_DIR})
  include_directories(${AliRoot_SOURCE_DIR}/GPU/TPCFastTransformation
                      ${AliRoot_SOURCE_DIR}/GPU/Common
                      ${AliRoot_SOURCE_DIR}/HLT/BASE
                      ${AliRoot_SOURCE_DIR}/HLT/TPCLib
                      ${AliRoot_SOURCE_DIR}/TPC/TPCbase
                      ${AliRoot_SOURCE_DIR}/STEER/STEERBase)

  # Generate the dictionary
  get_directory_property(incdirs INCLUDE_DIRECTORIES)
  generate_dictionary("Ali${MODULE}" "TPCFastTransformationLinkDef_AliRoot.h"
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
# o2_add_test_root_macro(macro/generateTPCDistortionNTuple.C
# PUBLIC_LINK_LIBRARIES O2::TPCFastTransformation O2::DataFormatsTPC
# O2::TPCSimulation PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_BINARY_DIR}/stage/include
# LABELS gpu tpc)
