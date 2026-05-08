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

set(MODULE GPUUtils)

set(SRCS
    Spline.cxx
    SplineSpec.cxx
    Spline1DSpec.cxx
    Spline1D.cxx
    Spline2DSpec.cxx
    Spline2D.cxx
)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(SRCS ${SRCS}
      MultivariatePolynomialHelper.cxx
      SplineHelper.cxx
      Spline1DHelper.cxx
      Spline2DHelper.cxx
      Spline1DHelperOld.cxx
      SymMatrixSolver.cxx
      BandMatrixSolver.cxx
  )
endif()

string(REPLACE ".cxx" ".h" HDRS_CINT "${SRCS}")
set(HDRS_CINT ${HDRS_CINT}
              FlatObject.h
              GPUCommonBitSet.h
              SplineUtil.h
)

if(NOT ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  set(HDRS_CINT ${HDRS_CINT}
                MultivariatePolynomial.h
                NDPiecewisePolynomials.h
  )
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 SOURCES ${SRCS}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES O2::GPUCommon
                                       Vc::Vc
                                       ROOT::RIO ROOT::Core ROOT::Matrix ROOT::Tree ROOT::Gpad ROOT::Minuit
  )
  target_include_directories(${targetName}
                             PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                    $<INSTALL_INTERFACE:include/GPU>)

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT}
                            LINKDEF GPUUtilsLinkDef.h)

endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")
  add_library(${MODULE} SHARED ${SRCS})
  set(targetName ${MODULE})
  target_include_directories(${targetName} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  install(TARGETS ${MODULE})
endif()


install(FILES ${HDRS_CINT} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/GPU)
