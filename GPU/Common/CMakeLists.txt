# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUCommon)

set(HDRS_CINT FlatObject.h)

set(HDRS_INSTALL
    GPUCommonAlgorithm.h
    GPUCommonDef.h
    GPUCommonDefAPI.h
    GPUCommonDefSettings.h
    GPUCommonFairLogger.h
    GPUCommonMath.h
    GPUCommonRtypes.h
    GPUCommonTransform3D.h
    GPUDef.h
    GPUDefConstantsAndSettings.h
    GPUDefGPUParameters.h
    GPUDefOpenCL12Templates.h)

if(ALIGPU_BUILD_TYPE STREQUAL "O2")
  o2_add_library(${MODULE}
                 TARGETVARNAME targetName
                 PUBLIC_LINK_LIBRARIES FairLogger::FairLogger ROOT::RIO
                 PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR})

  o2_target_root_dictionary(${MODULE}
                            HEADERS ${HDRS_CINT}
                            LINKDEF GPUCommonLinkDef.h)

  target_compile_definitions(${targetName} PRIVATE GPUCA_O2_LIB
                             GPUCA_TPC_GEOMETRY_O2 HAVE_O2HEADERS)

  install(FILES ${HDRS_CINT} ${HDRS_INSTALL} DESTINATION include/GPU)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "ALIROOT")
  add_definitions(-DGPUCA_ALIROOT_LIB)

  set(SRCS ${SRCS} GPUCommon.cxx)

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

  install(FILES ${HDRS_CINT} ${HDRS_INSTALL} DESTINATION include)
endif()

if(ALIGPU_BUILD_TYPE STREQUAL "Standalone")

endif()
