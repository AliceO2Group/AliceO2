# Copyright CERN and copyright holders of ALICE O2. This software is distributed
# under the terms of the GNU General Public License v3 (GPL Version 3), copied
# verbatim in the file "COPYING".
#
# See http://alice-o2.web.cern.ch/license for full licensing information.
#
# In applying this license CERN does not waive the privileges and immunities
# granted to it by virtue of its status as an Intergovernmental Organization or
# submit itself to any jurisdiction.

set(MODULE GPUO2Interface)

set(SRCS GPUO2Interface.cxx GPUO2InterfaceRefit.cxx GPUO2InterfaceQA.cxx GPUO2InterfaceConfiguration.cxx)

string(REPLACE ".cxx" ".h" HDRS_TMP "${SRCS}")
set(HDRS ${HDRS_TMP})

o2_add_library(${MODULE}
               SOURCES ${SRCS}
               TARGETVARNAME targetName
               PRIVATE_LINK_LIBRARIES O2::GPUTracking
               PUBLIC_LINK_LIBRARIES O2::GPUCommon
               PUBLIC_INCLUDE_DIRECTORIES ../Definitions
                                          ../DataTypes
                                          ../Interface
                                          ../)

o2_target_root_dictionary(${MODULE}
                          HEADERS ${HDRS}
                          LINKDEF GPUO2InterfaceLinkDef.h)

target_include_directories(${targetName}
                           PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
                                  $<INSTALL_INTERFACE:include/GPU>)

target_compile_definitions(${targetName} PRIVATE $<TARGET_PROPERTY:O2::GPUTracking,COMPILE_DEFINITIONS>)


install(FILES ${HDRS} DESTINATION include/GPU)
