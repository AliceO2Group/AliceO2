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


o2_add_executable(alpide-response-generator
                  SOURCES AlpideResponse.cxx
                  PUBLIC_LINK_LIBRARIES O2::ITSMFTSimulation
                                        ROOT::Core
                  TARGETVARNAME targetName)

set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/data/AlpideResponseData/AlpideResponse.cxx)

if(ITSRESPONSE)
  message(STATUS "ITSRESPONSE option provided, setting ITSRESPONSE_DIR from it: " ${ITSRESPONSE})
  set(ITSRESPONSE_DIR ${ITSRESPONSE})
else()
  message(STATUS "ITSRESPONSE option not provided, setting ITSRESPONSE_DIR from environment ITSRESPONSE_ROOT: " $ENV{ITSRESPONSE_ROOT})
  set(ITSRESPONSE_DIR $ENV{ITSRESPONSE_ROOT})
endif()

add_custom_command(TARGET O2exe-alpide-response-generator POST_BUILD
                   COMMAND ${CMAKE_BINARY_DIR}/stage/bin/o2-alpide-response-generator -i ${ITSRESPONSE_DIR}/response/AlpideResponseData/ -o ${CMAKE_CURRENT_BINARY_DIR}/
                   DEPENDS ${CMAKE_SOURCE_DIR}/Detectors/ITSMFT/common/data/AlpideResponseData/AlpideResponse.cxx
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/AlpideResponseData.root" DESTINATION "${CMAKE_INSTALL_PREFIX}/share/Detectors/ITSMFT/data/AlpideResponseData/")
