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

o2_add_library(MCHStatus
        SOURCES
          src/StatusMap.cxx
          src/StatusMapCreatorParam.cxx
          src/StatusMapCreatorSpec.cxx
        PUBLIC_LINK_LIBRARIES
          fmt::fmt
          O2::DataFormatsMCH
          O2::MCHGlobalMapping)

o2_target_root_dictionary(MCHStatus
                          HEADERS
                            include/MCHStatus/StatusMap.h
                            include/MCHStatus/StatusMapCreatorParam.h)

o2_add_executable(
  statusmap-creator-workflow
  COMPONENT_NAME mch
  SOURCES src/statusmap-creator-workflow.cxx
  PUBLIC_LINK_LIBRARIES
    O2::CCDB
    O2::Framework
    O2::MCHMappingImpl4
    O2::MCHStatus
    )

if(BUILD_TESTING)

  o2_add_test(
    statusmap
    SOURCES src/testStatusMap.cxx
    COMPONENT_NAME mch
    LABELS "muon;mch"
    PUBLIC_LINK_LIBRARIES
      O2::MCHGlobalMapping
      O2::MCHMappingImpl4
      O2::MCHStatus
      )

endif()

