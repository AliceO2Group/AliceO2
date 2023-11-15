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

o2_add_library(MCHGlobalMapping
        SOURCES
          src/ChannelCode.cxx
          src/DsIndex.cxx
          src/HV.cxx
          src/LV.cxx
          src/Mapper.cxx
          src/Quadrant.cxx
          src/Slat.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHRawElecMap
                              O2::MCHMappingInterface
                              O2::MCHConditions
        PRIVATE_LINK_LIBRARIES O2::MCHConstants)

o2_target_root_dictionary(MCHGlobalMapping
                          HEADERS
                            include/MCHGlobalMapping/ChannelCode.h)

o2_add_executable(
        global-mapper
        SOURCES src/global-mapper.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES Boost::program_options
                              O2::DataFormatsMCH
                              O2::MCHGlobalMapping
                              O2::MCHMappingImpl4
                              RapidJSON::RapidJSON)

if(BUILD_TESTING)

  o2_add_test(
    global-mapper
    SOURCES src/testGlobalMapper.cxx
    COMPONENT_NAME mch
    LABELS "muon;mch;dcs"
    PUBLIC_LINK_LIBRARIES O2::MCHGlobalMapping O2::MCHMappingImpl4)

  o2_add_test(
    channelcode
    SOURCES src/testChannelCode.cxx
    COMPONENT_NAME mch
    LABELS muon;mch
    PUBLIC_LINK_LIBRARIES O2::MCHGlobalMapping O2::MCHMappingImpl4)

endif()

