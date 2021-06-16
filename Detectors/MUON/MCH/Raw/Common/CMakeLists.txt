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

o2_add_library(MCHRawCommon
               SOURCES src/CoDecParam.cxx
                       src/DataFormats.cxx
                       src/SampaBunchCrossingCounter.cxx
                       src/SampaCluster.cxx
                       src/SampaHeader.cxx
               PUBLIC_LINK_LIBRARIES fmt::fmt Microsoft.GSL::GSL Boost::boost
                                     O2::Headers O2::CommonConstants
                                     O2::DetectorsRaw
               PRIVATE_LINK_LIBRARIES O2::MCHRawImplHelpers)

o2_target_root_dictionary(MCHRawCommon
                          HEADERS include/MCHRawCommon/CoDecParam.h)
if(BUILD_TESTING)
        add_subdirectory(test)
endif()
