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

o2_add_library(MCHTimeClustering
        SOURCES src/ROFTimeClusterFinder.cxx src/TimeClusterizerParam.cxx src/TimeClusterFinderSpec.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHBase O2::Framework O2::MCHDigitFiltering O2::MCHROFFiltering)

o2_target_root_dictionary(MCHTimeClustering HEADERS include/MCHTimeClustering/TimeClusterizerParam.h)

o2_add_executable(
        digits-to-timeclusters-workflow
        SOURCES src/digits-to-timeclusters-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHTimeClustering)

if(BUILD_TESTING)

        o2_add_test(time-cluster-finder
                SOURCES src/testROFTimeClusterFinder.cxx
                COMPONENT_NAME mchtimeclustering
                LABELS "muon;mch;timeclustering"
                PUBLIC_LINK_LIBRARIES O2::MCHRawDecoder O2::MCHTimeClustering Boost::boost)

endif()
