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

o2_add_library(MCHPreClustering
        SOURCES src/PreClusterFinder.cxx
        src/PreClusterFinderMapping.cxx
        src/PreClusterFinderParam.cxx
        src/PreClusterFinderSpec.cxx
        PUBLIC_LINK_LIBRARIES O2::MCHMappingInterface O2::MCHBase O2::Framework)

o2_target_root_dictionary(MCHPreClustering
        HEADERS include/MCHPreClustering/PreClusterFinderParam.h)

o2_add_executable(
        digits-to-preclusters-workflow
        SOURCES src/digits-to-preclusters-workflow.cxx
        COMPONENT_NAME mch
        PUBLIC_LINK_LIBRARIES O2::MCHPreClustering O2::MCHMappingImpl4)
