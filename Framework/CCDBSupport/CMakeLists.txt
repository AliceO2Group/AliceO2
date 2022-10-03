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
o2_add_library(FrameworkCCDBSupport
               SOURCES 
                src/Plugin.cxx
                src/CCDBHelpers.cxx
               PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
               PUBLIC_LINK_LIBRARIES O2::Framework O2::CCDB)

o2_add_test(CCDBHelpers NAME test_Framework_test_CCDBHelpers
            SOURCES test/test_CCDBHelpers.cxx
            COMPONENT_NAME Framework
            LABELS framework
            PUBLIC_LINK_LIBRARIES O2::Framework O2::FrameworkCCDBSupport)
