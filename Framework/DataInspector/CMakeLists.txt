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

# Given GCC 7.3 does not provide std::filesystem we use Boost instead
# Drop this once we move to GCC 8.2+
o2_add_library(FrameworkDataInspector
        SOURCES src/Plugin.cxx
        src/DataInspector.cxx
        src/DataInspectorService.cxx
        src/DIMessages.cxx
        src/DISocket.cxx
        PRIVATE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_LIST_DIR}/src
        PUBLIC_LINK_LIBRARIES O2::Framework)