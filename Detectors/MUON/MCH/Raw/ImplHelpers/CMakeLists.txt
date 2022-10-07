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

o2_add_header_only_library(
  MCHRawImplHelpers
  INTERFACE_LINK_LIBRARIES fmt::fmt Microsoft.GSL::GSL O2::MCHConstants
  INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIRECTORY})

o2_add_test(movebuffer
            SOURCES testMoveBuffer.cxx
            PUBLIC_LINK_LIBRARIES O2::MCHRawImplHelpers
            COMPONENT_NAME mchraw
            LABELS "muon;mch;raw")
