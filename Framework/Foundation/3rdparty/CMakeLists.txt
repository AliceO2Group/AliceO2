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

o2_add_header_only_library(FrameworkFoundation3rdparty)

o2_add_library(Catch2
               SOURCES catch2/catch_amalgamated.cxx
               TARGETVARNAME targetName
               PUBLIC_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/catch2)

install(FILES ${CMAKE_CURRENT_LIST_DIR}/catch2/catch_amalgamated.hpp
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
