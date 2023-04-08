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

o2_add_library(
  MIDFiltering
  SOURCES src/ChannelMasksHandler.cxx src/ChannelScalers.cxx src/FetToDead.cxx
          src/MaskMaker.cxx src/FiltererBC.cxx src/FiltererBCParam.cxx
  PUBLIC_LINK_LIBRARIES O2::CommonDataFormat O2::DataFormatsMID O2::MIDBase O2::MIDRaw Microsoft.GSL::GSL)

o2_target_root_dictionary(MIDFiltering
  HEADERS include/MIDFiltering/FiltererBCParam.h)

add_subdirectory(exe)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()
