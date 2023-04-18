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

# @brief  cmake setup for the DataFormats module of AliceO2

add_subdirectory(Common)
add_subdirectory(TPC)
add_subdirectory(ITSMFT)
add_subdirectory(MUON)
add_subdirectory(TOF)
add_subdirectory(FIT)
add_subdirectory(FOCAL)
add_subdirectory(HMPID)
add_subdirectory(EMCAL)
add_subdirectory(ZDC)
add_subdirectory(TRD)
add_subdirectory(PHOS)
add_subdirectory(CPV)
add_subdirectory(CTP)
add_subdirectory(GlobalTracking)
add_subdirectory(DCS)

if(ENABLE_UPGRADES)
  add_subdirectory(Upgrades)
else()
  message(STATUS "Not building dataformats for upgrades")
endif()
