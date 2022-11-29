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

add_subdirectory(Base)
add_subdirectory(Raw)
add_subdirectory(CTF)

add_subdirectory(Passive) # must be first as some detector's macros use it

add_subdirectory(CTP)
add_subdirectory(PHOS)
add_subdirectory(CPV)
add_subdirectory(EMCAL)
add_subdirectory(FIT)
add_subdirectory(HMPID)
add_subdirectory(TOF)
add_subdirectory(ZDC)

add_subdirectory(ITSMFT)
add_subdirectory(TRD)

add_subdirectory(MUON)

add_subdirectory(TPC)

add_subdirectory(FOCAL)

add_subdirectory(GlobalTracking)
add_subdirectory(GlobalTrackingWorkflow)
add_subdirectory(Vertexing)

if(BUILD_ANALYSIS)
  add_subdirectory(AOD)
endif()

add_subdirectory(Filtering)

add_subdirectory(Calibration)
add_subdirectory(DCS)

add_subdirectory(Align)

if(BUILD_SIMULATION)
  add_subdirectory(gconfig)
  o2_data_file(COPY gconfig DESTINATION Detectors)
endif()

o2_data_file(COPY Geometry DESTINATION Detectors)

if(ENABLE_UPGRADES)
  add_subdirectory(Upgrades)
else()
  message(STATUS "Not building detectors for upgrades")
endif()

add_subdirectory(GRP)
