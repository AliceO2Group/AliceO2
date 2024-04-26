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
  MCHConditions
  SOURCES src/Cathode.cxx
          src/Chamber.cxx
          src/DCSAliases.cxx
          src/DetectionElement.cxx
          src/HVAliases.cxx
          src/LVAliases.cxx
          src/MeasurementType.cxx
          src/Number.cxx
          src/Plane.cxx
          src/Side.cxx
          src/SolarCrate.cxx
  PUBLIC_LINK_LIBRARIES fmt::fmt O2::MCHConstants)

o2_add_executable(
  bad-channels-ccdb
  COMPONENT_NAME mch
  SOURCES src/bad-channels-ccdb.cxx
  PUBLIC_LINK_LIBRARIES
    O2::CCDB
    O2::DataFormatsMCH
    O2::Framework
    O2::MCHMappingImpl4
    O2::MCHRawElecMap
    O2::MCHGlobalMapping
    )

if(BUILD_TESTING)
  o2_add_test(
    dcs-aliases
    SOURCES test/testDCSAliases.cxx
    COMPONENT_NAME mch
    LABELS "muon;mch;dcs"
    PUBLIC_LINK_LIBRARIES O2::MCHConditions)
endif()

