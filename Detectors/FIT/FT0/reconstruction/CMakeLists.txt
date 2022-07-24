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

o2_add_library(FT0Reconstruction
               SOURCES src/CollisionTimeRecoTask.cxx
                       src/CTFCoder.cxx
                       src/InteractionTag.cxx
               PUBLIC_LINK_LIBRARIES O2::FT0Base
                                     O2::FT0Simulation
                                     O2::DataFormatsFT0
                                     O2::SimulationDataFormat
                                     O2::Framework
                                     O2::DetectorsRaw
                                     O2::CommonDataFormat
                                     O2::rANS
                                     O2::Headers
                                     O2::DetectorsCalibration)

o2_target_root_dictionary(FT0Reconstruction
                          HEADERS include/FT0Reconstruction/CollisionTimeRecoTask.h
                                  include/FT0Reconstruction/InteractionTag.h)

o2_add_executable(
  test-raw-conversion
  COMPONENT_NAME ft0
  SOURCES src/test-raw-conversion.cxx
  PUBLIC_LINK_LIBRARIES O2::FT0Reconstruction)

o2_add_executable(
  test-raw2digit
  COMPONENT_NAME ft0
  SOURCES src/test-raw2digit.cxx
  PUBLIC_LINK_LIBRARIES O2::FT0Reconstruction)
