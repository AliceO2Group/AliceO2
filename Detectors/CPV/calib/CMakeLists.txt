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

add_subdirectory(CPVCalibWorkflow)

o2_add_library(CPVCalibration
               TARGETVARNAME targetName
               SOURCES src/NoiseCalibrator.cxx
                       src/PedestalCalibrator.cxx
                       src/GainCalibrator.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsCPV
                                     O2::CPVBase
                                     O2::DetectorsCalibration
                                     O2::CCDB)

o2_target_root_dictionary(CPVCalibration
                          HEADERS include/CPVCalibration/NoiseCalibrator.h
                                  include/CPVCalibration/PedestalCalibrator.h
                                  include/CPVCalibration/GainCalibrator.h)

o2_add_executable(cpv-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/cpv-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::CPVCalibration
                                        O2::DetectorsCalibration
                                        O2::CPVBase
                                        O2::DataFormatsCPV)

if(BUILD_TESTING)
  o2_add_test_root_macro(macros/readPedestalsFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/readBadChannelMapFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/readCalibParamsFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/readGainCalibDataFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::CPVCalibration
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/readPedEfficienciesFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/readThresholdsFEEFromCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/makeBadMapFromPedestalRun.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::CPVBase O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro(macros/PostBadMapCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

  o2_add_test_root_macro( macros/PostCalibCCDB.C
                         PUBLIC_LINK_LIBRARIES O2::CCDB O2::DataFormatsCPV
                         LABELS CPV COMPILE_ONLY)

endif()

