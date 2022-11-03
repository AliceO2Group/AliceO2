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

o2_add_library(EMCALCalibration
               TARGETVARNAME targetName
               SOURCES  src/EMCALChannelData.cxx
                        src/EMCALTimeCalibData.cxx
                        src/EMCALCalibExtractor.cxx
                        src/EMCALCalibParams.cxx
                        src/EMCDCSProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::CCDB O2::EMCALBase
                                     O2::EMCALCalib
                                     O2::CommonUtils
                                     O2::DetectorsCalibration
                                     O2::DetectorsDCS
                                     O2::DataFormatsEMCAL
                                     O2::Framework
                                     O2::Algorithm
                                     Microsoft.GSL::GSL
                                     )
if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_target_root_dictionary(EMCALCalibration
                          HEADERS include/EMCALCalibration/EMCALCalibExtractor.h
                                  include/EMCALCalibration/EMCALChannelCalibrator.h
                                  include/EMCALCalibration/EMCALChannelData.h
                                  include/EMCALCalibration/EMCALTimeCalibData.h
                                  include/EMCALCalibration/EMCALCalibParams.h
                                  include/EMCALCalibration/EMCDCSProcessor.h
                          LINKDEF src/EMCALCalibrationLinkDef.h)

o2_add_executable(emcal-channel-calib-workflow
                  COMPONENT_NAME calibration
                  SOURCES testWorkflow/emc-channel-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::EMCALCalibration
                                        O2::DetectorsRaw
                                        O2::DetectorsCalibration)
if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_add_executable(run-calib-offline
                  COMPONENT_NAME emcal
                  TARGETVARNAME targetName
                  SOURCES run/runCalibOffline.cxx
                  PUBLIC_LINK_LIBRARIES O2::EMCALCalibration)


if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()

o2_add_test_root_macro(macros/makeEMCALCCDBEntryForDCS.C
                                PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::CCDB
                                LABELS emcal COMPILE_ONLY)

o2_add_test_root_macro(macros/readEMCALDCSentries.C
                                PUBLIC_LINK_LIBRARIES O2::DetectorsDCS O2::CCDB O2::EMCALCalibration
                                LABELS emcal COMPILE_ONLY)