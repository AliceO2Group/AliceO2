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

add_subdirectory(macros)

o2_add_library(ITSCalibration
               TARGETVARNAME targetName
               SOURCES src/NoiseCalibrator.cxx
               SOURCES src/NoiseSlotCalibrator.cxx
               SOURCES src/NoiseCalibratorSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::DataFormatsITS
                                     O2::DataFormatsDCS
                                     O2::ITSBase
                                     O2::ITSMFTReconstruction
                                     O2::DetectorsCalibration
                                     O2::CCDB)

o2_target_root_dictionary(ITSCalibration
                          HEADERS include/ITSCalibration/NoiseCalibrator.h
                          HEADERS include/ITSCalibration/NoiseSlotCalibrator.h
                          LINKDEF src/ITSCalibrationLinkDef.h)

o2_add_executable(noise-calib-workflow
                  COMPONENT_NAME its
                  SOURCES testWorkflow/its-noise-calib-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::ITSCalibration)

if (OpenMP_CXX_FOUND)
    target_compile_definitions(${targetName} PRIVATE WITH_OPENMP)
    target_link_libraries(${targetName} PRIVATE OpenMP::OpenMP_CXX)
endif()
