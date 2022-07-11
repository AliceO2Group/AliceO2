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

o2_add_library(FT0DCSMonitoring
               SOURCES src/FT0DCSDataProcessor.cxx
               PUBLIC_LINK_LIBRARIES O2::FITDCSMonitoring)

o2_target_root_dictionary(FT0DCSMonitoring
                          HEADERS include/FT0DCSMonitoring/FT0DCSConfigReader.h)

o2_add_executable(dcs-sim-workflow
                  COMPONENT_NAME ft0
                  SOURCES workflow/ft0-dcs-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(dcs-workflow
                  COMPONENT_NAME ft0
                  SOURCES workflow/ft0-dcs-data-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DetectorsDCS
                                        O2::Framework
                                        O2::FT0DCSMonitoring)

o2_add_executable(dcs-config-workflow
                  COMPONENT_NAME ft0
                  SOURCES workflow/ft0-dcs-config-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DetectorsCalibration
                                        O2::FITDCSMonitoring
                                        O2::Framework
                                        O2::FT0DCSMonitoring)
