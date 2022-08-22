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

o2_add_library(GRPCalibrationWorkflow
  TARGETVARNAME targetName
               SOURCES src/GRPLHCIFfileSpec.cxx
                       src/GRPDCSDPsSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Framework
                                     O2::Headers
                                     O2::CCDB
                                     O2::CommonUtils
                                     Microsoft.GSL::GSL
                                     O2::GRPCalibration
                                     O2::DataFormatsParameters
                                     O2::DetectorsCalibration)

o2_add_executable(grp-lhc-if-file-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/gpr-lhc-if-file-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                                        O2::CCDB
                                        O2::GRPCalibrationWorkflow
                                        O2::GRPCalibration
                                        O2::DetectorsCalibration)

o2_add_executable(grp-create
                  COMPONENT_NAME ecs
                  SOURCES src/create-grp-ecs.cxx
                  PUBLIC_LINK_LIBRARIES O2::DetectorsCommonDataFormats
                                        O2::DataFormatsParameters
                                        O2::CommonUtils
                                        O2::CCDB
                                        Boost::program_options)

o2_add_executable(workflow
                  COMPONENT_NAME create-aligned-geometry
                  SOURCES src/create-aligned-geometry.cxx
                  PUBLIC_LINK_LIBRARIES O2::DetectorsBase
                                        O2::CommonUtils
                                        O2::CCDB
                                        O2::DetectorsRaw
                                        Boost::program_options)

o2_add_executable(grp-dcs-dps-sim-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/grp-dcs-dps-sim-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::DCStestWorkflow)

o2_add_executable(grp-dcs-dps-workflow
                  COMPONENT_NAME calibration
                  SOURCES src/grp-dcs-dps-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Framework
                  O2::CCDB
                  O2::GRPCalibrationWorkflow
                  O2::GRPCalibration
                  O2::DetectorsCalibration)
