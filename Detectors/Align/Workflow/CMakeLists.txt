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

o2_add_library(AlignmentWorkflow
               SOURCES src/BarrelAlignmentSpec.cxx
               PUBLIC_LINK_LIBRARIES O2::Align
                                     O2::FrameworkLogger
                                     O2::DataFormatsGlobalTracking
                                     O2::ReconstructionDataFormats
                                     O2::DetectorsCommonDataFormats
                                     O2::DetectorsBase
                                     O2::DataFormatsITSMFT
                                     O2::DataFormatsITS
                                     O2::DataFormatsTPC
                                     O2::DataFormatsTRD
                                     O2::DataFormatsTOF
                                     O2::GPUTracking
                                     O2::TPCCalibration
                                     ROOT::Core
                                     ROOT::Geom
                                     ROOT::MathCore
                                     ROOT::Matrix
                                     ROOT::Hist
                                     ROOT::RIO
                                     ROOT::Tree)

o2_add_executable(alignment-workflow
                  COMPONENT_NAME barrel
                  SOURCES src/barrel-alignment-workflow.cxx
                  PUBLIC_LINK_LIBRARIES O2::Align
                                        O2::AlignmentWorkflow
                                        O2::GlobalTrackingWorkflow
                                        O2::TOFWorkflowIO
                                        )
